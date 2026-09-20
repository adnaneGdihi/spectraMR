"""Tier-2 synthetic forward probe — audit-ladder spec, 2026-05-03.

Catches at config-load time:

  * Shape mismatches inside the model (linear-layer mat1xmat2 errors,
    transposed-conv resolution mismatches).
  * AMP / GradScaler double-unscale traps (the wrapper actually
    triggers under a real ``GradScaler.step()``).
  * Output-vs-target shape contracts on the loss path.
  * OOM at the configured batch / patch size (when run on GPU).

The probe instantiates the model, runs ONE forward + backward pass on
a dummy tensor matching the dataset's promised output shape, drops
everything, and returns a JSON-serialisable :class:`ProbeResult`. No
data loaders, no optimizer state, no full pipeline.

CPU is the default for safety in CI; pass ``device="cuda"`` to opt
into the OOM-detection benefit.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from spectramr.infrastructure.builders.generator_kwargs import (
    apply_gradient_checkpointing,
    resolve_contract,
    resolve_full_generator_kwargs,
)

logger = logging.getLogger(__name__)


__all__ = ["ProbeResult", "synthetic_forward_probe"]


@dataclass
class ProbeResult:
    """Outcome of a single :func:`synthetic_forward_probe` invocation.

    JSON-serialisable so the audit aggregator and the campaign
    leaderboard can store / group probe failures the same way they
    handle :class:`HealthCheckResult` records.
    """

    passed: bool
    category: str  # 'forward_pass_shape' / 'sample_path_shape' / 'amp_double_unscale' / 'oom' / 'instantiation' / 'no_probe'
    message: str
    severity: str = "error"  # 'error' / 'warning' / 'info'
    yaml_keys: list[str] = field(default_factory=list)
    fix_hint: str | None = None
    device: str = "cpu"
    elapsed_seconds: float = 0.0
    traceback: str | None = None

    @property
    def accelerated(self) -> bool:
        """Did the probe actually exercise an accelerator?

        A CPU probe cannot catch CUDA OOM or AMP/GradScaler traps — the two
        failures the probe exists to catch — so a passing CPU probe is a far
        weaker claim than a passing GPU one. The audit CLI gates on this
        (``tier2_probe_accelerated``); this property is the SSOT for the answer.
        """
        return not str(self.device).startswith("cpu")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


_GRAD_EXPLOSION_THRESHOLD = 1.0e6  # |grad|_2 > this ⇒ explosion
_IDENTITY_COLLAPSE_EPSILON = 1.0e-3  # |y - x| / |x| < this ⇒ identity
_INPUT_INVARIANT_EPSILON = 1.0e-4  # |y(x1)-y(x2)| / |y| < this ⇒ output ignores input


def _input_invariance_stats(y, y_same, y_diff) -> tuple[float, float]:
    """Relative L1 spread of the output under (re-run, changed-input).

    ``d_same`` = ||y - f(x)|| / ||y||  — the stochastic noise floor (re-running the
    SAME input; ~0 for a deterministic model, >0 under dropout/internal sampling).
    ``d_diff`` = ||y - f(x2)|| / ||y|| — sensitivity to a DIFFERENT input.
    A facade that ignores its input has d_diff ≈ d_same (changing the input changes
    the output no more than re-running does).
    """
    scale = y.detach().abs().mean().clamp_min(1.0e-12)
    d_same = float((y.detach() - y_same.detach()).abs().mean() / scale)
    d_diff = float((y.detach() - y_diff.detach()).abs().mean() / scale)
    return d_same, d_diff


def _is_input_invariant(
    d_same: float, d_diff: float, eps: float = _INPUT_INVARIANT_EPSILON
) -> bool:
    """Classify an output as input-invariant (measurement-independent / facade).

    True when changing the input barely moves the output: either the absolute
    spread is below ``eps`` (deterministic facade), or — for a stochastic model
    with a real noise floor — the input-change spread is no larger than ~1.5x the
    re-run spread (the input contributes no signal beyond internal randomness).
    """
    return d_diff < eps or (d_same >= eps and (d_diff / max(d_same, 1.0e-12)) < 1.5)


def _coerce_probe_skip(raw: object) -> set[str]:
    """Normalise a model's ``synthetic_forward_probe_skip`` into a ``set``.

    The contract is ``set[str]`` (probe categories to suppress) but the field
    is hand-declared on models and at least one (``bloch_manifold_projector``)
    declares it ``bool``. The old ``"identity_collapse" not in <flag>`` test
    raised an *uncaught* ``TypeError`` on a bool, crashing the whole audit.
    A truthy bool means "skip the probe checks"; a falsy value means none.
    """
    if isinstance(raw, bool):
        return {"identity_collapse"} if raw else set()
    if not raw:
        return set()
    if isinstance(raw, str):
        return {raw}
    return set(raw)


def _declared_schedule_length(config: object) -> int | None:
    """The arm's diffusion schedule length, or ``None`` if it declares none.

    Two config locations are legitimate and BOTH are in live use, which is why
    reading only one of them silently under-reports:

    * ``training.diffusion.timesteps`` — the strategy-side schedule, read by
      :class:`DiffusionTrainingStrategy` to sample ``t`` per step.
    * ``model.model_kwargs.timesteps`` — the schedule baked into the generator
      at construction. The k-space cold-diffusion arms declare it ONLY here
      (``experiment_11_attention_none`` sets ``model_kwargs.timesteps: 28`` with
      no ``training.diffusion`` block at all), so a reader that consults the
      strategy path alone concludes "no schedule" for the entire cohort.

    Returns ``None`` — not a fallback integer — when neither answers, so callers
    can distinguish "this arm has no diffusion schedule" from "its schedule is
    short". Conflating those is what makes an optional ``timesteps`` kwarg
    impossible to gate correctly.
    """
    candidates = (
        getattr(
            getattr(getattr(config, "training", None), "diffusion", None),
            "timesteps",
            None,
        ),
        (getattr(getattr(config, "model", None), "model_kwargs", None) or {}).get("timesteps")
        if isinstance(getattr(getattr(config, "model", None), "model_kwargs", None), dict)
        else None,
    )
    for candidate in candidates:
        try:
            n_int = int(candidate)
        except (TypeError, ValueError):
            continue
        if n_int > 0:
            return n_int
    return None


def _resolve_probe_timestep(config: object) -> int:
    """Pick a representative (mid-schedule) diffusion timestep for the probe.

    Reads the configured schedule length via :func:`_declared_schedule_length`
    and returns its midpoint, so the time-embedding path is actually exercised.
    When the length is undiscoverable we fall back to ``1`` — still non-trivial
    (unlike t=0) and guaranteed in-range for any real schedule (so a
    learned-embedding table is never indexed out of bounds).

    Both original candidates were dead, so this ALWAYS returned the fallback:
    ``training.diffusion.num_timesteps`` is the retired spelling (it folds to
    ``.timesteps``, so nothing ever answers to it post-load) and
    ``training.num_timesteps`` is a path no schema has ever carried. Every
    diffusion arm was therefore probed at t=1 rather than mid-schedule — the
    gate did not fail, it just stopped testing what this docstring claims.
    No legacy fallback replaces them: the fold happens at load, so an arm
    declaring the old spelling already arrives here as ``.timesteps``.

    Fixing those two left a THIRD blind spot with the same symptom: reading only
    ``training.diffusion.timesteps`` still returns 1 for every arm that declares
    its schedule in ``model.model_kwargs`` instead — the whole k-space
    cold-diffusion cohort. ``experiment_11_attention_none`` declares 28 there
    and was probed at t=1; the midpoint is 14.
    """
    n = _declared_schedule_length(config)
    if n is None:
        return 1
    return n // 2 if n > 1 else 1


# Parameter-name vocabularies for the best-effort ``sample()`` probe. A
# generative model's reverse process exposes ONE of these step-count knobs;
# we require one so the probe is cost-bounded (set to 2) — a 1000-step
# default sampler would otherwise hang the mandatory smoke gate.
_SAMPLE_STEP_PARAMS = frozenset(
    {
        "num_steps",
        "num_inference_steps",
        "n_steps",
        "steps",
        "sampling_steps",
        "num_sampling_steps",
        "n_timesteps",
    }
)
_SAMPLE_BATCH_PARAMS = frozenset({"batch_size", "n", "num_samples", "n_samples", "num"})
_SAMPLE_SHAPE_PARAMS = frozenset({"shape", "size", "image_shape", "out_shape"})
_SAMPLE_COND_PARAMS = frozenset(
    {
        "cond",
        "condition",
        "cond_image",
        "context",
        "x",
        "y",
        "image",
        "source",
        "src",
        "lr",
        "lr_image",
        "observation",
        "measurement",
        "noisy_images",
    }
)


def _probe_sample_path(
    model: Any, x: Any, expected_spatial_rank: int, batch_size: int, device: str
) -> tuple[bool, str]:
    """Best-effort probe of a generative ``sample()`` path.

    ``forward()`` exercises the training step; many diffusion / latent models
    GENERATE through a separate ``sample()`` whose shape contract ``forward``
    never touches (a different decoder, an upsampling head, a latent→image
    projection). A shape bug there is otherwise invisible until the first
    validation image.

    Safety policy (this runs inside the MANDATORY smoke gate, so it must never
    hang or false-positive):

    * **Cost-bounded** — only runs when the signature exposes a step-count
      kwarg we can pin to ``2``; otherwise returns ``(True, note)`` ("couldn't
      bound, skipped"), never an unbounded reverse process.
    * **Fail only on a concrete mismatch** — a *successful* ``sample()`` call
      whose output spatial rank differs from ``forward``'s is a real bug
      → ``(False, reason)``. An unrecognised required arg, an unintrospectable
      signature, a non-tensor output, or any raised exception → ``(True, note)``
      (the synthetic args, not the model, are the likely culprit).

    Returns ``(ok, note)``; the caller fails the probe only on ``ok is False``.
    """
    import inspect

    import torch

    sample_fn = getattr(model, "sample", None)
    if not callable(sample_fn):
        return True, ""
    try:
        sig = inspect.signature(sample_fn)
    except (TypeError, ValueError):
        return True, " | sample(): unintrospectable signature, skipped"

    kwargs: dict[str, object] = {}
    has_step_bound = False
    for pname, p in sig.parameters.items():
        if pname == "self" or p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        plow = pname.lower()
        if plow in _SAMPLE_STEP_PARAMS:
            kwargs[pname] = 2  # pin tiny so the reverse process is cheap
            has_step_bound = True
            continue
        # Beyond the step count, only fill REQUIRED args (no default).
        if p.default is not inspect.Parameter.empty:
            continue
        if plow in _SAMPLE_BATCH_PARAMS:
            kwargs[pname] = batch_size
        elif plow in _SAMPLE_SHAPE_PARAMS:
            kwargs[pname] = tuple(x.shape)
        elif plow in _SAMPLE_COND_PARAMS:
            kwargs[pname] = x
        elif "timestep" in plow or plow in ("t", "ts"):
            kwargs[pname] = torch.full((batch_size,), 1, dtype=torch.long, device=device)
        else:
            return True, f" | sample(): required arg {pname!r} unrecognised, skipped"

    if not has_step_bound:
        return True, " | sample(): no step-count kwarg to bound cost, skipped"

    try:
        with torch.no_grad():
            out = sample_fn(**kwargs)
    except Exception as exc:
        # A raised sample() is treated as "couldn't probe" (the synthetic args,
        # not the model, are the likely cause) — never a probe failure.
        return True, f" | sample(): raised {type(exc).__name__}, not probed"

    if isinstance(out, tuple):
        out = out[0]
    elif isinstance(out, dict):
        out = next(iter(out.values()), None)
    ndim = getattr(out, "ndim", None)
    if not isinstance(ndim, int):
        return True, " | sample(): non-tensor output, not probed"

    shape = tuple(getattr(out, "shape", ()))
    got_rank = ndim - 2 if ndim >= 2 else ndim
    if got_rank != expected_spatial_rank:
        return False, (
            f"sample() output spatial rank {got_rank} (shape {shape}) != "
            f"forward() spatial rank {expected_spatial_rank}. The generation "
            "path produces a different dimensionality than training — validation "
            "images / metrics will crash or be silently wrong."
        )
    return True, f" | sample() OK ({shape})"


# Sentinel: the configured-loss probe does not apply to this config (not a
# declarative image-loss reconstruction arm, or the loss path could not be
# built). Distinct from a computed total of ``None`` (which IS a dead signal).
_LOSS_PROBE_SKIP = object()


def _loss_gradient_verdict(
    total: Any, model_type: str, device: str, elapsed: float
) -> ProbeResult | None:
    """Verdict on the configured ``g_total_loss`` at iteration 0.

    Returns a ``dead_loss`` :class:`ProbeResult` when the total carries no usable
    gradient — ``None`` total, exactly ``0.0``, or ``requires_grad=False`` — and
    ``None`` (continue) when it is healthy. A NaN/Inf total is deferred to the
    existing ``nan_in_loss`` path (returns ``None``). This is the warmup-gate
    dead_loss class (cs_mno cohort, 2026-06-27): l1 held at weight 0 while the
    only other finite loss is NaN-skipped → total 0.0 → zero gradient.
    """
    import torch

    if isinstance(total, torch.Tensor):
        if not torch.isfinite(total).all():
            return None  # NaN/Inf — handled by the dummy-L1 nan_in_loss branch
        if float(total.detach()) != 0.0 and total.requires_grad:
            return None  # healthy: finite, nonzero, grad-carrying
    elif total is not None:
        return None  # non-tensor — cannot judge, leave to other checks
    return ProbeResult(
        passed=False,
        category="dead_loss",
        message=(
            f"{model_type}: the CONFIGURED loss g_total_loss is "
            f"{'None' if total is None else '0.0'} at iteration 0 — zero gradient, "
            "the model would not train. Most often the spatial-loss warmup gate "
            "(losses.reconstruction.warmup_iterations) holds the only active loss "
            "(l1) at weight 0 while a sibling loss is NaN-skipped on the model's "
            "off-scale output."
        ),
        severity="error",
        yaml_keys=["losses.reconstruction.warmup_iterations", "losses.image_losses"],
        fix_hint=(
            "Set losses.reconstruction.warmup_iterations: 0 (operator/deterministic "
            "arms need no warmup), OR add a non-gated loss that survives at iter 0, "
            "OR fix the model output scale so the secondary loss does not go NaN."
        ),
        device=device,
        elapsed_seconds=elapsed,
    )


def _configured_loss_total(config: Any, y: Any, target: Any) -> Any:
    """Run the configured reconstruction loss at iteration 0 → total, or skip.

    Returns the ``g_total_loss`` tensor (which may be a zero-gradient leaf — the
    dead_loss signal) for declarative image-loss reconstruction arms, or
    :data:`_LOSS_PROBE_SKIP` for any inapplicable config or build error. NEVER
    raises — the probe must stay tolerant.
    """
    try:
        losses_block = getattr(config, "losses", None)
        image_losses = getattr(losses_block, "image_losses", None)
        if not image_losses:
            return _LOSS_PROBE_SKIP
        out_domain = losses_block.policy.output_domain if losses_block else "image"
        if out_domain not in (None, "image"):
            return _LOSS_PROBE_SKIP  # k-space/complex loss paths not modelled here
        from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
            UnifiedReconstructionLossComputer,
        )
        from spectramr.models.losses.registry import create_loss

        losses_dict = {}
        for lc in image_losses:
            if isinstance(lc, dict):
                name, enabled = lc.get("name"), lc.get("enabled", True)
            else:
                name, enabled = getattr(lc, "name", None), getattr(lc, "enabled", True)
            if name and enabled:
                # A loss that can't be built skips the whole check (the outer
                # except returns _LOSS_PROBE_SKIP) — the runtime DEAD LOSS guard
                # backstops. Better than silently dropping one loss and judging
                # on a partial objective.
                #
                # ``.to(y.device)`` is what makes that sentence true. ``create_loss``
                # returns the module on CPU, while ``pred``/``target`` live on the
                # probe's device — so every loss carrying a parameter or buffer
                # (``perceptual``'s VGG19 and its ImageNet mean/std, ``lpips``,
                # ``dino_perceptual``, ``gram_style``, …) raised a device
                # ``RuntimeError`` inside ``compute``. That is NOT caught by the
                # outer ``except`` here: the computer downgrades a non-ValueError
                # to ``logger.warning`` and drops the term, so the probe judged the
                # arm on exactly the partial objective this comment refuses — and
                # only ever on a non-CPU run, which is why CI never saw it.
                losses_dict[name] = create_loss(name).to(y.device)
        if not losses_dict:
            return _LOSS_PROBE_SKIP
        comp = UnifiedReconstructionLossComputer(config=config, device=y.device)
        out = comp.compute(pred=y, target=target, epoch=0, iteration=0, losses_dict=losses_dict)
        return out.total
    except Exception:
        return _LOSS_PROBE_SKIP


def synthetic_forward_probe(
    config: Any,
    device: str = "cpu",
    backward: bool = True,
    use_phantom: bool = True,
    save_images_dir: str | None = None,
    arm_name: str = "probe",
) -> ProbeResult:
    """Run one forward (+ optional backward) pass on a synthetic input.

    Args:
        config: A loaded :class:`TrainingSettings` instance (or any
            object exposing ``model.model_type``, ``model.in_channels``,
            ``model.out_channels``, ``model.model_kwargs``, and
            ``data.patch_size`` / ``data.batch_size``).
        device: ``"cpu"`` (default, CI-safe) or ``"cuda"`` (catches OOM
            at the actual configured batch / patch size).
        backward: If True (default), runs an L1 dummy loss + backward
            so AMP / GradScaler interactions actually fire.
        use_phantom: If True (default), the dummy input/target are
            built from a Shepp-Logan phantom rather than ``randn``.
            Recognisable structure makes saved probe images visually
            inspectable — you can immediately see whether the model
            is preserving anatomy, scrambling channels, or mis-handling
            the k-space ↔ image-domain conversion. Set to ``False`` for
            pure-noise tests of the gradient path.
        save_images_dir: If non-None, write input / output / target
            mosaics (PNG) into this directory after the forward pass.
            Useful in the smoke wrapper to produce inspectable
            artefacts for every audited arm.
        arm_name: Filename prefix for the saved images (one PNG per
            role: ``<arm>_input.png``, ``<arm>_output.png``,
            ``<arm>_target.png``). Default ``"probe"``.

    Returns:
        :class:`ProbeResult` describing the outcome.

    The probe deliberately does no fallback / autocorrection. If the
    model fails to instantiate or fails the forward pass, the exception
    type drives the ``category`` field. The traceback is captured for
    the audit log but not printed by default.
    """
    import time

    t0 = time.perf_counter()

    try:
        import torch
    except ImportError:
        return ProbeResult(
            passed=True,
            category="no_probe",
            message="PyTorch unavailable — Tier-2 probe skipped.",
            severity="info",
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ── Shape contract ────────────────────────────────────────────────
    try:
        in_channels = int(getattr(config.model, "in_channels", 1))
        out_channels = int(getattr(config.model, "out_channels", in_channels))
        model_type = getattr(config.model, "model_type", None)
    except AttributeError as exc:
        return ProbeResult(
            passed=False,
            category="instantiation",
            message=f"config.model is missing required attributes: {exc}",
            severity="error",
            yaml_keys=["model.in_channels", "model.out_channels", "model.model_type"],
            fix_hint="Ensure the model section declares model_type, in_channels, out_channels.",
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
        )

    if model_type is None:
        return ProbeResult(
            passed=False,
            category="instantiation",
            message="model.model_type is not set.",
            severity="error",
            yaml_keys=["model.model_type"],
            fix_hint="Set model.model_type to a registered model name.",
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
        )

    patch_size = list(config.data.sampling.patch_size or []) if hasattr(config, "data") else []
    if not patch_size or not all(isinstance(s, int) and s > 0 for s in patch_size):
        # Default to a tiny 2D patch to keep probes cheap.
        patch_size = [32, 32]
    # ``patch_size: [H, W, 1]`` (or [L, 1, 1] for 1D PDE benchmarks) is
    # the canonical "lower-dim-in-3D-pipeline" convention — TorchIO
    # patches always have a depth axis even when depth=1. Probing a 2D
    # model with a 5D ``[B, C, H, W, 1]`` tensor crashes ``conv2d``
    # ("Expected 4D input ..."); a 1D model with [B, C, L, 1, 1]
    # similarly fails. Strip *all* trailing singleton axes so the probe
    # materialises a tensor of the right rank. Real 3D probes still
    # work (``[H, W, D]`` with D>1 is preserved).
    while len(patch_size) > 1 and patch_size[-1] == 1:
        patch_size = patch_size[:-1]

    # Clamp batch to 2 — enough to exercise BN/contrast paths but cheap.
    batch_size = int(config.data.loader.batch_size) if hasattr(config, "data") else 1
    batch_size = max(1, min(2, batch_size))

    # ── Contract-aware spatial rank (Layer-3 ↔ Layer-1 bridge) ────────
    # Read the model's DECLARED ModelCapabilities (Layer 1) and let it steer
    # the synthetic shape (Layer 3). Without this the probe always runs at
    # whatever rank survives the singleton-strip above, so a model that
    # declares ``spatial_dims=(3,)`` and is fed a ``[H, W, 1]`` TorchIO patch
    # (stripped to 2D) is probed at 2D — it then either crashes conv3d
    # misleadingly or, worse, silently passes a degenerate 2D fallback. The
    # whole point of declaring a contract is to exercise the path it names.
    #
    # Reconciliation is **expand-only**: we lift the rank UP to the smallest
    # declared rank when the (stripped) patch is too low, but we never collapse
    # a genuinely-higher-rank patch DOWN — a 2D-declared model fed real 3D data
    # is a real bug, and the probe must let that surface (forward crash), not
    # hide it. Lazy import so the test fixtures that patch the registry module
    # take effect (mirrors the get_model_class import below).
    declared_caps = None
    try:
        from spectramr.models.registry import get_model_capabilities

        declared_caps = get_model_capabilities(model_type)
    except Exception:  # pragma: no cover — never fail the probe on introspection
        declared_caps = None

    if declared_caps is not None and getattr(declared_caps, "spatial_dims", None):
        allowed_ranks = tuple(int(d) for d in declared_caps.spatial_dims)
        cur_rank = len(patch_size)
        if cur_rank not in allowed_ranks and cur_rank < min(allowed_ranks):
            target_rank = min(allowed_ranks)
            patch_size = patch_size + [8] * (target_rank - cur_rank)

    # ── Model instantiation ───────────────────────────────────────────
    try:
        from spectramr.models.registry import get_model_class

        cls = get_model_class(model_type)
    except Exception as exc:
        return ProbeResult(
            passed=False,
            category="instantiation",
            message=f"get_model_class({model_type!r}) failed: {exc}",
            severity="error",
            yaml_keys=["model.model_type"],
            fix_hint=f"Ensure {model_type!r} is registered with @register_model.",
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
            traceback=traceback.format_exc(),
        )

    # Kwarg resolution is the SSOT shared with training (ModelBuilder ->
    # GeneratorBuilder). The probe used to assemble its own ``ctor_kwargs``
    # and apply its own signature filter -- its comment admitted it duplicated
    # "the same filter in ModelFactory.create_generator" -- so ``audit
    # --probe`` validated a differently-constructed model than training
    # builds, and an arm could pass the probe and still diverge in training.
    # Three contract-gated SSOT injections never reached the probed model at
    # all: acceleration_config, kspace_log_scaled and physics.data_consistency.
    # ``device`` is the fourth (step 3d) and is passed here for the same reason:
    # the probe must construct the model training constructs, and on the probe's
    # own device -- otherwise a probe run under ``--device cuda`` would exercise
    # the CPU mask path while training exercises the table path (#1508).
    try:
        resolved = resolve_full_generator_kwargs(config, model_cls=cls, device=device)
    except ValueError as exc:
        return ProbeResult(
            passed=False,
            category="instantiation",
            message=f"config resolution for {model_type} failed: {exc}",
            severity="error",
            yaml_keys=[
                "model.model_kwargs",
                "data.processing.enable_log_scaling",
                "physics.data_consistency",
            ],
            fix_hint=(
                "A model_kwargs entry contradicts the config block that owns "
                "that fact. Remove the model_kwargs copy and let the owning "
                "block be the single source of truth."
            ),
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
            traceback=traceback.format_exc(),
        )

    try:
        from spectramr.models.factories.model_factory import ModelFactory, ModelRegistry

        if ModelRegistry().has_generator(model_type):
            # The training path. Parameter mapping, signature filtering and
            # diffusion-wrapper inner-module construction all live in the
            # factory; the probe no longer reimplements any of them.
            model = ModelFactory().create_generator(
                model_type,
                in_channels=in_channels,
                out_channels=out_channels,
                _declared_keys=resolved.declared_keys,
                **resolved.kwargs,
            )
        else:
            # ``get_model_class`` resolved the name but the factory's registry
            # snapshot does not carry it. Construct the class we already hold,
            # from the SAME resolved kwargs, filtered to its contract.
            #
            # This is not a silent fallback to a default (pitfall #9): it is
            # the exact class the registry lookup returned, with the same
            # kwargs the factory branch would use. Production cannot reach it
            # -- ModelConfigSchema rejects an unregistered ``model_type`` when
            # the config is constructed, long before the probe runs -- so it
            # exists for classes registered after the snapshot was taken.
            contract = resolve_contract(model_cls=cls)
            if contract.accepts_var_kwargs:
                ctor_kwargs = dict(resolved.kwargs)
            else:
                ctor_kwargs = {k: v for k, v in resolved.kwargs.items() if k in contract.accepted}
            for _name, _value in (
                ("in_channels", in_channels),
                ("out_channels", out_channels),
            ):
                if contract.accepts_var_kwargs or _name in contract.accepted:
                    ctor_kwargs[_name] = _value
            model = cls(**ctor_kwargs)
        model = model.to(device)
        # The probe runs a real backward pass below, so an arm that enables
        # gradient checkpointing in training must checkpoint here too --
        # otherwise the probe validates a memory profile training never uses.
        apply_gradient_checkpointing(model, config)
        model.train()
    except Exception as exc:
        return ProbeResult(
            passed=False,
            category="instantiation",
            message=f"{model_type}(...) raised on construction: {exc}",
            severity="error",
            yaml_keys=["model.model_type", "model.model_kwargs"],
            fix_hint=(
                "Check model.model_kwargs against the model's __init__ signature. "
                "Likely an unsupported kwarg or wrong type."
            ),
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
            traceback=traceback.format_exc(),
        )

    # ── Runtime input-concatenation hint ─────────────────────────────
    # Some models build their first conv expecting the post-concat
    # shape that the *training strategy* produces, not the raw
    # ``in_channels`` declared in YAML. Example: ``kspace_cold_diffusion``
    # sets ``backbone_in_channels = in_channels * 2`` because the
    # diffusion strategy concatenates S-maps onto the input at runtime
    # (see ``diffusion.py::_prepare_diffusion_inputs``). The probe
    # doesn't run the strategy, so without mirroring the concat the
    # raw ``[B, in_channels, H, W]`` tensor crashes the model's first
    # conv ("expected 16 channels, got 8").
    #
    # Convention: a model class can declare any of these to tell the
    # probe how many channels its forward path actually expects:
    #
    #   * ``model_expects_smaps_concat(model)`` -> 2x in_channels
    #       (the one resolver — CLAUDE.md #17. NOT ``condition_with_smaps``,
    #       which stays True on the internal-DC arms whose backbone is 1x.)
    #   * ``self.synthetic_forward_probe_input_channels = N``  -> exact N
    #       (escape hatch for models with non-2× expansions)
    #
    # If neither is set, the raw ``in_channels`` is used. This keeps
    # the probe correct for models that mirror the strategy's concat
    # while staying lossless for the common case.
    runtime_in_channels = (
        int(getattr(model, "synthetic_forward_probe_input_channels", 0)) or in_channels
    )
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        model_expects_smaps_concat,
    )

    # Recorded, not re-derived. The measurement handed to the model's probe
    # kwargs hook below must be at the DECLARED width, so that inverse needs to
    # know whether *this* call inflated. Re-deriving it there from
    # ``x.shape[1] > in_channels`` would be a second resolver for one invariant
    # (CLAUDE.md #17) and would also mis-fire on the
    # ``synthetic_forward_probe_input_channels`` escape hatch, whose exact-N
    # width is the model's own declaration and must not be sliced.
    probe_input_was_inflated = False
    if runtime_in_channels == in_channels and model_expects_smaps_concat(model):
        runtime_in_channels = in_channels * 2
        probe_input_was_inflated = True

    # ── Build dummy tensor (Shepp-Logan phantom by default) ──────────
    shape = (batch_size, runtime_in_channels, *patch_size)
    target_shape = (batch_size, out_channels, *patch_size)
    try:
        if use_phantom:
            from spectramr.infrastructure.validation.phantom_builder import synthetic_phantom

            x = synthetic_phantom(shape, device=device)
            target = synthetic_phantom(target_shape, device=device)
        else:
            x = torch.randn(shape, device=device, requires_grad=False)
            target = torch.randn(target_shape, device=device, requires_grad=False)
    except RuntimeError as exc:  # CUDA OOM, etc.
        return _oom_or_runtime_result(exc, device, time.perf_counter() - t0, stage="dummy_tensor")
    except Exception as exc:
        # Phantom builder failure must not break the probe — fall back
        # to randn so the forward path still gets exercised.
        logger.warning("synthetic_phantom failed (%s); falling back to randn input.", exc)
        x = torch.randn(shape, device=device, requires_grad=False)
        target = torch.randn(target_shape, device=device, requires_grad=False)

    # ── Contract-aware complex dtype ──────────────────────────────────
    # A model that DECLARES ``accepts_complex=True`` (and is NOT the
    # interleaved-real-channel idiom) expects a genuine ``torch.complex``
    # tensor — feeding it the real-valued phantom would crash with a dtype
    # error that looks like a model bug. Coerce to complex (real part = the
    # recognisable phantom, imag = 0) so the complex path is exercised. The
    # common ``expects_real_imag_interleaved`` idiom needs no coercion (it
    # already takes a real, even-channel tensor — the phantom is real).
    if (
        declared_caps is not None
        and getattr(declared_caps, "accepts_complex", None) is True
        and getattr(declared_caps, "expects_real_imag_interleaved", None) is not True
    ):
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        if not torch.is_complex(target):
            target = torch.complex(target, torch.zeros_like(target))

    # ── Forward pass ──────────────────────────────────────────────────
    # Some models (diffusion / latent-diffusion) declare *required* forward
    # kwargs beyond the input tensor — most commonly ``timesteps``. The
    # plain ``model(x)`` call would TypeError on those even though the
    # model itself is healthy. Inspect the signature and pass synthetic
    # values for any required positional/keyword parameter we recognize.
    forward_extra: dict[str, object] = {}
    try:
        import inspect

        sig = inspect.signature(model.forward)
        for pname, p in sig.parameters.items():
            if pname == "self" or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            # ``x`` is the input — skip the first positional.
            if pname == "x" or pname == "input" or pname == "noisy_images":
                continue
            plow = pname.lower()
            # Image-conditioning (source / low-res / reference). The MICCAI
            # cross-field score nets (FieldGuidedScoreUNet) take a source image
            # they REQUIRE even though its default is None — passing None would
            # raise. Synthesise an x-like tensor. Checked BEFORE both the
            # default-skip and the generic 'cond' branch (``cond_image`` contains
            # 'cond', which would otherwise be set to None).
            if plow in (
                "cond_image",
                "source",
                "source_image",
                "src",
                "lr",
                "lr_image",
                "low_res",
            ):
                forward_extra[pname] = torch.randn_like(x)
                continue
            # Per-sample contrast index (T1w/T2w/FLAIR) for contrast-conditioned
            # mrixfields models (FieldVelocityUNet(use_contrast_conditioning=True)).
            # It carries a default (None) but the model RAISES on None when
            # conditioning is on (#15), so synthesise a valid in-range index
            # BEFORE the default-skip below. Index 0 is valid for any
            # num_contrasts >= 1; contrast-blind models absorb it via **kwargs.
            if plow in ("contrast_id", "contrast_idx", "contrast_index"):
                forward_extra[pname] = torch.zeros(batch_size, dtype=torch.long, device=device)
                continue
            # Diffusion timestep on a forward that makes it OPTIONAL. Third
            # instance of the pattern the two branches above already handle: the
            # parameter carries a default, and the model nonetheless needs a real
            # value. ``KSpaceColdDiffusionGenerator.forward(x, timesteps=None,
            # **kwargs)`` warns and DEGRADES TO t=0 when it is absent — the
            # fully-denoised boundary, where cold diffusion legitimately
            # approaches identity — so the probe was exercising the one timestep
            # at which the time embedding cannot be wrong and the
            # identity-collapse check below is meaningless. Its own warning text
            # concedes "The probe path is expected", i.e. the model had been
            # taught to expect the probe to get this wrong.
            #
            # Gated on the arm DECLARING a schedule, not applied unconditionally:
            # a model whose ``timesteps=None`` selects a genuinely different mode
            # must keep receiving None when no schedule is configured. Required
            # ``timesteps`` params are unaffected — they fall through to the
            # branch past the default-skip and fill as before.
            if ("timestep" in plow or plow in ("t", "ts")) and (
                p.default is not inspect.Parameter.empty
            ):
                if _declared_schedule_length(config) is not None:
                    forward_extra[pname] = torch.full(
                        (batch_size,),
                        _resolve_probe_timestep(config),
                        dtype=torch.long,
                        device=device,
                    )
                continue
            # Only fill REQUIRED args (no default).
            if p.default is not inspect.Parameter.empty:
                continue
            if "timestep" in plow or plow in ("t", "ts"):
                # A *mid-schedule* timestep probes the typical training path.
                # t=0 is the trivial fully-denoised boundary: for cold
                # diffusion the model legitimately approaches identity there,
                # so a t=0 probe both misses time-embedding bugs and can
                # spuriously trip the identity-collapse check below.
                t_mid = _resolve_probe_timestep(config)
                forward_extra[pname] = torch.full(
                    (batch_size,), t_mid, dtype=torch.long, device=device
                )
            elif "field_strength" in plow or plow in ("b0", "field"):
                # Continuous field coordinate (Tesla) for the MICCAI cross-field
                # arms (field_flow / cross_field / field_guided_diffusion). A
                # representative mid-range field; covers field_strength_target too
                # (substring match). Without this the probe would TypeError on the
                # required keyword-only field_strength of those forwards.
                forward_extra[pname] = torch.full(
                    (batch_size,), 1.5, dtype=torch.float32, device=device
                )
            elif plow in ("s", "style", "style_code"):
                # Strategy-computed style vector for AdaIN generators
                # (StarGANv2Generator.forward(x, s), disentangled/SimAE style
                # generators). The style is produced at runtime by a mapping
                # network / style encoder the probe doesn't run, so synthesise a
                # ``[B, style_dim]`` code — reading ``style_dim`` off the model
                # (default 64). Only fires for a REQUIRED style arg (this branch
                # is past the default-skip above), so generators that make style
                # optional are untouched. Without this the probe would TypeError
                # on the required positional ``s`` and every StarGAN v2 arm would
                # false-fail Tier-2 even though the model is healthy.
                style_dim = int(getattr(model, "style_dim", 64))
                forward_extra[pname] = torch.randn(batch_size, style_dim, device=device)
            elif "context" in plow or "cond" in plow:
                forward_extra[pname] = None
    except (TypeError, ValueError):  # pragma: no cover — defensive
        pass

    # ── Model-declared contract kwargs (the ``**kwargs`` half) ────────
    # Signature inspection above cannot reach a forward contract that arrives
    # through ``**kwargs``: ``VAR_KEYWORD`` is skipped by construction, and no
    # amount of introspection can enumerate names a signature does not list.
    # ``KSpaceColdDiffusionGenerator`` takes ``kspace_measured`` and ``mask``
    # that way, and RAISES when an arm declares data consistency or an output
    # magnitude bound without them (both mechanisms are gated on the
    # measurement, so their absence would return the unconstrained backbone
    # proposal while config, audit and provenance all report them active). The
    # probe therefore false-failed every such arm at Tier 2 — the model was
    # healthy and the audit's own call was incomplete.
    #
    # A model-side hook rather than a special case here, because the requirement
    # is per-INSTANCE, not per-class: it holds only when *this* arm constructed a
    # ``dc_layer`` or a clip ratio. A registry capability or a class attribute
    # cannot express that. Same ``synthetic_forward_probe_*`` family as the
    # ``synthetic_forward_probe_skip`` opt-out read further down.
    #
    # Signature-derived values WIN on conflict (``setdefault``): those are
    # config-aware (``_resolve_probe_timestep`` reads the declared schedule),
    # whereas the hook only knows the model. A hook that raises is reported, not
    # swallowed — a broken contract declaration is a real finding, and silently
    # dropping it would put us back to probing an unconstrained forward.
    _probe_kwargs_hook = getattr(model, "synthetic_forward_probe_kwargs", None)
    if callable(_probe_kwargs_hook):
        # The hook's contract is "a copy of what the TRAINING strategy passes"
        # -- and the strategy passes the UN-inflated batch
        # (``diffusion.py`` sets ``kspace_measured = input_batch``, which is at
        # the declared ``in_channels``). The smaps-concat inflation above exists
        # only to satisfy the backbone's FIRST CONV; handing the inflated ``x``
        # to the hook fabricates a measurement twice as wide as any real one.
        #
        # That fabrication is invisible on most arms because the generator's own
        # DC path narrows a too-wide measurement before use -- but a backbone
        # with an INTERNAL ``MaskedReplacementDataConsistency`` receives the kwarg verbatim
        # and has no such narrowing. ``swin_diff_rec`` then died on
        # ``k_guessed=[2,4,H,W]`` vs ``measured=[2,8,H,W]`` (complex channels;
        # 4 == out_channels/2, 8 == in_channels), failing a healthy arm on a
        # tensor training never produces (#1346).
        #
        # Slice the FIRST ``in_channels``: the concat this mirrors is
        # ``torch.cat([x, smaps_k], dim=1)`` -- data first, smaps last. Taking
        # the last N here would silently feed sensitivity maps as the
        # measurement, which is a wrong-answer bug rather than a crash.
        _hook_x = x[:, :in_channels] if probe_input_was_inflated else x
        for _k, _v in dict(_probe_kwargs_hook(_hook_x)).items():
            forward_extra.setdefault(_k, _v)

    try:
        y = model(x, **forward_extra) if forward_extra else model(x)
        # Some models return tuples / dicts; pick the principal tensor.
        if isinstance(y, tuple):
            y = y[0]
        elif isinstance(y, dict):
            y = next(iter(y.values()))
        if not isinstance(y, torch.Tensor):
            return ProbeResult(
                passed=False,
                category="forward_pass_shape",
                message=(
                    f"{model_type}(...).forward(x) returned a non-tensor "
                    f"({type(y).__name__}); cannot validate shape."
                ),
                severity="error",
                device=device,
                elapsed_seconds=time.perf_counter() - t0,
            )
    except RuntimeError as exc:
        return _oom_or_runtime_result(exc, device, time.perf_counter() - t0, stage="forward")
    except Exception as exc:
        # The hint must follow the exception, not assume a shape problem. A
        # genuine shape/dtype mismatch from a conv or a matmul arrives as
        # RuntimeError and is handled by the branch above, so this branch is
        # dominated by TypeError (the probe built the wrong call) and by models
        # RAISING from inside their own body (a contract guard firing) — for
        # which "check that model_kwargs are compatible with the configured
        # patch_size" is actively misleading. It sent the
        # experiment_11_attention_none audit to inspect patch_size and
        # model_kwargs over a ValueError whose own message named the missing
        # `kspace_measured` kwarg and the two mechanisms gated on it.
        #
        # `category` stays "forward_pass_shape" deliberately: it is the wire
        # format the audit aggregator and campaign leaderboard group on, and
        # renaming it is a consumer change, not a diagnosis fix.
        _kind = type(exc).__name__
        if isinstance(exc, TypeError):
            _keys = ["model.model_type", "model.model_kwargs"]
            _hint = (
                f"forward() rejected the arguments the probe supplied "
                f"({', '.join(sorted(forward_extra)) or 'x only'}). The probe "
                "builds its call from the forward SIGNATURE, so a contract that "
                "arrives via **kwargs is invisible to it. Declare it on the "
                "model as `synthetic_forward_probe_kwargs(self, x) -> dict` — "
                "the probe merges that in without overriding signature-derived "
                "values."
            )
        else:
            _keys = ["model.model_kwargs", "physics.data_consistency"]
            _hint = (
                f"The model raised {_kind} from inside its own forward — this is "
                "the model's contract talking, not a shape problem, so read its "
                "message above first (it names the specific knob or kwarg). A "
                "guard that fires because a DECLARED mechanism has nothing to "
                "act on has two honest resolutions: supply what it needs, or "
                "drop the knob so the arm stops advertising physics it does not "
                "apply. If the missing input is one the training strategy "
                "supplies at runtime, the probe needs it declared via "
                "`synthetic_forward_probe_kwargs(self, x) -> dict` on the model."
            )
        return ProbeResult(
            passed=False,
            category="forward_pass_shape",
            message=f"forward(x) raised {_kind}: {exc}",
            severity="error",
            yaml_keys=_keys,
            fix_hint=_hint,
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
            traceback=traceback.format_exc(),
        )

    if y.shape != target.shape:
        return ProbeResult(
            passed=False,
            category="forward_pass_shape",
            message=(
                f"forward output shape {tuple(y.shape)} != target shape "
                f"{tuple(target.shape)}. The loss path will fail at runtime."
            ),
            severity="error",
            yaml_keys=["model.out_channels", "data.patch_size"],
            fix_hint=(
                f"The model emits {y.shape[1]} channels at "
                f"{tuple(y.shape[2:])} resolution. Set model.out_channels "
                f"to {y.shape[1]} OR adjust model_kwargs so the spatial "
                f"resolution matches the target."
            ),
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ── Identity-collapse (residual / consistency models) ─────────────
    # If the model receives a random tensor and produces (almost) the
    # same tensor, it has collapsed to identity — the prototype bug
    # behind the 12 ``identity collapse`` warnings on
    # ``exp_05_consistency_model``. Only fires when shapes match.
    #
    # Opt-out: a model class can declare
    #   synthetic_forward_probe_skip = {"identity_collapse"}
    # to suppress this check (residual / consistency / normalising-flow
    # models legitimately produce output ≈ input at initialisation).
    _probe_skip = _coerce_probe_skip(getattr(model, "synthetic_forward_probe_skip", None))
    if y.shape == x.shape and "identity_collapse" not in _probe_skip:
        try:
            x_norm = x.detach().abs().mean().clamp_min(1.0e-12)
            rel_diff = (y.detach() - x.detach()).abs().mean() / x_norm
            if float(rel_diff) < _IDENTITY_COLLAPSE_EPSILON:
                return ProbeResult(
                    passed=False,
                    category="identity_collapse",
                    message=(
                        f"{model_type}: forward output is within "
                        f"{_IDENTITY_COLLAPSE_EPSILON:g} relative L1 of the input "
                        f"on a random tensor (rel_diff={float(rel_diff):.2e}). "
                        "Likely an identity-collapse pathology — the model is "
                        "ignoring its weights."
                    ),
                    severity="warning",
                    yaml_keys=["model.model_type", "training.strategy_class"],
                    fix_hint=(
                        "If the model is intentionally a residual / consistency "
                        "block (output ≈ input by design), suppress this check "
                        "by adding `synthetic_forward_probe_skip = "
                        "{'identity_collapse'}` to the model class."
                    ),
                    device=device,
                    elapsed_seconds=time.perf_counter() - t0,
                )
        except Exception:  # pragma: no cover — never fail the probe on a stat
            pass

    # ── Input-invariance (measurement-independent / facade output) ────
    # If the output barely changes when the INPUT changes, the model is
    # ignoring its measurement — the DC-blob / NeSVoR-grid / zerorf-.expand
    # facade class (#16/#20) that the identity-collapse check above MISSES
    # (a constant output is not ≈ its input, so identity-collapse never fires).
    # Re-runs the SAME input as a stochastic noise floor, then a DIFFERENT one.
    #
    # Opt-out: synthetic_forward_probe_skip = {"input_invariant"} for models
    # whose output is legitimately measurement-independent (e.g. an
    # unconditional prior sampled from noise alone).
    if "input_invariant" not in _probe_skip:
        try:

            def _principal(out: object) -> object:
                if isinstance(out, tuple):
                    return out[0]
                if isinstance(out, dict):
                    return next(iter(out.values()))
                return out

            with torch.no_grad():
                y_same = _principal(model(x, **forward_extra) if forward_extra else model(x))
                x2 = torch.randn_like(x)
                y_diff = _principal(model(x2, **forward_extra) if forward_extra else model(x2))
            if (
                isinstance(y_same, torch.Tensor)
                and isinstance(y_diff, torch.Tensor)
                and y_same.shape == y.shape
                and y_diff.shape == y.shape
            ):
                d_same, d_diff = _input_invariance_stats(y, y_same, y_diff)
                if _is_input_invariant(d_same, d_diff):
                    return ProbeResult(
                        passed=False,
                        category="input_invariant",
                        message=(
                            f"{model_type}: forward output is (near-)invariant to the "
                            f"input (rel change on a different input d_diff={d_diff:.2e} "
                            f"vs re-run noise floor d_same={d_same:.2e}). The model "
                            "ignores its measurement — a measurement-independent / "
                            "facade output (the DC-blob class) that smoke never catches."
                        ),
                        severity="warning",
                        yaml_keys=["model.model_type", "model.model_kwargs"],
                        fix_hint=(
                            "Wire the input into the forward path. If the model is "
                            "intentionally measurement-independent (e.g. an "
                            "unconditional prior), declare "
                            "`synthetic_forward_probe_skip = {'input_invariant'}` on "
                            "the model class."
                        ),
                        device=device,
                        elapsed_seconds=time.perf_counter() - t0,
                    )
        except Exception:  # pragma: no cover — never fail the probe on the extra forwards
            pass

    # ── NaN/Inf in forward output ────────────────────────────────────
    if not torch.isfinite(y).all():
        return ProbeResult(
            passed=False,
            category="nan_in_forward",
            message=(
                f"{model_type}: forward output contains NaN/Inf on random input. "
                "This will silently poison training metrics."
            ),
            severity="error",
            yaml_keys=["model.model_kwargs", "data.normalization_type"],
            fix_hint=(
                "Likely sources: an unbounded activation (no clamp / norm), "
                "a divide-by-zero in a custom block, or an unstable "
                "initialisation. Inspect the model's forward path."
            ),
            device=device,
            elapsed_seconds=time.perf_counter() - t0,
        )

    # ── Configured-loss gradient-fires (dead_loss audit gate) ────────
    # The dummy L1 backward below exercises AMP/grad paths but NOT the
    # *configured* objective. Run the real reconstruction loss at iteration 0
    # (the most adversarial point — warmup gates are fully active) and fail if
    # g_total_loss carries no gradient. Catches the warmup-eats-the-only-loss
    # dead_loss class (cs_mno cohort) BEFORE a cluster run instead of after.
    # Skips silently for any non-reconstruction / non-image-domain config.
    #
    # Opt-out: synthetic_forward_probe_skip = {"dead_loss"} for SELF-COMPUTING
    # strategies (Fisher-Rao geodesic, McCann ICNN, …) whose YAML image_losses is a
    # placeholder — the real objective is computed by the strategy, not this
    # LossComputer. Those models are also identity-at-init (geodesic endpoint / ICNN),
    # so the configured l1 == 0 (output == phantom target) and this gate would
    # FALSE-POSITIVE even though the arm trains on the cluster. Mirrors the
    # identity_collapse / input_invariant opt-outs above.
    if "dead_loss" not in _probe_skip:
        _cfg_total = _configured_loss_total(config, y, target)
        if _cfg_total is not _LOSS_PROBE_SKIP:
            verdict = _loss_gradient_verdict(
                _cfg_total, str(model_type), device, time.perf_counter() - t0
            )
            if verdict is not None:
                return verdict

    # ── Backward pass (catches AMP double-unscale, NaN gradients) ────
    if backward:
        try:
            loss = (y - target).abs().mean()
            if not torch.isfinite(loss).all():
                return ProbeResult(
                    passed=False,
                    category="nan_in_loss",
                    message=(
                        f"{model_type}: dummy L1 loss is NaN/Inf "
                        f"({float(loss)!r}). Training cannot progress."
                    ),
                    severity="error",
                    yaml_keys=["model.model_kwargs"],
                    fix_hint=(
                        "Forward output contains NaN/Inf — check normalisation / "
                        "activations / weight init."
                    ),
                    device=device,
                    elapsed_seconds=time.perf_counter() - t0,
                )
            loss.backward()
        except RuntimeError as exc:
            cat = "amp_double_unscale" if "unscale" in str(exc).lower() else "forward_pass_shape"
            return ProbeResult(
                passed=False,
                category=cat,
                message=f"backward() raised {type(exc).__name__}: {exc}",
                severity="error",
                yaml_keys=[
                    "optimization.precision.enabled",
                    "optimization.gradient.clip.enabled",
                ],
                fix_hint=(
                    "AMP + gradient clipping double-unscale: see "
                    "check_amp_grad_clip_interaction. Set "
                    "gradient_clip_value > 0 OR disable use_amp."
                    if cat == "amp_double_unscale"
                    else "Backward pass shape mismatch — check loss reduction "
                    "and any per-loss .reshape() call."
                ),
                device=device,
                elapsed_seconds=time.perf_counter() - t0,
                traceback=traceback.format_exc(),
            )

        # ── Gradient explosion / NaN-in-grad ─────────────────────────
        try:
            total_grad_sq = 0.0
            saw_nan_grad = False
            for p in model.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    saw_nan_grad = True
                    break
                total_grad_sq += float(p.grad.detach().pow(2).sum())
            total_grad_norm = total_grad_sq**0.5
            if saw_nan_grad:
                return ProbeResult(
                    passed=False,
                    category="nan_in_gradient",
                    message=(
                        f"{model_type}: at least one parameter has NaN/Inf "
                        "gradient after one backward pass. Training will "
                        "diverge on the first step."
                    ),
                    severity="error",
                    yaml_keys=["model.model_kwargs", "optimization.optimizer.learning_rate"],
                    fix_hint=(
                        "Reduce optimization.optimizer.learning_rate, add a normaliser to "
                        "the unstable layer, or clamp the activation that's "
                        "blowing up."
                    ),
                    device=device,
                    elapsed_seconds=time.perf_counter() - t0,
                )
            if total_grad_norm > _GRAD_EXPLOSION_THRESHOLD:
                return ProbeResult(
                    passed=False,
                    category="gradient_explosion",
                    message=(
                        f"{model_type}: total gradient norm "
                        f"{total_grad_norm:.2e} exceeds "
                        f"{_GRAD_EXPLOSION_THRESHOLD:.0e} on the first synthetic "
                        "backward pass. Real training will explode on iter 1."
                    ),
                    severity="warning",
                    yaml_keys=[
                        "optimization.optimizer.learning_rate",
                        "optimization.gradient.clip.enabled",
                        "optimization.gradient.clip.value",
                    ],
                    fix_hint=(
                        "Enable optimization.gradient.clip.enabled with a "
                        "positive optimization.gradient.clip.value (e.g. 1.0), "
                        "OR reduce the initial learning rate."
                    ),
                    device=device,
                    elapsed_seconds=time.perf_counter() - t0,
                )
        except Exception:  # pragma: no cover — never break the probe on a stat
            pass

    # ── Optional: dump probe images for visual inspection ────────────
    if save_images_dir is not None:
        try:
            from spectramr.infrastructure.validation.phantom_builder import save_probe_images

            save_probe_images(
                save_images_dir,
                arm_name,
                input_tensor=x,
                output_tensor=y,
                target_tensor=target,
            )
        except Exception as exc:  # pragma: no cover — never fail probe on save
            logger.debug("save_probe_images skipped: %s", exc)

    # ── Tier-2 model-specific invariant probe ────────────────────────
    # For the Phase-3 models, assert the defining mathematical invariant
    # (Glow round-trip, divergence-free ∇·v, vMF sphere constraint, ...).
    # A structurally-broken model fails here instead of producing
    # silently-wrong science. See validation/phase3_probes.py.
    try:
        from spectramr.infrastructure.validation.phase3_probes import run_invariant_probe

        inv = run_invariant_probe(model_type, model, x, device)
        if inv is not None and not inv[0]:
            return ProbeResult(
                passed=False,
                category="invariant",
                message=f"{model_type}: invariant probe FAILED — {inv[1]}",
                severity="error",
                yaml_keys=["model.model_type", "model.model_kwargs"],
                fix_hint=(
                    "The model violates its defining invariant on a synthetic "
                    "input. Check model_kwargs and the model implementation."
                ),
                device=device,
                elapsed_seconds=time.perf_counter() - t0,
            )
        invariant_note = f" | invariant: {inv[1]}" if inv is not None else ""
    except Exception as exc:  # pragma: no cover — never break the probe
        invariant_note = ""
        logger.debug("invariant probe skipped: %s", exc)

    # ── Generative sample() path (latent / diffusion) ────────────────
    # forward() above is the TRAINING step; many diffusion / latent models
    # generate through a separate, cost-bounded sample() whose shape contract
    # forward never touches. Best-effort + bounded (see _probe_sample_path):
    # fails ONLY on a concrete generation-path rank mismatch, never on the
    # synthetic args. Opt out per-model via probe_skip={"sample_path"}.
    sample_note = ""
    if "sample_path" not in _probe_skip:
        try:
            sample_ok, sample_note = _probe_sample_path(
                model,
                x,
                (y.ndim - 2 if y.ndim >= 2 else y.ndim),
                batch_size,
                device,
            )
            if not sample_ok:
                return ProbeResult(
                    passed=False,
                    category="sample_path_shape",
                    message=f"{model_type}: {sample_note.lstrip(' |')}",
                    severity="error",
                    yaml_keys=["model.model_type", "model.model_kwargs"],
                    fix_hint=(
                        "The model's sample() generation path emits a different "
                        "spatial rank than its forward() training path. Reconcile "
                        "the decoder / projection head so both produce the same "
                        "dimensionality, OR (if sample() is intentionally a "
                        "different shape) declare "
                        "synthetic_forward_probe_skip = {'sample_path'} on the model."
                    ),
                    device=device,
                    elapsed_seconds=time.perf_counter() - t0,
                )
        except Exception as exc:  # pragma: no cover — never break the probe
            sample_note = ""
            logger.debug("sample() probe skipped: %s", exc)

    return ProbeResult(
        passed=True,
        category="forward_pass_shape",
        message=(
            f"{model_type}: forward {tuple(x.shape)} -> {tuple(y.shape)} "
            f"+ backward OK on {device}.{invariant_note}{sample_note}"
        ),
        severity="info",
        device=device,
        elapsed_seconds=time.perf_counter() - t0,
    )


def _oom_or_runtime_result(exc: Exception, device: str, elapsed: float, stage: str) -> ProbeResult:
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda oom" in msg:
        return ProbeResult(
            passed=False,
            category="oom",
            message=f"CUDA OOM during {stage}: {exc}",
            severity="error",
            yaml_keys=["data.batch_size", "data.patch_size", "model.model_kwargs"],
            fix_hint=(
                "Reduce data.batch_size or data.patch_size, OR shrink "
                "model.model_kwargs (e.g. fewer channels / blocks). "
                "If running on CPU, this is unexpected — re-check the "
                "model's allocator."
            ),
            device=device,
            elapsed_seconds=elapsed,
            traceback=traceback.format_exc(),
        )
    return ProbeResult(
        passed=False,
        category="forward_pass_shape" if stage == "forward" else "instantiation",
        message=f"RuntimeError during {stage}: {exc}",
        severity="error",
        device=device,
        elapsed_seconds=elapsed,
        traceback=traceback.format_exc(),
    )
