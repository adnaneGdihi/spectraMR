"""CLI subcommand handlers for the seven novel research arms.

Provides three subcommands documented in
``TODO/audit/audit_plan_novel.md``:

- ``audit-ksd`` (Idea 7) — run the Tier-3 KSD defensibility audit on
  a config + samples and emit a structured report.
- ``infer-protocol`` (Idea 2) — Langevin chain in the contrast-direction
  ``c`` with the image ``x`` frozen, to recover the most likely
  acquisition parameters that produced a given image.
- ``simulate-acquisition`` (Idea 4) — end-to-end synthetic acquisition
  trajectory with per-step reconstruction metrics for comparison
  against ``bald`` / uniform-random baselines.

Each handler is registered into the existing ``argparse`` subparser
tree in :mod:`spectramr.cli.app`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# NOTE: ``torch`` and the physics FFT ops are imported lazily inside the
# command handlers below — never at module scope. ``attach_subparsers``
# (called from ``cli.app`` so ``--help`` lists these subcommands) must
# touch only argparse + stdlib so the help path stays torch-free.
# ``from __future__ import annotations`` keeps the ``torch.Tensor`` type
# hints as strings, so they do not force an eager import either; the
# ``TYPE_CHECKING`` block below makes ``torch`` resolvable for static
# type-checkers and linters without importing it at runtime.
if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# audit-ksd — Idea 7
# --------------------------------------------------------------------- #


def audit_ksd_cmd(args: argparse.Namespace) -> int:
    """Run the Tier-3 KSD defensibility audit standalone.

    The subcommand has two modes:

    - ``--samples <path>`` — load a precomputed ``[n, d]`` tensor of
      samples (``.pt`` / ``.npy``) and test against a registered score
      callable resolved from the config.
    - ``--smoke`` — run the self-test path (unit Gaussian) so that the
      integration can be exercised in CI without a trained model.

    Exits 0 on pass, 1 on warnings, 2 on hard failure (small p-value
    below the configured threshold).
    """
    from spectramr.infrastructure.validation.ksd_defensibility import (
        smoke_mode_self_test,
        verify_ksd_defensibility,
    )

    config_path: Path = args.config
    threshold: float = args.threshold
    n_bootstrap: int = args.bootstrap
    n_samples: int = args.samples_count

    if args.smoke:
        result = smoke_mode_self_test(n_samples=n_samples)
        message_lines = ["[audit-ksd smoke mode]", result.message]
    else:
        samples_path: Path = args.samples
        if not samples_path.is_file():
            print(f"audit-ksd: samples file not found: {samples_path}", file=sys.stderr)
            return 2
        samples = _load_samples(samples_path)

        def _samples_provider(n: int) -> torch.Tensor:
            if samples.shape[0] < n:
                logger.warning(
                    "Requested %d samples but only %d available; using all.",
                    n,
                    samples.shape[0],
                )
                return samples
            return samples[:n]

        # Default score: unit-Gaussian (callers wanting a real model
        # should pass --score-module). The default is intentionally
        # the same as the smoke mode so that running audit-ksd
        # without a registered score is *meaningful* (it tests that
        # the samples are standardised) rather than silently incorrect.
        score_fn = _resolve_score_callable(args.score_module)

        result = verify_ksd_defensibility(
            samples_provider=_samples_provider,
            score_fn=score_fn,
            n_samples=n_samples,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
        )
        message_lines = [
            f"[audit-ksd] config: {config_path}",
            f"           samples: {samples_path} (n={result.n_samples})",
            result.message,
        ]

    output: dict[str, Any] = {
        "passed": result.passed,
        "ksd_squared": result.ksd_squared,
        "p_value": result.p_value,
        "threshold": result.threshold,
        "n_samples": result.n_samples,
        "bandwidth": result.bandwidth,
        "message": result.message,
    }
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        for line in message_lines:
            print(line)

    # Optional PDF certificate (regulatory submission artefact).
    pdf_path: Path | None = getattr(args, "certificate_pdf", None)
    if pdf_path is not None:
        try:
            from spectramr.infrastructure.reporting.plotters.ksd_certificate import (
                render_ksd_certificate,
            )

            render_ksd_certificate(
                pdf_path,
                result=result,
                config_path=config_path,
                samples_path=getattr(args, "samples", None),
            )
            print(f"[audit-ksd] wrote certificate to {pdf_path}")
        except Exception as exc:
            logger.warning("audit-ksd: certificate render failed: %s", exc)

    if not result.passed:
        return 2
    if result.p_value < 2 * result.threshold:
        # close-call warning — pass but flag for the operator.
        return 1
    return 0


def _load_samples(path: Path) -> torch.Tensor:
    import torch

    suffix = path.suffix.lower()
    if suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "samples" in obj:
            return obj["samples"]
        if isinstance(obj, torch.Tensor):
            return obj
        raise ValueError(
            f"audit-ksd: .pt file at {path} must contain a Tensor or a dict with key 'samples'."
        )
    if suffix in {".npy", ".npz"}:
        import numpy as np

        arr = np.load(path)
        return torch.as_tensor(arr)
    raise ValueError(f"audit-ksd: unsupported samples format {suffix!r}; use .pt / .npy")


def _resolve_score_callable(module_path: str | None):
    """Return a callable score(samples) → scores.

    When ``module_path`` is None or unset, returns the unit-Gaussian
    score ``-x`` (also used by the smoke self-test). Otherwise imports
    the attribute named by ``module_path`` (dotted, e.g.
    ``my_pkg.scores.diffusion_score``) and uses it.
    """
    if not module_path:
        return lambda x: -x
    import importlib

    mod, attr = module_path.rsplit(".", 1)
    fn = getattr(importlib.import_module(mod), attr)
    if not callable(fn):
        raise TypeError(
            f"audit-ksd: resolved {module_path!r} is not callable ({type(fn).__name__})"
        )
    return fn


# --------------------------------------------------------------------- #
# infer-protocol — Idea 2
# --------------------------------------------------------------------- #


def infer_protocol_cmd(args: argparse.Namespace) -> int:
    """Recover the most likely acquisition parameters from a given image.

    Operates by Langevin sampling in the contrast-direction ``c`` with
    the image ``x`` frozen, using the ``s_c`` head of a trained
    score-field-tomography model.

    The handler is a thin wrapper that:
      1. Loads the model checkpoint.
      2. Loads the input image.
      3. Runs ``n_steps`` Langevin updates in ``c``.
      4. Reports the posterior-mode estimate ``ĉ`` and a per-coordinate
         credible interval.
    """
    checkpoint: Path = args.checkpoint
    image_path: Path = args.image
    n_steps: int = args.n_steps
    step_size: float = args.step_size
    # ``None`` when the user did not pass ``--cond-dim``: the checkpoint's own
    # shapes answer that question, and a default here could not be told apart
    # from a declaration (the D01#4 pattern).
    cond_dim: int | None = args.cond_dim

    import torch

    from spectramr.infrastructure.checkpointing.restore import read_part

    if not checkpoint.is_file():
        print(f"infer-protocol: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    if not image_path.is_file():
        print(f"infer-protocol: image not found: {image_path}", file=sys.stderr)
        return 2

    # The generator part of a RunState; read without pickle by the checkpoint codecs.
    state = read_part(checkpoint, "generator")
    model = _instantiate_model_from_state(state, cond_dim=cond_dim)
    image = _load_image_tensor(image_path)

    # One owner of "how wide is c": the constructed model. Reading it back off
    # the module keeps the Langevin chain and the network in agreement even
    # when the CLI said nothing (non-negotiable 17).
    cond_dim = model.head_sc.out_features
    c = torch.zeros(1, cond_dim, requires_grad=True)
    sigma = torch.tensor([args.sigma])
    trajectory: list[torch.Tensor] = []
    for _ in range(n_steps):
        _, s_c = model(image, sigma, c)
        noise = torch.randn_like(c)
        c_new = c.detach() + step_size * s_c.detach() + (2 * step_size) ** 0.5 * noise
        c = c_new.requires_grad_(True)
        trajectory.append(c.detach().clone())

    samples = torch.stack(trajectory[-min(50, len(trajectory)) :], dim=0)
    mean_c = samples.mean(dim=0).squeeze(0)
    std_c = samples.std(dim=0).squeeze(0)
    print("[infer-protocol] posterior-mode estimate:")
    for i, (m, s) in enumerate(zip(mean_c.tolist(), std_c.tolist())):
        print(
            f"  c[{i}] = {m: .4f}  ±  {s: .4f}  (95% CI: [{m - 1.96 * s: .4f}, {m + 1.96 * s: .4f}])"
        )
    return 0


def _architecture_from_state_dict(sd: dict) -> dict[str, int]:
    """Read ``ScoreFieldUNet``'s constructor arguments off the tensor shapes.

    A RunState carries **no config block**, so there is nothing to resolve
    through ``MODEL_REGISTRY``: the only architecture the checkpoint actually
    carries is the shape of its own tensors. Those pin all three arguments
    exactly:

    ``lift.weight``       -> ``[base_width, in_channels, 3, 3]``
    ``head_sc.weight``    -> ``[cond_dim, base_width]``
    ``cond_embed.0.weight`` -> ``[4 * base_width, cond_dim + 1]``  (cross-check)

    A missing or inconsistent key raises. Absent is a state to report, never
    a state to infer -- guessing here is what let ``infer-protocol`` print a
    credible interval sampled from an untrained network (#1351).
    """

    def _shape(key: str) -> tuple[int, ...]:
        if key not in sd:
            raise KeyError(
                f"infer-protocol: checkpoint state dict has no {key!r}; it does not "
                f"hold a score_field_unet (keys: {sorted(sd)[:8]}...)"
            )
        return tuple(sd[key].shape)

    lift = _shape("lift.weight")
    head_sc = _shape("head_sc.weight")
    cond_embed = _shape("cond_embed.0.weight")
    if len(lift) != 4 or len(head_sc) != 2 or len(cond_embed) != 2:
        raise ValueError(
            f"infer-protocol: unexpected tensor ranks in checkpoint "
            f"(lift={lift}, head_sc={head_sc}, cond_embed={cond_embed})"
        )

    base_width, in_channels = lift[0], lift[1]
    cond_dim = head_sc[0]
    expected = (4 * base_width, cond_dim + 1)
    if cond_embed != expected:
        raise ValueError(
            f"infer-protocol: checkpoint is internally inconsistent -- lift/head_sc "
            f"imply cond_embed.0.weight {expected}, found {cond_embed}"
        )
    if head_sc[1] != base_width:
        raise ValueError(
            f"infer-protocol: checkpoint is internally inconsistent -- lift implies "
            f"base_width={base_width}, head_sc.weight is {head_sc}"
        )
    return {
        "in_channels": int(in_channels),
        "cond_dim": int(cond_dim),
        "base_width": int(base_width),
    }


def _instantiate_model_from_state(sd: dict, *, cond_dim: int | None):
    """Instantiate ``score_field_unet`` from a generator state dict.

    Every mismatch raises: loading is strict and the architecture is read off
    the tensor shapes, so ``infer-protocol`` cannot print a credible interval
    computed from a randomly-initialised network (#1351, D01#1).

    ``cond_dim`` is the value the user *declared* on the command line, or
    ``None`` when they did not. It is a cross-check, never the source of
    truth: the checkpoint's own shapes are.
    """
    from spectramr.models.diffusion.score_field_unet import ScoreFieldUNet

    arch = _architecture_from_state_dict(sd)
    model = ScoreFieldUNet(**arch)
    # strict=True: a key mismatch is a wrong checkpoint, not a warning.
    model.load_state_dict(sd, strict=True)

    resolved = model.head_sc.out_features
    if cond_dim is not None and cond_dim != resolved:
        raise ValueError(
            f"infer-protocol: --cond-dim {cond_dim} contradicts the checkpoint, "
            f"which was trained with cond_dim={resolved}"
        )
    model.eval()
    return model


def _load_image_tensor(path: Path) -> torch.Tensor:
    import torch

    suffix = path.suffix.lower()
    if suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        img = obj["image"] if isinstance(obj, dict) and "image" in obj else obj
    elif suffix in {".npy", ".npz"}:
        import numpy as np

        img = torch.as_tensor(np.load(path)).float()
    else:
        raise ValueError(f"infer-protocol: unsupported image format {suffix!r}; use .pt / .npy")
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(0)
    return img.float()


# --------------------------------------------------------------------- #
# simulate-acquisition — Idea 4
# --------------------------------------------------------------------- #


def simulate_acquisition_cmd(args: argparse.Namespace) -> int:
    """Run a synthetic IB-acquisition trajectory and report per-step metrics.

    Reads a YAML config, builds a phantom-style ground-truth image,
    starts from a small low-frequency mask, and repeatedly calls
    :meth:`IBActiveAcquisitionStrategy.select_next_line` to extend the
    mask. After every acquisition the routine produces a baseline
    zero-filled reconstruction and logs PSNR / SSIM versus the
    ground truth. The output CSV is consumed by the reporting figure
    ``fig_2_15_active_acquisition_trajectory``.
    """
    import torch

    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
    from spectramr.infrastructure.training.strategies.ib_active_acquisition_strategy import (
        IBActiveAcquisitionStrategy,
    )

    config_path: Path = args.config
    output_csv: Path = args.output
    n_steps: int = args.n_steps
    H = int(args.height)
    W = int(args.width)

    if not config_path.is_file():
        print(f"simulate-acquisition: config not found: {config_path}", file=sys.stderr)
        return 2

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    image = _shepp_logan_phantom(H, W)
    target_kspace = fft2c(image)
    mask = torch.zeros(H, dtype=torch.bool)
    # Seed with the low-frequency centre band.
    centre_lo = H // 2 - max(int(0.04 * H), 1)
    centre_hi = H // 2 + max(int(0.04 * H), 1)
    mask[centre_lo:centre_hi] = True

    psnr_log: list[tuple[int, float]] = []
    for step in range(n_steps):
        # Zero-filled reconstruction at the current mask.
        recon_k = target_kspace * mask.view(1, 1, -1, 1)
        recon = ifft2c(recon_k).real
        psnr = _psnr(recon, image)
        psnr_log.append((step, psnr))
        next_line = IBActiveAcquisitionStrategy.select_next_line(
            mask, recon, mode="linear_gaussian"
        )
        mask[next_line] = True

    with output_csv.open("w") as f:
        f.write("step,psnr_db,acquired_lines\n")
        for step, psnr in psnr_log:
            f.write(f"{step},{psnr:.6f},{int(mask.sum().item())}\n")
    print(f"[simulate-acquisition] wrote {len(psnr_log)} rows to {output_csv}")
    if not args.quiet:
        for step, psnr in psnr_log[:: max(1, n_steps // 8)]:
            print(f"  step {step:4d}: PSNR = {psnr:6.2f} dB")
    return 0


def _shepp_logan_phantom(H: int, W: int) -> torch.Tensor:
    """Coarse Shepp-Logan-style phantom on a [H, W] grid."""
    import torch

    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij",
    )
    img = torch.zeros(H, W)
    img[(x**2 + y**2) <= 0.7] = 0.7
    img[((x - 0.2) ** 2 + (y - 0.1) ** 2) <= 0.05] = 1.0
    img[((x + 0.3) ** 2 + (y + 0.2) ** 2) <= 0.10] = 0.3
    return img.unsqueeze(0).unsqueeze(0)


def _psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    import torch

    mse = ((pred - target) ** 2).mean()
    return float((10.0 * torch.log10(1.0 / (mse + eps))).item())


# --------------------------------------------------------------------- #
# Argparse plumbing
# --------------------------------------------------------------------- #


def attach_subparsers(subparsers: Any) -> None:
    """Wire the three subcommands into the existing ``argparse`` tree.

    Call from :func:`spectramr.cli.app.main` right before ``parse_args`` so
    the new commands are visible to ``--help``.
    """
    ksd_p = subparsers.add_parser(
        "audit-ksd",
        help=(
            "Run the Tier-3 KSD defensibility audit on samples drawn from "
            "a registered generative-prior model (Idea 7 of audit_plan_novel.md)."
        ),
    )
    ksd_p.add_argument("config", type=Path, help="Path to experiment YAML (for context).")
    ksd_p.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Path to a .pt / .npy tensor of model samples (shape [n, d]).",
    )
    ksd_p.add_argument(
        "--smoke",
        action="store_true",
        help="Run the unit-Gaussian self-test instead of loading samples.",
    )
    ksd_p.add_argument(
        "--score-module",
        type=str,
        default=None,
        help=(
            "Dotted import path of a callable score_fn(samples) → scores. "
            "Defaults to the unit-Gaussian score (mostly useful for testing "
            "the integration path)."
        ),
    )
    ksd_p.add_argument("--threshold", type=float, default=0.05)
    ksd_p.add_argument("--bootstrap", type=int, default=500)
    ksd_p.add_argument("--samples-count", type=int, default=256)
    ksd_p.add_argument("--json", action="store_true")
    ksd_p.add_argument(
        "--certificate-pdf",
        type=Path,
        default=None,
        metavar="PATH",
        help="Render a one-page PDF defensibility certificate to PATH.",
    )
    ksd_p.set_defaults(func=audit_ksd_cmd)

    ip_p = subparsers.add_parser(
        "infer-protocol",
        help=(
            "Posterior-mode inference over acquisition parameters c using the "
            "s_c head of a trained score-field-tomography model (Idea 2)."
        ),
    )
    ip_p.add_argument("--checkpoint", type=Path, required=True)
    ip_p.add_argument("--image", type=Path, required=True)
    ip_p.add_argument("--n-steps", type=int, default=200)
    ip_p.add_argument("--step-size", type=float, default=1e-3)
    ip_p.add_argument("--sigma", type=float, default=0.1)
    # No default: an unset ``--cond-dim`` means "read it from the checkpoint".
    # A concrete default here is indistinguishable from a declared 5, which is
    # exactly how a mismatched checkpoint used to pass unnoticed.
    ip_p.add_argument(
        "--cond-dim",
        type=int,
        default=None,
        help=(
            "Acquisition-parameter dimension. Omit to read it from the "
            "checkpoint; pass it to assert the checkpoint agrees."
        ),
    )
    ip_p.set_defaults(func=infer_protocol_cmd)

    sa_p = subparsers.add_parser(
        "simulate-acquisition",
        help=(
            "Synthetic IB-acquisition trajectory with per-step metrics "
            "(Idea 4). Output is a CSV consumed by "
            "fig_2_15_active_acquisition_trajectory."
        ),
    )
    sa_p.add_argument("config", type=Path)
    sa_p.add_argument("--output", "-o", type=Path, required=True)
    sa_p.add_argument("--n-steps", type=int, default=32)
    sa_p.add_argument("--height", type=int, default=64)
    sa_p.add_argument("--width", type=int, default=64)
    sa_p.add_argument("--quiet", action="store_true")
    sa_p.set_defaults(func=simulate_acquisition_cmd)


__all__ = [
    "attach_subparsers",
    "audit_ksd_cmd",
    "infer_protocol_cmd",
    "simulate_acquisition_cmd",
]
