"""Domain-bridge contract for ``LossBuilder._build_list_based_losses``.

Targets ``spectramr.infrastructure.training.builders.loss_builder``.

The regression under test is the **double Fourier bridge** (issue #467). With
``losses.output_domain: kspace`` the builder wraps every ``image_losses`` entry in
``DifferentiableFourierBridge`` (iFFT -> magnitude) and every ``complex_losses``
entry in the same bridge with ``return_complex=True``. Several losses
(``complex_spatial_gradient``, ``sense_adjoint_l1``, ``rician_consistency``,
``background_suppression``) already carry their OWN bridge, so declaring them
under a bridged list inverse-transformed the tensor twice:

  [B, 8, H, W] k-space -> ifft2c -> abs() -> [B, 4, H, W] real magnitude
                       -> re-paired as [B, 2, H, W] complex  (coil-0 magnitude
                          becomes the real part, coil-1 the imaginary part)
                       -> ifft2c AGAIN

Nothing raised, no shape error, no NaN, and the value stayed finite — so the whole
``kspace_filling`` cohort trained a ``complex_spatial_gradient`` term at weight 1.0
that never saw phase, and a ``sense_adjoint_l1`` fidelity term whose 4-coil
sensitivity maps were silently truncated to 2 to match the halved channel count.

These tests pin the rejection and the two accepted placements.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.loss import LossComponentConfig, LossConfigSchema
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

# Losses that bridge from k-space internally. Kept in sync with the guard, which
# detects them structurally via the ``use_fourier_bridge`` attribute rather than
# by name — so a new self-bridging loss is covered without touching this list.
SELF_BRIDGING = ("complex_spatial_gradient", "sense_adjoint_l1")


class _ConfigShim:
    """Minimal config surface ``_build_list_based_losses`` reads.

    Mirrors the shim ``infrastructure.domain_validation`` already uses, so the
    test exercises the real builder rather than a mock of it.
    """

    def __init__(self, losses: LossConfigSchema) -> None:
        self.losses = losses
        self.objectives = None
        self.training = None
        self.training_mode = "reconstruction"
        self.deep_supervision_weight = 0.0


def _build(**lists: list[LossComponentConfig]) -> LossBuilder:
    losses = LossConfigSchema(output_domain="kspace", **lists)
    builder = LossBuilder(_ConfigShim(losses), device="cpu")  # type: ignore[arg-type]
    builder._build_list_based_losses()
    return builder


def _entry(name: str, **kwargs: object) -> LossComponentConfig:
    return LossComponentConfig(name=name, weight=1.0, enabled=True, kwargs=kwargs)


@pytest.mark.parametrize("name", SELF_BRIDGING)
def test_self_bridging_loss_under_image_losses_is_rejected(name: str) -> None:
    """#467: the double bridge must fail loud, not produce a finite wrong number."""
    with pytest.raises(ConfigurationError) as exc:
        _build(image_losses=[_entry(name)])

    msg = str(exc.value)
    assert "inverse-transformed twice" in msg, msg
    # The message must name the fix, not just the symptom (pitfall #9).
    assert "losses.kspace_losses" in msg, msg
    assert "use_fourier_bridge" in msg, msg
    # It must NOT be buried under the generic "Failed to create <name> loss".
    assert not msg.startswith("Failed to create"), msg


@pytest.mark.parametrize("name", SELF_BRIDGING)
def test_self_bridging_loss_under_complex_losses_is_rejected(name: str) -> None:
    """``complex_losses`` bridges too (return_complex=True) — same double iFFT."""
    with pytest.raises(ConfigurationError, match="inverse-transformed twice"):
        _build(complex_losses=[_entry(name)])


@pytest.mark.parametrize("name", SELF_BRIDGING)
def test_self_bridging_loss_under_kspace_losses_is_unwrapped(name: str) -> None:
    """The correct placement: bridge mode 'none', so the loss's own iFFT is the only one."""
    builder = _build(kspace_losses=[_entry(name)])

    loss_fn = builder._losses[name]
    assert type(loss_fn).__name__ != "_BridgedLoss", (
        f"{name} under kspace_losses must be registered unwrapped; got {type(loss_fn).__name__}"
    )
    assert loss_fn.use_fourier_bridge is True


@pytest.mark.parametrize("name", SELF_BRIDGING)
def test_inner_bridge_can_be_disabled_to_keep_the_outer_one(name: str) -> None:
    """The documented escape hatch: one transform, applied by the outer bridge.

    Note the key is ``kwargs:``, not ``config:`` — ``LossComponentConfig`` is
    ``extra="ignore"``, so a ``config:`` block is silently dropped (issue #468).
    """
    builder = _build(image_losses=[_entry(name, use_fourier_bridge=False)])

    loss_fn = builder._losses[name]
    assert type(loss_fn).__name__ == "_BridgedLoss"
    assert loss_fn.bridge.spatial_loss_fn.use_fourier_bridge is False


def test_plain_image_loss_still_gets_the_bridge() -> None:
    """The guard must not disturb a genuine image-domain loss (no false positives)."""
    builder = _build(image_losses=[_entry("l1")])

    loss_fn = builder._losses["l1"]
    assert type(loss_fn).__name__ == "_BridgedLoss"
    assert loss_fn.bridge.return_complex is False


def test_plain_kspace_loss_is_unwrapped() -> None:
    builder = _build(kspace_losses=[_entry("complex_l1")])

    assert type(builder._losses["complex_l1"]).__name__ != "_BridgedLoss"


def test_bridge_matrix_docstring_matches_the_code() -> None:
    """The docstring's bridge matrix drifted from the code once already.

    It claimed ``output_domain=image`` FFT-bridges ``kspace_losses`` and that
    ``complex_image`` FFT-bridges them too; the code sets ``none`` for every list
    in both cases. A wrong matrix is worse than none — it is what makes a
    misplacement look deliberate on review.
    """
    doc = LossBuilder._build_list_based_losses.__doc__ or ""
    assert "FFT bridge (forward)" not in doc, (
        "stale bridge matrix: the code never inserts a forward-FFT bridge"
    )
    assert "inverse-transformed TWICE" in doc, "the self-bridge rule must be documented"


# ─────────────────────────────────────────────────────────────────────────────
# Schema-declared per-loss kwargs reach list-built losses (issue #467 follow-up)
#
# ``losses.reconstruction.log_spectral_skip_fft: true`` was honoured only by the
# LEGACY ``_build_all_dynamic`` path. A list-based arm read it, logged it as
# configured, and then built ``log_spectral`` with the constructor default
# ``skip_fft=False`` — so the loss applied a forward ``fft2c`` to an
# already-k-space tensor, making a log-SPECTRAL penalty into an image-domain
# log-magnitude one. 17 kspace_filling arms carried exactly that.
# ─────────────────────────────────────────────────────────────────────────────


def _build_with_recon(recon: dict, **lists: list[LossComponentConfig]) -> LossBuilder:
    losses = LossConfigSchema(output_domain="kspace", reconstruction=recon, **lists)
    builder = LossBuilder(_ConfigShim(losses), device="cpu")  # type: ignore[arg-type]
    builder._build_list_based_losses()
    return builder


def test_schema_declared_kwargs_reach_list_built_losses() -> None:
    """The knob is declared on the loss schema, not on the list entry."""
    builder = _build_with_recon(
        {"log_spectral_skip_fft": True},
        kspace_losses=[LossComponentConfig(name="log_spectral", weight=0.1, enabled=True)],
    )

    assert builder._losses["log_spectral"].skip_fft is True, (
        "losses.reconstruction.log_spectral_skip_fft was dropped, so log_spectral "
        "would re-apply fft2c to an already-k-space tensor"
    )


def test_schema_kwargs_default_is_still_respected() -> None:
    """Not declaring it must leave the constructor default alone."""
    builder = _build_with_recon(
        {"log_spectral_skip_fft": False},
        kspace_losses=[LossComponentConfig(name="log_spectral", weight=0.1, enabled=True)],
    )

    assert builder._losses["log_spectral"].skip_fft is False


def test_per_entry_kwargs_win_over_the_schema() -> None:
    """A ``kwargs:`` on the list entry is an override, not a tie."""
    builder = _build_with_recon(
        {"log_spectral_skip_fft": True},
        kspace_losses=[
            LossComponentConfig(
                name="log_spectral", weight=0.1, enabled=True, kwargs={"skip_fft": False}
            )
        ],
    )

    assert builder._losses["log_spectral"].skip_fft is False


def test_schema_loss_kwargs_is_a_single_source_of_truth() -> None:
    """Both build paths must read the same map.

    The legacy path built its ``kwargs_map`` inline; duplicating that mapping for
    the list path would be two sources of truth for the same knob (the failure
    mode CLAUDE.md #13b describes for loss WEIGHTS).
    """
    import inspect

    src = inspect.getsource(LossBuilder._build_all_dynamic)
    assert "kwargs_map = self._schema_loss_kwargs()" in src, (
        "the legacy path must consume the shared SSOT, not rebuild the map inline"
    )
    assert 'kwargs_map["log_spectral"]' not in src, "inline map still present"


class TestPerEntryKwargsReachTheConstructor:
    """A kwarg that never reaches the constructor changes the objective silently.

    `sobolev_order: 1` was declared by 56 arms, dropped by `extra="ignore"`, and
    deleted as dump-identical -- because `SobolevKSpaceLoss` had no `order`
    parameter at all (#560, #615). Nothing in any artifact said so.

    Posture is RAISE, not drop-and-record, and that is a measured choice: across
    the loadable corpus 33 loss entries declare a non-empty `kwargs:` and all 33
    keys reach their constructor, and all 19 schema-derived `kwargs_map` entries
    do too. At zero violations the raise costs nothing today, and it puts losses
    beside `scheduler_resolution`, which raises on an unroutable knob.
    """

    def test_unknown_kwarg_raises_naming_the_loss_and_the_accepted_set(self):
        with pytest.raises(ConfigurationError) as exc:
            _build(
                kspace_losses=[
                    LossComponentConfig(
                        name="sobolev_kspace",
                        weight=1.0,
                        kwargs={"sobolev_order": 1},
                    )
                ]
            )
        msg = str(exc.value)
        assert "sobolev_kspace" in msg
        assert "sobolev_order" in msg
        assert "reduction" in msg, "the error must say what the ctor DOES accept"

    def test_valid_kwarg_reaches_the_constructor(self):
        builder = _build(
            kspace_losses=[
                LossComponentConfig(
                    name="sobolev_kspace", weight=1.0, kwargs={"reduction": "sum"}
                )
            ]
        )
        assert builder._losses["sobolev_kspace"].reduction == "sum"

    def test_var_kwargs_loss_is_exempt(self):
        """`focal_frequency` owns a **kwargs __init__, so it accepts everything;
        raising would reject keys it may legitimately read via kwargs.get()."""
        _build(
            image_losses=[
                LossComponentConfig(
                    name="focal_frequency", weight=1.0, kwargs={"anything_at_all": 1}
                )
            ]
        )

    def test_entry_without_kwargs_is_unaffected(self):
        builder = _build(
            kspace_losses=[LossComponentConfig(name="sobolev_kspace", weight=1.0)]
        )
        assert "sobolev_kspace" in builder._losses


# ---------------------------------------------------------------------------
# #2338: every k-space -> image step on the loss path decompresses the arm's
# log1p-compressed k-space. The builder sets `log_scaled` from the data spec and
# refuses a self-bridging loss that cannot take it.
# ---------------------------------------------------------------------------


class _DataShim:
    def __init__(self, *, log_scaling: bool, enabled: bool = True) -> None:
        self.processing = type(
            "P",
            (),
            {
                "enable_kspace_normalization": enabled,
                "kspace_percentile": 0.95,
                "kspace_scale_domain": "image",
                "enable_log_scaling": log_scaling,
                "log_scaling_center_fraction": 0.25,
            },
        )()


def _build_with_data(data, **lists: list[LossComponentConfig]) -> dict:
    losses = LossConfigSchema(output_domain="kspace", **lists)
    shim = _ConfigShim(losses)
    shim.data = data
    builder = LossBuilder(shim, device="cpu")  # type: ignore[arg-type]
    builder._build_list_based_losses()
    return builder.build()


_KSPACE_TO_IMAGE_STACK = {
    "kspace_losses": [_entry("complex_spatial_gradient"), _entry("sense_adjoint_l1")],
    "image_losses": [_entry("hfen")],
    "complex_losses": [_entry("coil_subspace_residual", input_domain="image")],
}


def _flags(built: dict) -> dict[str, list[bool]]:
    """``log_scaled`` of every module that actually moves k-space to the image.

    A loss behind the builder's bridge with ``input_domain: image`` carries the
    attribute but performs no transform, so it is not a step and is not listed.
    """
    from spectramr.models.losses.physics_losses import DifferentiableFourierBridge

    return {
        name: [
            m.log_scaled
            for m in loss.modules()
            if hasattr(m, "log_scaled")
            and (
                isinstance(m, DifferentiableFourierBridge)
                or getattr(m, "use_fourier_bridge", False) is True
            )
        ]
        for name, loss in built.items()
    }


def test_a_log_scaled_arm_flags_every_kspace_to_image_step() -> None:
    flags = _flags(_build_with_data(_DataShim(log_scaling=True), **_KSPACE_TO_IMAGE_STACK))
    assert set(flags) == {
        "complex_spatial_gradient",
        "sense_adjoint_l1",
        "hfen",
        "coil_subspace_residual",
    }
    for name, values in flags.items():
        assert values and all(values), (name, values)


@pytest.mark.parametrize(
    "data", [_DataShim(log_scaling=False), _DataShim(log_scaling=True, enabled=False), None]
)
def test_an_uncompressed_arm_leaves_every_step_unflagged(data) -> None:
    """No log scaling, normalisation off (log never applied), or no data block."""
    for name, values in _flags(_build_with_data(data, **_KSPACE_TO_IMAGE_STACK)).items():
        assert values and not any(values), (name, values)


def test_a_self_bridging_loss_without_the_flag_is_refused() -> None:
    """Planted: a loss that iFFTs by itself but cannot be told the plane is compressed."""
    import torch

    class _BlindBridge(torch.nn.Module):
        use_fourier_bridge = True

        def forward(self, pred, target):
            return (pred - target).abs().mean()

    losses = LossConfigSchema(output_domain="kspace", kspace_losses=[_entry("complex_l1")])
    shim = _ConfigShim(losses)
    shim.data = _DataShim(log_scaling=True)
    builder = LossBuilder(shim, device="cpu")  # type: ignore[arg-type]
    builder._build_list_based_losses()
    builder._losses["blind"] = _BlindBridge()
    with pytest.raises(ConfigurationError, match="log_scaled"):
        builder.build()


def test_a_loss_that_owns_a_bridge_child_is_covered_through_the_child() -> None:
    """ComplexGradientLoss has no flag of its own; its bridge child carries it."""
    built = _build_with_data(
        _DataShim(log_scaling=True), kspace_losses=[_entry("complex_spatial_gradient")]
    )
    loss = built["complex_spatial_gradient"]
    assert not hasattr(loss, "log_scaled")
    assert loss.bridge.log_scaled is True


def test_the_bridge_wrapper_table_covers_exactly_the_two_bridged_modes() -> None:
    """The mode -> (return_complex, label) table replaced an if/elif; 'none' stays native."""
    from spectramr.infrastructure.training.builders.loss_builder import _BRIDGE_WRAPPERS

    assert _BRIDGE_WRAPPERS == {
        "ifft_magnitude": (False, "iFFT→magnitude bridge"),
        "ifft_complex": (True, "iFFT→complex bridge"),
    }
