"""Domain inference must read the leaves where the schema actually keeps them.

Three leaves the priority cascade consults moved when the ``data:`` block was
decomposed, and all three were still read flat:

==============================  ===============================================
legacy (what the code read)     canonical (where the value actually is)
==============================  ===============================================
``data.output_domain``          ``data.domain.output``
``data.normalize_kspace``       ``data.processing.enable_kspace_normalization``
``data.coil_processing_mode``   ``data.coils.processing_mode``
==============================  ===============================================

``_get_attr_safe(..., default)`` does not raise on a moved leaf -- it returns the
default. So on every real (Pydantic) config the P6 guard saw ``""`` / ``False``
and was **inert**, and an ``rss_image`` target (already an image) was reported as
needing an IFFT: the mirror-image regression ``_ensure_image_domain_target``
exists to prevent. Measured on ``exp_prcc_bloch_field`` before the fix::

    cfg.data.coils.processing_mode                   -> 'rss_image'
    _get_attr_safe(cfg.data, 'coil_processing_mode') -> ''
    needs_ifft_for_visualization(cfg)                -> (True, True)   # wrong

It hid because the unit tests stubbed ``data`` with a hand-rolled namespace
spelling the leaves FLAT -- the same spelling the code read. Test and code were
consistently wrong, so the suite was green. Driving the shared ``DataConfigStub``,
which puts the leaves where the schema does, is what exposed it.

P4 is the asymmetric one and needs its own cover: the retired flat spelling
defaulted to ``None`` while its canonical home defaults to ``'image'``, so a
naive repair would make P4 answer for *every* config and leave P5/P6/P7 dead.
Only ``model_fields_set`` separates a declaration from a default.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.utils.domain_inference import (
    IMAGE_DOMAIN_COIL_MODES,
    infer_output_domain,
    metric_transform_produced_image,
    needs_ifft_for_visualization,
)
from tests.utils.corpus import tracked_yamls
from tests.utils.data_config_stub import DataConfigStub

EXPERIMENTS = Path(__file__).resolve().parents[5] / "experiments"

KSPACE_EMITTING_COIL_MODES = ("svd", "complex_sum")


def _config(**data_kwargs) -> SimpleNamespace:
    """A config whose ``data`` block has the real, decomposed shape.

    ``model`` and ``physics`` are ``None`` on purpose: a named model such as
    ``standard_unet`` sits in ``KNOWN_IMAGE_OUTPUT_MODELS`` and short-circuits
    the cascade at P3, leaving the data-side priorities under test unreached.
    """
    return SimpleNamespace(
        data=DataConfigStub(**data_kwargs),
        model=None,
        physics=None,
    )


@pytest.mark.unit
@pytest.mark.parametrize("coil_mode", sorted(IMAGE_DOMAIN_COIL_MODES))
def test_image_domain_coil_mode_overrides_kspace_dataset_type(coil_mode: str) -> None:
    """``dataset_type: kspace`` + an image-emitting coil mode is IMAGE domain.

    Both modes IFFT inside the dataset's TorchIO pipeline, so the target that
    reaches the visualiser is already an image. Reporting it as k-space is what
    produced the spectral / tiled-noise fakes. This is the assertion that was
    silently false: the mode lives under ``coils:`` and the reader looked for it
    flat, so it never matched.
    """
    config = _config(dataset_type="kspace", coil_processing_mode=coil_mode)

    assert config.data.coils.processing_mode == coil_mode, "stub must be canonical"
    assert infer_output_domain(config) == "image"
    assert needs_ifft_for_visualization(config) == (False, False)


@pytest.mark.unit
@pytest.mark.parametrize("coil_mode", KSPACE_EMITTING_COIL_MODES)
def test_kspace_dataset_without_an_image_coil_mode_stays_kspace(coil_mode: str) -> None:
    """The mirror case: a k-space-emitting mode must still be IFFT'd.

    Without it, "never IFFT" would satisfy the test above -- the failure mode
    the original guard was written against, inverted.
    """
    config = _config(dataset_type="kspace", coil_processing_mode=coil_mode)

    assert infer_output_domain(config) == "kspace"
    assert needs_ifft_for_visualization(config)[1] is True


@pytest.mark.unit
def test_kspace_normalization_marks_targets_as_kspace() -> None:
    """``processing.enable_kspace_normalization`` drives the k-space decision.

    Read flat as ``data.normalize_kspace`` it defaulted to ``False`` forever, so
    an image-``dataset_type`` arm that normalises k-space was misclassified.
    """
    config = _config(dataset_type="nifti_paired", normalize_kspace=True)

    assert config.data.processing.enable_kspace_normalization is True
    assert infer_output_domain(config) == "kspace"
    assert needs_ifft_for_visualization(config)[1] is True


@pytest.mark.unit
@pytest.mark.parametrize("declared", ["image", "kspace"])
def test_data_domain_output_declaration_wins_over_the_heuristic(declared: str) -> None:
    """P4 reads ``data.domain.output``; flat ``data.output_domain`` is retired.

    Declared ``image`` against ``dataset_type: kspace`` is the discriminating
    case -- the P6 heuristic would answer ``kspace``, so a passing assertion
    here proves P4 ran at all.
    """
    config = _config(dataset_type="kspace", output_domain=declared)

    assert infer_output_domain(config) == declared


@pytest.mark.unit
def test_undeclared_output_domain_does_not_short_circuit_the_cascade() -> None:
    """P4 must read a DECLARATION, not the schema default.

    ``data.domain.output`` defaults to ``'image'`` where the retired flat
    ``data.output_domain`` defaulted to ``None``. Reading it plainly would make
    P4 answer for every config and leave P5/P6/P7 dead -- so a silent arm whose
    ``dataset_type`` is k-space must still reach P6 and come back ``kspace``.
    """
    config = _config(dataset_type="kspace")

    assert "output" not in config.data.domain.model_fields_set
    assert config.data.domain.output == "image"  # the default P4 must ignore
    assert infer_output_domain(config) == "kspace"


@pytest.mark.unit
def test_a_real_arm_keeps_the_output_domain_its_yaml_declares() -> None:
    """Resolved through the real loader and fold, not a stand-in.

    ``_get_declared_path`` consults ``model_fields_set``, and every other test
    here drives ``DataConfigStub``, which sets fields via ``model_copy``. That
    proves nothing about the production path: if the RENAMES fold populated
    ``data.domain.output`` in a way that left ``model_fields_set`` empty, P4
    would ignore an explicit declaration and P6 would answer over the top of the
    user.

    Checks EVERY declaring arm, not the first. An earlier version returned after
    one and still claimed a corpus-wide result in this docstring — a measurement
    claim a single sample cannot support, which is the facade pattern this file
    exists to guard, one layer up. Both spellings count: the legacy
    ``data.output_domain`` (which folds) and the canonical
    ``data.domain.output``.
    """
    import yaml

    arms = tracked_yamls(EXPERIMENTS / "inprogress")
    if not arms:
        pytest.skip("no inprogress arms present")

    checked = 0
    for arm in arms:
        try:
            document = yaml.safe_load(arm.read_text())
        except Exception:  # a malformed arm is another test's problem
            continue
        if not isinstance(document, dict):
            continue
        data = document.get("data") or {}
        domain = data.get("domain")
        if "output_domain" in data:
            declared = data["output_domain"]
        elif isinstance(domain, dict) and "output" in domain:
            declared = domain["output"]
        else:
            continue

        settings = TrainingSettings.from_yaml(str(arm))
        assert "output" in settings.data.domain.model_fields_set, arm
        assert settings.data.domain.output == declared, arm
        checked += 1

    if not checked:
        pytest.skip("no arm declares an output domain")
    assert checked >= 50, (
        f"only {checked} arms declare an output domain — the corpus shrank far "
        "below the 72 measured on 2026-08-08; this test is going vacuous"
    )


@pytest.mark.unit
def test_a_flat_only_config_still_resolves() -> None:
    """Duck-typed stubs spelling the leaves flat keep working.

    The module is deliberately tolerant of non-Pydantic stand-ins, so the
    canonical-first read falls back rather than hard-failing. A real config never
    reaches this path -- the loader folds the legacy spelling into the block.
    """
    config = SimpleNamespace(
        data=SimpleNamespace(
            dataset_type="kspace",
            coil_processing_mode="rss_image",
        ),
        model=None,
        physics=None,
    )

    assert needs_ifft_for_visualization(config) == (False, False)


@pytest.mark.unit
def test_absent_data_block_does_not_raise() -> None:
    """The helper is used on partially-built configs during the audit probe."""
    config = SimpleNamespace(data=None, model=None, physics=None)

    assert infer_output_domain(config) == "image"
    assert needs_ifft_for_visualization(config) == (False, False)


# ---------------------------------------------------------------------------
# metric_transform_produced_image (#927)
# ---------------------------------------------------------------------------
#
# ``_apply_metric_transforms`` has FOUR paths that return their input
# unchanged: ``metrics.domain == "none"``, ``validation.domain == "none"``, no
# ``transform_name`` resolvable anywhere, and a non-complex non-2-channel
# prediction with no configured transform. The diffusion validation path used to
# set ``is_preds_image = True`` unconditionally after calling it, so on a no-op
# the flag was a lie. Measured consequence on
# ``experiment_11_attention_none`` (cluster run 2026-08-08): the tensor came back
# ``(36, 8, 256, 256)`` float32 -- still k-space -- and every one of 135
# validation-image writes hit the ``kspace_to_image(already_image=True)`` guard
# and raised, so the run produced ZERO images while PSNR (computed on k-space)
# read 58 dB and ``robust_mri_psnr`` went NaN.
#
# The postcondition these pin: every metric transform in this codebase ends in a
# magnitude, so it changes the tensor's shape (coil collapse) or its dtype
# (complex -> real). Something that changes neither did not convert a domain.


@pytest.mark.unit
def test_a_noop_transform_returning_its_input_is_not_image_domain() -> None:
    """The exact #927 shape: 8-channel real k-space handed straight back."""
    kspace = torch.zeros(36, 8, 256, 256)

    assert metric_transform_produced_image(kspace, kspace) is False


@pytest.mark.unit
def test_an_unchanged_copy_is_not_image_domain() -> None:
    """A new object is not evidence of a conversion; shape+dtype must move."""
    kspace = torch.zeros(2, 8, 64, 64)

    assert metric_transform_produced_image(kspace, kspace.clone()) is False


@pytest.mark.unit
def test_a_coil_collapsing_transform_is_image_domain() -> None:
    """``ifft_magnitude`` / ``ifft_sense_adjoint`` RSS the coils away."""
    kspace = torch.zeros(2, 8, 64, 64)
    image = torch.zeros(2, 1, 64, 64)

    assert metric_transform_produced_image(kspace, image) is True


@pytest.mark.unit
def test_a_magnitude_transform_of_complex_is_image_domain() -> None:
    """``magnitude`` keeps the shape but drops complex -> real."""
    kspace = torch.zeros(2, 4, 64, 64, dtype=torch.complex64)
    image = torch.zeros(2, 4, 64, 64)

    assert metric_transform_produced_image(kspace, image) is True


@pytest.mark.unit
def test_per_coil_sense_adjoint_without_smaps_is_image_domain() -> None:
    """``sense_adjoint(smaps=None)`` skips the coil combine but still IFFTs.

    8 interleaved real channels pair into 4 complex coils, so the returned
    magnitude has 4 channels: the shape moved, and it IS image domain.
    """
    kspace = torch.zeros(2, 8, 64, 64)
    per_coil_image = torch.zeros(2, 4, 64, 64)

    assert metric_transform_produced_image(kspace, per_coil_image) is True


@pytest.mark.unit
def test_a_non_tensor_result_does_not_raise() -> None:
    """The helper runs inside validation; it must never be the thing that fails."""
    assert metric_transform_produced_image(None, None) is False
    assert metric_transform_produced_image(object(), object()) is True


class TestDeclarationsOutrankLegacyTables:
    """``data.domain.output`` beats the hardcoded model-name sets (#986).

    The two were the other way round: a table of model names decided the output
    domain even when the arm declared it. That is the shape #977 found one tier
    up and #937 found downstream -- a mechanism speaking for a model other than
    the one configured.

    Measured over all 725 loadable arms in ``experiments/``, the swap changes 0
    verdicts, so these tests are the only thing that exercises it. Without them
    the reorder is unobservable and free to be undone.
    """

    @staticmethod
    def _cfg(model_type: str, declared: str):
        from types import SimpleNamespace

        return SimpleNamespace(
            model=SimpleNamespace(model_type=model_type, target_domain=None),
            data=SimpleNamespace(domain=SimpleNamespace(output=declared)),
            physics=None,
        )

    def test_declared_image_beats_the_kspace_legacy_set(self):
        from spectramr.infrastructure.training.utils.domain_inference import (
            KNOWN_KSPACE_OUTPUT_MODELS,
            infer_output_domain,
        )

        legacy_kspace = sorted(KNOWN_KSPACE_OUTPUT_MODELS)[0]
        assert infer_output_domain(self._cfg(legacy_kspace, "image")) == "image"

    def test_declared_kspace_beats_the_image_legacy_set(self):
        from spectramr.infrastructure.training.utils.domain_inference import (
            KNOWN_IMAGE_OUTPUT_MODELS,
            infer_output_domain,
        )

        legacy_image = sorted(KNOWN_IMAGE_OUTPUT_MODELS)[0]
        assert infer_output_domain(self._cfg(legacy_image, "kspace")) == "kspace"

    def test_the_legacy_set_still_answers_when_nothing_is_declared(self):
        """The reorder must not disable tier 4, only rank it."""
        from types import SimpleNamespace

        from spectramr.infrastructure.training.utils.domain_inference import (
            KNOWN_KSPACE_OUTPUT_MODELS,
            infer_output_domain,
        )

        legacy_kspace = sorted(KNOWN_KSPACE_OUTPUT_MODELS)[0]
        cfg = SimpleNamespace(
            model=SimpleNamespace(model_type=legacy_kspace, target_domain=None),
            data=SimpleNamespace(domain=SimpleNamespace(output=None)),
            physics=None,
        )
        assert infer_output_domain(cfg) == "kspace"

    def test_model_target_domain_still_outranks_both(self):
        """Tier 1 is unmoved; the swap is strictly between 3 and 4."""
        from types import SimpleNamespace

        from spectramr.infrastructure.training.utils.domain_inference import (
            KNOWN_KSPACE_OUTPUT_MODELS,
            infer_output_domain,
        )

        cfg = SimpleNamespace(
            model=SimpleNamespace(
                model_type=sorted(KNOWN_KSPACE_OUTPUT_MODELS)[0],
                target_domain="image",
            ),
            data=SimpleNamespace(domain=SimpleNamespace(output="kspace")),
            physics=None,
        )
        assert infer_output_domain(cfg) == "image"


class TestTheKspaceToImageBridgeHasAnOwner:
    """``needs_kspace_to_image_bridge`` answers "must the strategy apply A^H?".

    Until it existed the answer rode on ``use_dc``, so a k-space arm over a plain
    U-Net with no DC layer got raw k-space into its ``forward`` and rendered the
    DC-spike blob (``ei_unet_m4raw_r4``, cluster run 2026-09-20).
    """

    @staticmethod
    def _cfg(model_type: str, *, dataset_type: str = "kspace", coil_mode: str = "none"):
        from types import SimpleNamespace

        return SimpleNamespace(
            model=SimpleNamespace(model_type=model_type),
            data=SimpleNamespace(
                dataset_type=dataset_type,
                processing=SimpleNamespace(enable_kspace_normalization=True),
                coils=SimpleNamespace(processing_mode=coil_mode),
            ),
            physics=None,
        )

    def test_an_image_domain_model_over_a_kspace_loader_needs_it(self):
        from spectramr.infrastructure.training.utils.domain_inference import (
            needs_kspace_to_image_bridge,
        )
        from spectramr.models.init_registry import populate_model_registry

        populate_model_registry()
        assert needs_kspace_to_image_bridge(self._cfg("standard_unet")) is True

    def test_a_kspace_native_model_does_not(self):
        """``complex_unet`` declares ``input_domain='kspace'`` — the n2n backbone."""
        from spectramr.infrastructure.training.utils.domain_inference import (
            needs_kspace_to_image_bridge,
        )
        from spectramr.models.init_registry import populate_model_registry

        populate_model_registry()
        assert needs_kspace_to_image_bridge(self._cfg("complex_unet")) is False

    def test_an_unannotated_model_is_not_guessed_at(self):
        """443 of 593 registered models declare no ``input_domain``; "nobody said"
        must not resolve to "image"."""
        from spectramr.infrastructure.training.utils.domain_inference import (
            needs_kspace_to_image_bridge,
        )
        from spectramr.models.init_registry import populate_model_registry

        populate_model_registry()
        assert needs_kspace_to_image_bridge(self._cfg("varnet")) is False

    def test_an_image_domain_coil_mode_vetoes_it(self):
        """``rss_image`` IFFTs inside the dataset's TorchIO chain, so the batch is
        already an image even though ``dataset_type`` still reads kspace."""
        from spectramr.infrastructure.training.utils.domain_inference import (
            needs_kspace_to_image_bridge,
        )
        from spectramr.models.init_registry import populate_model_registry

        populate_model_registry()
        cfg = self._cfg("standard_unet", coil_mode="rss_image")
        assert needs_kspace_to_image_bridge(cfg) is False

    def test_loader_serves_kspace_is_the_one_owner_the_visualiser_also_reads(self):
        """Three call sites recomputed this predicate inline from the same four
        leaves. One owner, so they cannot drift (non-negotiable 17)."""
        from spectramr.infrastructure.training.utils.domain_inference import (
            loader_serves_kspace,
            needs_ifft_for_visualization,
        )

        for coil_mode in ("none", "rss_image"):
            cfg = self._cfg("standard_unet", coil_mode=coil_mode)
            assert needs_ifft_for_visualization(cfg)[1] is loader_serves_kspace(cfg)
