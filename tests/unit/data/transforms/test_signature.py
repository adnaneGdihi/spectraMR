"""Transform-chain signatures and the train<->infer parity gate.

``enforce_train_infer_parity`` is the single implementation of the
``data.modes.infer.strict_train_parity`` check. It used to exist twice: once
inline in ``pipelines/infer.py`` (live, untested) and once in
``DataPipelineDirector.build_inference_handle`` (dead, carrying the entire
test suite). Behavioural coverage of the gate itself lives in
``tests/unit/data/test_strict_train_parity_enforcement.py``; this file covers
the signature primitives.
"""

from __future__ import annotations

from spectramr.data.transforms.signature import (
    compute_infer_signature,
    diff_signatures,
)


def _settings():
    from spectramr.config.schemas.data import DataConfigSchema

    class _Settings:
        def __init__(self):
            self.data = DataConfigSchema()
            self.undersampling = None

    return _Settings()


def test_diff_signatures_equal_is_none() -> None:
    assert diff_signatures("a" * 64, "a" * 64) is None


def test_diff_signatures_both_none_is_none() -> None:
    """No signature at either end ⇒ no constraint to enforce."""
    assert diff_signatures(None, None) is None


def test_diff_signatures_missing_train_side_reports_missing() -> None:
    """A pre-Phase-2 checkpoint must be named as such, not silently passed."""
    msg = diff_signatures(None, "b" * 64)
    assert msg is not None and "missing" in msg


def test_diff_signatures_divergence_names_both_sides() -> None:
    msg = diff_signatures("a" * 64, "b" * 64)
    assert msg is not None
    assert "aaaaaaaa" in msg and "bbbbbbbb" in msg


def test_compute_infer_signature_is_deterministic() -> None:
    """Two computations over the same config agree."""
    first = compute_infer_signature(_settings())
    second = compute_infer_signature(_settings())
    assert first == second
    assert len(first) == 64


def test_compute_infer_signature_tracks_the_config() -> None:
    """A config change must move the signature, or parity cannot detect drift.

    Guards the premise the whole gate rests on: a signature that ignored the
    config would compare equal forever and the check would be decorative.
    """
    from spectramr.config.schemas.data import DataConfigSchema, DataSamplingConfigSchema

    baseline = compute_infer_signature(_settings())

    class _Changed:
        def __init__(self):
            self.data = DataConfigSchema(sampling=DataSamplingConfigSchema(patch_size=(16, 16, 1)))
            self.undersampling = None

    assert compute_infer_signature(_Changed()) != baseline


def test_the_signature_sees_the_digital_twin_and_stays_deterministic() -> None:
    """``physics`` used to be dropped on the way to the builder, so a twin arm's
    signature equalled the no-twin one."""
    from spectramr.config.schemas.physics import PhysicsConfigSchema

    def _twin():
        s = _settings()
        s.physics = PhysicsConfigSchema(digital_twin={"apply_as_transform": True})
        return s

    first = compute_infer_signature(_twin())
    assert first == compute_infer_signature(_twin())
    assert first != compute_infer_signature(_settings())
