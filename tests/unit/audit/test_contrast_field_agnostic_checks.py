"""Tests for the contrast/field-agnostic bundle Tier-1 audit checks.

Covers ``acq_vector_present`` / ``spectral_norm_enabled`` (M3 LCAH),
``dispersion_identifiability`` / ``dispersion_monotonicity_weight_positive``
(M4 DL-BAE) and ``mcgi_invariance_declared`` (M2 MCGI) in isolation.

The polarity matters as much as the pass/fail: an arm that is *un-identifiable*
or whose invariance claim is a facade must ERROR (the run would otherwise look
converged while meaning nothing), whereas a deliberate ablation that merely
voids a certificate WARNS.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker

pytestmark = pytest.mark.unit


def _by_name(results):
    return {r.check_name: r for r in results}


def _lcah_cfg(acq: dict | None, *, expose: bool = True, per_class: bool = False):
    """Config stub. ``expose`` drives the per-sample route, ``per_class`` the other."""
    return SimpleNamespace(
        training=SimpleNamespace(acq_hypernetwork=acq),
        data=SimpleNamespace(
            acquisition_metadata=SimpleNamespace(enabled=expose),
            multi_contrast=SimpleNamespace(
                acquisition_params=[{"TE": 80.0}] if per_class else None
            ),
        ),
        model=SimpleNamespace(name="lcah_encoder", model_kwargs={}),
    )


def _dlbae_cfg(dl: dict | None, kwargs: dict | None = None):
    return SimpleNamespace(
        training=SimpleNamespace(dispersion_bloch_ae=dl),
        data=SimpleNamespace(),
        model=SimpleNamespace(name="disp_bloch_ae", model_kwargs=kwargs or {}),
    )


_FIVE = [0.1, 1.5, 3.0, 5.0, 7.0]
_PROTOCOL = {"repetition_times_s": [0.5] * 5, "echo_times_s": [0.015] * 5}


def _mcgi_cfg(model_type: str, kwargs: dict):
    return SimpleNamespace(
        training=SimpleNamespace(),
        data=SimpleNamespace(),
        model=SimpleNamespace(model_type=model_type, model_kwargs=kwargs),
    )


class TestAcqHypernetworkChecks:
    def test_no_checks_when_block_absent(self) -> None:
        assert ConfigHealthChecker().check_acq_hypernetwork_config(_lcah_cfg(None)) == []

    def test_clean_arm_passes_both(self) -> None:
        res = _by_name(
            ConfigHealthChecker().check_acq_hypernetwork_config(
                _lcah_cfg({"spectral_norm": True}, expose=True)
            )
        )
        assert res["acq_vector_present"].passed
        assert res["spectral_norm_enabled"].passed

    def test_missing_acquisition_vector_is_an_error(self) -> None:
        """Conditioning on a vector the pipeline never emits is a dead knob."""
        res = _by_name(
            ConfigHealthChecker().check_acq_hypernetwork_config(
                _lcah_cfg({"spectral_norm": True}, expose=False)
            )
        )
        assert not res["acq_vector_present"].passed
        assert res["acq_vector_present"].severity == "error"

    def test_fixed_per_contrast_params_also_satisfy_the_check(self) -> None:
        """A constant-per-contrast protocol is a legitimate second route."""
        res = _by_name(
            ConfigHealthChecker().check_acq_hypernetwork_config(
                _lcah_cfg({"spectral_norm": True}, expose=False, per_class=True)
            )
        )
        assert res["acq_vector_present"].passed

    def test_spectral_norm_off_only_warns(self) -> None:
        """A certificate-free ablation is legitimate, but must be visible."""
        res = _by_name(
            ConfigHealthChecker().check_acq_hypernetwork_config(
                _lcah_cfg({"spectral_norm": False}, expose=True)
            )
        )
        assert not res["spectral_norm_enabled"].passed
        assert res["spectral_norm_enabled"].severity == "warning"


class TestDispersionBlochAEChecks:
    def test_no_checks_when_block_absent(self) -> None:
        assert ConfigHealthChecker().check_dispersion_bloch_ae_config(_dlbae_cfg(None)) == []

    def test_identifiable_arm_passes(self) -> None:
        res = _by_name(
            ConfigHealthChecker().check_dispersion_bloch_ae_config(
                _dlbae_cfg(
                    {"num_pools": 1, "fields_present": _FIVE, "monotonicity_weight": 1.0},
                    _PROTOCOL,
                )
            )
        )
        assert res["dispersion_identifiability"].passed
        assert res["dispersion_monotonicity_weight_positive"].passed
        assert res["dispersion_protocol_declared"].passed

    @pytest.mark.parametrize(
        ("n_pools", "fields"),
        # (2, five fields) is the dlbae_brain_5field shape the 2P+1 count passed (#2417).
        [(1, [0.3, 1.5, 3.0, 7.0]), (2, _FIVE), (2, [0.055, *_FIVE]), (3, [0.055, 0.3, *_FIVE])],
    )
    def test_under_determined_arm_is_an_error(self, n_pools: int, fields: list) -> None:
        res = _by_name(
            ConfigHealthChecker().check_dispersion_bloch_ae_config(
                _dlbae_cfg({"num_pools": n_pools, "fields_present": fields})
            )
        )
        assert not res["dispersion_identifiability"].passed
        assert res["dispersion_identifiability"].severity == "error"
        assert "reduce num_pools" in res["dispersion_identifiability"].fix_hint

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"echo_times_s": [0.015] * 5},
            {"repetition_times_s": [0.5] * 5},
            {**_PROTOCOL, "echo_times_s": [0.015] * 4},
            {**_PROTOCOL, "repetition_times_s": [0.5, 0.5, 0.0, 0.5, 0.5]},
        ],
        ids=["absent", "no_tr", "no_te", "short", "non_positive"],
    )
    def test_undeclared_protocol_is_an_error(self, kwargs: dict) -> None:
        """#1712: the render has no default TR/TE, so the audit must catch an absent one."""
        res = _by_name(
            ConfigHealthChecker().check_dispersion_bloch_ae_config(
                _dlbae_cfg({"num_pools": 1, "fields_present": _FIVE}, kwargs)
            )
        )
        assert not res["dispersion_protocol_declared"].passed
        assert res["dispersion_protocol_declared"].severity == "error"

    def test_duplicate_fields_do_not_buy_identifiability(self) -> None:
        """The check counts DISTINCT fields; repeats add no rank."""
        res = _by_name(
            ConfigHealthChecker().check_dispersion_bloch_ae_config(
                _dlbae_cfg({"num_pools": 1, "fields_present": [1.5, 1.5, 1.5, 1.5, 1.5]})
            )
        )
        assert not res["dispersion_identifiability"].passed

    def test_zero_monotonicity_weight_warns(self) -> None:
        res = _by_name(
            ConfigHealthChecker().check_dispersion_bloch_ae_config(
                _dlbae_cfg(
                    {
                        "num_pools": 1,
                        "fields_present": [0.3, 1.5, 3.0],
                        "monotonicity_weight": 0.0,
                    }
                )
            )
        )
        assert not res["dispersion_monotonicity_weight_positive"].passed
        assert res["dispersion_monotonicity_weight_positive"].severity == "warning"


class TestMCGIInvarianceCheck:
    def test_skips_when_arm_does_not_select_mcgi(self) -> None:
        res = ConfigHealthChecker().check_mcgi_invariance_declared(
            _mcgi_cfg("unet", {"hard_rank_eval": False})
        )
        assert res.passed and res.severity == "info"

    def test_hard_rank_passes(self) -> None:
        res = ConfigHealthChecker().check_mcgi_invariance_declared(
            _mcgi_cfg("mcgi_encoder", {"hard_rank_eval": True})
        )
        assert res.passed

    def test_default_is_hard_rank(self) -> None:
        res = ConfigHealthChecker().check_mcgi_invariance_declared(
            _mcgi_cfg("mcgi_encoder", {})
        )
        assert res.passed

    @pytest.mark.parametrize(
        "kwargs",
        [{"hard_rank_eval": False}, {"soft_rank_temperature": 0.05}, {"hidden_chanels": 8}],
    )
    def test_planted_illegal_kwargs_are_an_error(self, kwargs: dict) -> None:
        """Planted: the soft rank, the retired knob and a misspelt key turn it red."""
        res = ConfigHealthChecker().check_mcgi_invariance_declared(
            _mcgi_cfg("mcgi_encoder", kwargs)
        )
        assert not res.passed and res.severity == "error"

    def test_rank_off_control_is_info_not_a_warning(self) -> None:
        res = ConfigHealthChecker().check_mcgi_invariance_declared(
            _mcgi_cfg("mcgi_encoder", {"rank_transform": False})
        )
        assert res.passed and res.severity == "info" and "control" in res.message

    def test_reads_model_type_not_model_name(self) -> None:
        """The check was inert: it read `model.name`, which the arms never set."""
        cfg = SimpleNamespace(
            model=SimpleNamespace(name="unet", model_type="mcgi_encoder",
                                  model_kwargs={"hard_rank_eval": False})
        )
        assert not ConfigHealthChecker().check_mcgi_invariance_declared(cfg).passed
