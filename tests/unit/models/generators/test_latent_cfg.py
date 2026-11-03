"""Tests for multi-axis CFG on the latent ULF->HF path (LM-19)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.latent_cfg import (
    LEGACY_CONDITION_DROP,
    axis_keep_masks,
    guided_noise,
    resolve_latent_cfg,
    reverse_progress,
)


def _resolve(drops=None, weights=None, condition=True, contrast=True):
    return resolve_latent_cfg(drops, weights, has_condition=condition, has_contrast=contrast)


class TestResolve:
    def test_absent_knobs_keep_the_legacy_condition_drop_and_no_guidance(self):
        spec = _resolve()
        assert dict(spec.drop_probs) == {"condition": LEGACY_CONDITION_DROP}
        assert not spec.guided

    def test_absent_on_an_unconditional_model_drops_nothing(self):
        assert dict(_resolve(condition=False).drop_probs) == {}

    @pytest.mark.parametrize("axis", ["thickness", "pathology"])
    def test_axes_without_data_raise(self, axis):
        with pytest.raises(ValueError, match="cannot be conditioned on"):
            _resolve({axis: 0.05})

    def test_an_unknown_axis_raises(self):
        with pytest.raises(ValueError, match="unknown axis"):
            _resolve({"vendor": 0.1})

    def test_an_axis_the_model_lacks_raises(self):
        with pytest.raises(ValueError, match="does not condition on 'contrast'"):
            _resolve({"contrast": 0.1}, contrast=False)

    @pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
    def test_a_drop_probability_out_of_range_raises(self, p):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            _resolve({"condition": p})

    def test_guiding_an_undropped_axis_raises(self):
        """Its null branch never trained, so the guidance direction is noise."""
        with pytest.raises(ValueError, match="never dropped"):
            _resolve({"condition": 0.1}, {"contrast": 2.0})

    def test_a_negative_weight_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            _resolve({"condition": 0.1}, {"condition": -1.0})


class TestMasks:
    def test_the_legacy_draw_is_reproduced_bit_for_bit(self):
        """The generator used torch.bernoulli(zeros + 0.9); absent knobs must match it."""
        torch.manual_seed(7)
        legacy = torch.bernoulli(torch.zeros(16) + (1 - 0.1))
        torch.manual_seed(7)
        masks = axis_keep_masks(_resolve(), 16, torch.device("cpu"))
        assert torch.equal(masks["condition"], legacy)
        assert "contrast" not in masks

    def test_each_declared_axis_gets_an_independent_mask(self):
        torch.manual_seed(0)
        masks = axis_keep_masks(
            _resolve({"condition": 0.5, "contrast": 0.5}), 256, torch.device("cpu")
        )
        assert set(masks) == {"condition", "contrast"}
        assert not torch.equal(masks["condition"], masks["contrast"])


class TestGuidedNoise:
    """A stub denoiser whose output is a known function of each input, so the
    combination can be checked by hand."""

    @staticmethod
    def _denoiser(calls):
        def denoise(unet_input, t, keep):
            calls.append(unet_input.shape[0])
            cond_part = unet_input[:, 1:].mean(dim=(1, 2, 3))
            k = torch.zeros_like(cond_part) if keep is None else keep
            out = (cond_part + 10.0 * k).view(-1, 1, 1, 1)
            return out.expand(unet_input.shape[0], 1, 2, 2).contiguous()

        return denoise

    def _run(self, spec, calls):
        z = torch.zeros(2, 1, 2, 2)
        cond = torch.full((2, 1, 2, 2), 3.0)
        return guided_noise(
            self._denoiser(calls), z, torch.zeros(2, dtype=torch.long), cond, True, spec
        )

    def test_single_axis_keeps_contrast_on_in_every_term(self):
        """Single-axis control: eps_null = 0 + 10 (contrast on), eps_cond = 3 + 10."""
        calls = []
        spec = _resolve({"condition": 0.1}, {"condition": 2.0})
        out = self._run(spec, calls)
        assert torch.allclose(out, torch.full_like(out, 10.0 + 2.0 * 3.0))
        assert calls == [4], "null and conditioned passes run as ONE batched call"

    def test_two_axes_combine_from_the_joint_null(self):
        """eps_null = 0, eps_cond = 3, eps_contrast = 10 -> 2*3 + 1.5*10."""
        calls = []
        spec = _resolve({"condition": 0.1, "contrast": 0.1}, {"condition": 2.0, "contrast": 1.5})
        out = self._run(spec, calls)
        assert torch.allclose(out, torch.full_like(out, 2.0 * 3.0 + 1.5 * 10.0))
        assert calls == [6]


class TestSchedule:
    """``cfg_schedule`` multiplies each guided weight by s_i(progress)."""

    def _spec(self, schedule):
        return resolve_latent_cfg(
            {"condition": 0.1, "contrast": 0.1},
            {"condition": 2.0, "contrast": 1.5},
            has_condition=True,
            has_contrast=True,
            schedule=schedule,
        )

    def _run(self, spec, progress):
        z = torch.zeros(2, 1, 2, 2)
        cond = torch.full((2, 1, 2, 2), 3.0)
        return guided_noise(
            TestGuidedNoise._denoiser([]),
            z,
            torch.zeros(2, dtype=torch.long),
            cond,
            True,
            spec,
            progress=progress,
        )

    def test_progress_runs_from_the_noisiest_step_to_the_cleanest(self):
        assert reverse_progress(999, 1000) == 0.0
        assert reverse_progress(0, 1000) == 1.0
        assert reverse_progress(0, 1) == 1.0

    def test_linear_schedule_is_6_at_the_first_reverse_step_and_1_at_the_last(self):
        """eps = 2*s(p)*3 + 1.5*10 with s linear 6 -> 1."""
        spec = self._spec({"condition": {"type": "linear", "start": 6.0, "end": 1.0}})
        first = self._run(spec, reverse_progress(999, 1000))
        last = self._run(spec, reverse_progress(0, 1000))
        assert torch.allclose(first, torch.full_like(first, 2.0 * 6.0 * 3.0 + 15.0))
        assert torch.allclose(last, torch.full_like(last, 2.0 * 1.0 * 3.0 + 15.0))

    def test_no_schedule_matches_the_constant_weight_output_bit_exact(self):
        unscheduled = self._run(self._spec(None), 0.37)
        z = torch.zeros(2, 1, 2, 2)
        legacy = guided_noise(
            TestGuidedNoise._denoiser([]),
            z,
            torch.zeros(2, dtype=torch.long),
            torch.full((2, 1, 2, 2), 3.0),
            True,
            self._spec(None),
        )
        assert torch.equal(unscheduled, legacy)

    def test_scheduling_an_unguided_axis_raises(self):
        with pytest.raises(ValueError, match="does not guide"):
            resolve_latent_cfg(
                {"condition": 0.1, "contrast": 0.1},
                {"condition": 2.0},
                has_condition=True,
                has_contrast=True,
                schedule={"contrast": {"type": "constant"}},
            )

    def test_an_unknown_schedule_type_raises(self):
        with pytest.raises(ValueError, match="Unknown CFG schedule type"):
            self._spec({"condition": {"type": "sigmoid"}})

    def test_the_schedule_reaches_the_record(self):
        spec = self._spec({"contrast": {"type": "cosine", "amplitude": 0.5, "baseline": 1.5}})
        assert spec.to_dict()["cfg_schedule"] == {
            "contrast": {"type": "cosine", "amplitude": 0.5, "baseline": 1.5}
        }
