"""The objective's gradient must reach the generator's PARAMETERS, not merely exist.

``loss.requires_grad`` is not the property that matters. #1952 records the shape:
``.detach().requires_grad_(True)`` answers ``True`` to it and is a leaf with no
path to any parameter, so ``backward()`` succeeds, every ``p.grad`` stays ``None``,
``optimizer.step()`` moves nothing, and the run writes checkpoints having learned
nothing. #1896 records its sibling — a *connected* graph multiplied by a weight of
0.0, where ``requires_grad`` and ``grad_fn`` both look healthy and every gradient is
exactly zero.

The n2n cohort shipped in the second state. So these tests assert on
``p.grad``, per parameter tensor, after a real ``backward()``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import FFTTransformer, fft2c


@pytest.fixture(scope="module")
def registry() -> None:
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()


def _grad_census(model: torch.nn.Module) -> tuple[int, int, float]:
    """``(tensors with a nonzero grad, total tensors, sum |grad|)``."""
    params = list(model.parameters())
    nonzero = sum(1 for p in params if p.grad is not None and float(p.grad.abs().sum()) > 0)
    total = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
    return nonzero, len(params), total


class TestTheEquivariantImagingObjectiveReachesTheUNet:
    """EI's loss is strategy-owned: a measurement-consistency anchor plus an
    equivariance term over two forward passes. Neither goes through the loss
    builder, so nothing else asserts it is differentiable end to end.
    """

    @staticmethod
    def _strategy(generator: torch.nn.Module, *, bridge: bool):
        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.physics.group_actions import get_group_action
        from spectramr.infrastructure.training.strategies.equivariant_imaging_strategy import (
            EquivariantImagingStrategy,
        )

        cfg = TrainingSettings.from_yaml(
            "experiments/inprogress/self_supervised/ei_unet_m4raw_r4.yaml"
        )
        s = EquivariantImagingStrategy.__new__(EquivariantImagingStrategy)
        s.config = cfg
        s.env = SimpleNamespace(generator=generator, config=cfg, losses={})
        s.fft_transformer = FFTTransformer(device=torch.device("cpu"))
        s.logging_service = None
        s.state = SimpleNamespace(config=cfg)
        s.group_name, s.alpha, s.n_group_samples = "dihedral", 1.0, 1
        s.robust, s.multicoil_operator, s._gsure = False, False, None
        s._group = get_group_action("dihedral")
        s._rng = torch.Generator(device="cpu").manual_seed(0)
        s._owned_ei_loss = None
        s._ensure_conditioner = lambda _c: None  # conditioning is off on this arm
        s.dc_layer = None  # exactly the state the cluster run had
        s._kspace_bridge = bridge
        s._kspace_bridge_checked = True
        return s

    @staticmethod
    def _batch() -> tuple[torch.Tensor, torch.Tensor]:
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        img = torch.exp(-((yy - 32.0) ** 2 + (xx - 30.0) ** 2) / 300.0).to(torch.complex64)
        k = fft2c(img[None, None])
        mask = torch.zeros(1, 1, 64, 64)
        mask[..., ::4] = 1.0
        mask[..., 28:36] = 1.0  # ACS
        return torch.cat([(k * mask).real, (k * mask).imag], dim=1), mask

    @pytest.mark.parametrize("bridge", [True, False], ids=["bridge-on", "bridge-off"])
    def test_every_parameter_receives_a_nonzero_gradient(self, registry, bridge: bool) -> None:
        """Measured on both sides of the domain fix, because the two questions are
        independent: the bridge changes WHAT the network is fed, not whether the
        objective is connected to it. Pinning both stops a later domain change from
        quietly severing the graph."""
        from spectramr.models.registry import MODEL_REGISTRY

        torch.manual_seed(0)
        gen = MODEL_REGISTRY["standard_unet"]["class"](
            in_channels=2, out_channels=2, features=8
        ).train()
        s = self._strategy(gen, bridge=bridge)
        y, mask = self._batch()

        losses = s._compute_losses_impl(y, y, epoch=0, measured_kspace=y, mask=mask)
        total = losses["g_total_loss"]

        assert total.grad_fn is not None, (
            "the EI total is a graph leaf — #1952's detached-root shape, where "
            "backward() succeeds and no parameter moves"
        )
        total.backward()
        nonzero, count, magnitude = _grad_census(gen)
        assert nonzero == count, f"only {nonzero}/{count} parameter tensors got a gradient"
        assert magnitude > 0

    def test_both_ei_terms_are_differentiable_on_their_own(self, registry) -> None:
        """The anchor alone would train; the equivariance term alone is the half
        that was an inert facade before the strategy existed (pitfall #16)."""
        from spectramr.models.registry import MODEL_REGISTRY

        torch.manual_seed(0)
        gen = MODEL_REGISTRY["standard_unet"]["class"](
            in_channels=2, out_channels=2, features=8
        ).train()
        s = self._strategy(gen, bridge=True)
        y, mask = self._batch()

        # alpha=0 leaves only the consistency anchor; the equivariance term is
        # then the difference between the two runs.
        s.alpha = 0.0
        anchor_only = s._compute_losses_impl(y, y, epoch=0, measured_kspace=y, mask=mask)
        assert float(anchor_only["loss_equivariance"]) > 0, (
            "the equivariance term computed to exactly zero — it is being reported but not measured"
        )
        anchor_only["g_total_loss"].backward()
        assert _grad_census(gen)[0] > 0

        for p in gen.parameters():
            p.grad = None
        s.alpha = 1.0
        both = s._compute_losses_impl(y, y, epoch=0, measured_kspace=y, mask=mask)
        both["g_total_loss"].backward()
        assert _grad_census(gen)[0] > 0
        assert float(both["g_total_loss"]) != float(anchor_only["g_total_loss"]), (
            "alpha has no effect on the total, so the equivariance term is not in it"
        )


class TestTheN2nObjectiveReachesTheBackbone:
    """The shipped cohort's ``kspace_losses: [l1]`` resolved to weight 0.0 for its
    whole run, so this is the regression that matters most here.
    """

    @staticmethod
    def _loss_computer(loss_name: str, *, declare_warmup: bool):
        import tempfile
        from pathlib import Path

        import yaml

        from spectramr.config.settings import TrainingSettings
        from spectramr.models.losses.computers import UnifiedReconstructionLossComputer

        arm = Path("experiments/inprogress/n2n/n2n_a_r2r_single_ex_m4raw.yaml")
        d = yaml.safe_load(arm.read_text())
        d["losses"]["kspace_losses"] = [{"name": loss_name, "weight": 1.0}]
        if declare_warmup:
            d["losses"].setdefault("reconstruction", {})["warmup_losses"] = []
        else:
            d["losses"].pop("reconstruction", None)
        p = Path(tempfile.mkdtemp()) / "arm.yaml"
        p.write_text(yaml.safe_dump(d))
        cfg = TrainingSettings.from_yaml(str(p))
        return UnifiedReconstructionLossComputer(config=cfg, device=torch.device("cpu"))

    @staticmethod
    def _backbone():
        from spectramr.models.registry import MODEL_REGISTRY

        torch.manual_seed(0)
        return MODEL_REGISTRY["complex_unet"]["class"](
            in_channels=8,
            out_channels=8,
            features=[8, 16],
            img_size=(32, 32),
            feature_domain="kspace",
            attention_type="none",
        ).train()

    def _run(self, loss_name: str, declare_warmup: bool):
        gen = self._backbone()
        lc = self._loss_computer(loss_name, declare_warmup=declare_warmup)
        pred = gen(torch.randn(1, 8, 32, 32))
        if isinstance(pred, tuple):
            pred = pred[0]
        fn = torch.nn.L1Loss() if loss_name == "l1" else torch.nn.MSELoss()
        out = lc.compute(
            pred=pred,
            target=torch.randn(1, 8, 32, 32),
            epoch=0,
            iteration=1,
            losses_dict={loss_name: fn},
        )
        out.total.backward()
        return out, _grad_census(gen)

    def test_the_shipped_l1_declaration_moved_no_parameter(self, registry) -> None:
        """The planted violation. ``l1`` is in ``LEGACY_WARMUP_LOSSES`` and
        ``warmup_iterations`` defaults to 1000, against this cohort's
        ``max_iterations: 50`` — so the component is computed, reported, and
        multiplied by zero."""
        out, (nonzero, _, magnitude) = self._run("l1", declare_warmup=False)

        assert float(out.total) == 0.0
        assert out.total.grad_fn is not None, "not a severed graph — a zeroed one"
        assert nonzero == 0, "the shipped arm moved a parameter; the gate is no longer the defect"
        assert magnitude == 0.0
        assert "l1" in out.components and float(out.components["l1"]) > 0, (
            "the component is non-zero and reported, which is what made this invisible"
        )

    @pytest.mark.parametrize("loss_name", ["l1", "l2"])
    def test_declaring_warmup_losses_restores_the_gradient(self, registry, loss_name: str) -> None:
        """``warmup_losses: []`` states the gate instead of inheriting it. Both
        norms are pinned: the cohort ships ``l2``, and an arm that reverts to
        ``l1`` must still train rather than silently stall."""
        out, (nonzero, _, magnitude) = self._run(loss_name, declare_warmup=True)

        assert float(out.total) > 0
        assert nonzero > 0, f"{loss_name} still moves no parameter"
        assert magnitude > 0
