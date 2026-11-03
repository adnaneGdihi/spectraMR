"""Off-grid graph path: nodes from the acquired trajectory, image by adjoint NUFFT (#2142)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchio as tio

pytest.importorskip("torchkbnufft")

from spectramr.data.transforms.non_cartesian import NonCartesianSimulationTransform
from spectramr.infrastructure.training.strategies import offgrid_graph_path as ogp
from spectramr.infrastructure.training.strategies.offgrid_graph_path import (
    OffGridGraphPath,
    select_slice_samples,
)
from spectramr.infrastructure.training.strategies.physics_driven_strategy import (
    PhysicsDrivenTrainingStrategy,
)
from spectramr.models.generators import graph_unet_generator  # noqa: F401  (registers)
from spectramr.models.registry import get_model_class

SIZE = 32


def _phantom() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    disc = (((yy - SIZE // 2) ** 2 + (xx - SIZE // 2) ** 2) < (SIZE // 3) ** 2).float()
    return disc.view(1, SIZE, SIZE, 1)


def _batch_context() -> dict:
    """What the loader delivers for one spiral slice, collated to a batch of one."""
    image = _phantom()
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=image), target=tio.ScalarImage(tensor=image.clone())
    )
    out = NonCartesianSimulationTransform(pattern="spiral", im_size=(SIZE, SIZE))(subject)
    return {
        "hr": out["target"].data[..., 0].unsqueeze(0),
        "measured_kspace": out["measured_kspace"].unsqueeze(0),  # (B, D, C, N)
        "trajectory": out["trajectory"].unsqueeze(0),
        "dcf": out["dcf"].unsqueeze(0),
    }


def _dc_config(enabled: bool = True, method: str = "gradient", weight: float = 1.0):
    dc = SimpleNamespace(enabled=enabled, method=method, weight=weight)
    return SimpleNamespace(physics=SimpleNamespace(data_consistency=dc))


def _tiny_graph_unet(**kwargs):
    cls = get_model_class("graph_unet")
    return cls(
        in_channels=1,
        out_channels=1,
        hidden_dim=8,
        depth=1,
        k_neighbors=4,
        im_size=(SIZE, SIZE),
        node_source="trajectory",
        time_embedding_dim=0,
        **kwargs,
    )


def test_select_slice_samples_picks_the_patch_slice():
    measured = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 1, 4)
    location = torch.tensor([[0, 0, 2, 8, 8, 3], [0, 0, 1, 8, 8, 2]])
    out = select_slice_samples(measured, location, num_images=2)
    assert torch.equal(out[0], measured[0, 2]) and torch.equal(out[1], measured[1, 1])
    assert select_slice_samples(measured, None, num_images=6).shape == (6, 1, 4)
    with pytest.raises(ValueError, match="location"):
        select_slice_samples(measured, None, num_images=2)
    with pytest.raises(ValueError, match="coil"):
        select_slice_samples(torch.zeros(1, 2, 4), None, num_images=1)


def test_unknown_dc_method_raises():
    with pytest.raises(ValueError, match="gradient"):
        OffGridGraphPath(_dc_config(method="projection_2d_consistency"))
    assert OffGridGraphPath(_dc_config(enabled=False, method="anything")).dc_method is None


def test_nodes_carry_the_trajectory_and_the_adjoint_is_on_the_object_scale():
    context = _batch_context()
    seen = {}

    def identity_samples(nodes):
        seen["nodes"] = nodes
        return nodes[..., :2]

    path = OffGridGraphPath(_dc_config(enabled=False))
    image = path(identity_samples, context)
    nodes = seen["nodes"]
    traj = context["trajectory"][0]
    assert nodes.shape == (1, traj.shape[-1], 5)
    assert torch.allclose(nodes[0, :, 2:4], (traj / torch.pi).t())
    assert torch.allclose(nodes[0, :, 4], context["dcf"][0])
    assert image.shape == (1, 1, SIZE, SIZE)
    inside = context["hr"][0, 0] > 0
    ratio = image[0, 0][inside].mean() / context["hr"][0, 0][inside].mean()
    assert 0.3 < float(ratio) < 3.0


@pytest.mark.parametrize(("method", "weight"), [("gradient", 1.0), ("replace", 0.5)])
def test_each_dc_mode_keeps_the_object_scale(method, weight):
    context = _batch_context()
    path = OffGridGraphPath(_dc_config(method=method, weight=weight))
    image = path(lambda n: torch.zeros_like(n[..., :2]), context)
    inside = context["hr"][0, 0] > 0
    ratio = image[0, 0][inside].mean() / context["hr"][0, 0][inside].mean()
    assert 0.1 < float(ratio) < 3.0


def test_hard_replace_raises_because_every_sample_is_acquired():
    with pytest.raises(ValueError, match="never reaches"):
        OffGridGraphPath(_dc_config(method="replace", weight=1.0))


def test_off_resonance_changes_the_measurement_the_nodes_see():
    context = _batch_context()
    captured = []
    path = OffGridGraphPath(_dc_config(enabled=False))
    path(lambda n: captured.append(n) or n[..., :2], context)
    b0 = torch.full((1, 1, SIZE, SIZE), 100.0)
    path(lambda n: captured.append(n) or n[..., :2], context, b0)
    assert not torch.allclose(captured[0][..., :2], captured[1][..., :2])


def test_reachability_registry_to_physics_driven_strategy():
    """registry -> graph_unet(node_source=trajectory) -> PhysicsDriven runs the off-grid path."""
    generator = _tiny_graph_unet()
    env = MagicMock()
    env.device = torch.device("cpu")
    env.generator = generator
    env.models = {"generator": generator}
    env.config.physics.b0_simulation.enabled = False
    env.config.physics.field_strength = 0.3
    env.config.physics.pinn.enabled = False
    env.config.physics.data_consistency = _dc_config().physics.data_consistency
    env.config.optimization.precision.dtype = "float32"
    env.config.optimization.gradient.clip.method = "norm"
    env.config.optimization.gradient.clip.value = 1.0
    env.config.optimization.gradient.activation_offload = "none"
    env.config.metrics.train_metric_interval = 0
    env.config.training.max_iterations = 0
    env.config.model.model_type = "graph_unet"
    strategy = PhysicsDrivenTrainingStrategy(env=env, logging_service=MagicMock())
    assert isinstance(strategy._offgrid_path, OffGridGraphPath)

    context = _batch_context()
    with (
        patch.object(
            ogp, "NonCartesianDataConsistency", wraps=ogp.NonCartesianDataConsistency
        ) as dc,
        patch.object(
            generator, "_trajectory_adjacency", wraps=generator._trajectory_adjacency
        ) as knn,
    ):
        pred, _ = strategy._generate_predictions(torch.zeros(1, 1, SIZE, SIZE), {}, context)
        val = strategy._validation_forward(torch.zeros(1, 1, SIZE, SIZE), context)
    assert pred.shape == val.shape == (1, 1, SIZE, SIZE)
    assert pred.requires_grad
    assert dc.called and knn.called
    key = next(iter(generator._node_adjacency))
    assert key[0] == context["trajectory"].shape[-1]


def test_gat_cap_pools_along_the_readout():
    generator = _tiny_graph_unet(backbone_type="gat_unet", heads=2, gat_max_nodes=100)
    nodes = torch.randn(1, 768, 5)
    out = generator(nodes)
    assert out.shape == (1, 768, 2)
    assert next(iter(generator._node_adjacency))[0] <= 100
