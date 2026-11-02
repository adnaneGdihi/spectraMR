"""graph_unet: exact kNN across chunks, intensity-preserving input, sparse full-res GAT."""

import pytest
import torch

from spectramr.models.generators.graph_unet_generator import (
    GraphAttentionLayer,
    GraphUNetGenerator,
    knn_adjacency,
)


def _lattice(h: int, w: int) -> torch.Tensor:
    ky, kx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([kx, ky], dim=-1).reshape(-1, 2).float()


def _edges(adj: torch.Tensor) -> set[tuple[int, int]]:
    r, c = adj.coalesce().indices()
    return set(zip(r.tolist(), c.tolist(), strict=True))


class TestKnnAdjacency:
    def test_chunking_does_not_change_the_graph(self) -> None:
        pts = torch.rand(300, 2)
        assert _edges(knn_adjacency(pts, k=8, chunk_size=7)) == _edges(
            knn_adjacency(pts, k=8, chunk_size=10_000)
        )

    def test_edges_cross_chunk_boundaries(self) -> None:
        # #2389: the old builder kept every edge inside its 1024-node block.
        r, c = knn_adjacency(_lattice(16, 16), k=8, chunk_size=32).indices()
        assert ((r // 32) != (c // 32)).any()

    def test_a_lattice_interior_node_gets_whole_shells(self) -> None:
        adj = knn_adjacency(_lattice(16, 16), k=32, chunk_size=50)
        r, c = adj.indices()
        centre = 8 * 16 + 8
        # k + 1 = 33 falls inside the 8-point shell |z|^2 = 10; ties keep all 37 points.
        assert int((r == centre).sum()) == 37
        neighbours = set(c[r == centre].tolist())
        assert all(centre in set(c[r == n].tolist()) for n in neighbours)

    def test_complex_coordinates_raise(self) -> None:
        with pytest.raises(ValueError, match="real coordinates"):
            knn_adjacency(torch.rand(10, 2, dtype=torch.complex64), k=3)


def _model(**kw) -> GraphUNetGenerator:
    torch.manual_seed(0)
    base = {
        "in_channels": 1,
        "out_channels": 1,
        "hidden_dim": 8,
        "depth": 1,
        "k_neighbors": 8,
        "im_size": (16, 16),
        "time_embedding_dim": 0,
    }
    return GraphUNetGenerator(**{**base, **kw}).eval()


class TestDynamicBackbone:
    def test_contrast_reaches_the_output(self) -> None:
        # #2390: LayerNorm(2) on [a, 0] mapped every pixel above ~0.05 to one feature.
        m = _model()
        x = torch.full((1, 1, 16, 16), 0.5)
        y = x.clone()
        y[..., :, 8:] = 1.0
        with torch.no_grad():
            a, b = m(x), m(y)
        assert ((a - b).norm() / a.norm()) > 1e-2

    def test_receptive_field_crosses_a_chunk_boundary(self) -> None:
        m = _model(depth=2)
        x = torch.rand(1, 1, 48, 48, requires_grad=True)
        (g,) = torch.autograd.grad(m(x)[0, 0, 21, 24], x)
        rows = (g[0, 0].abs().sum(dim=1) > 0).nonzero().flatten()
        assert rows.min() < 21 < rows.max()

    def test_adjacency_is_cached_per_grid(self) -> None:
        m = _model()
        with torch.no_grad():
            m(torch.rand(1, 1, 16, 16))
            first = next(iter(m._grid_adjacency.values()))[0]
            m(torch.rand(1, 1, 16, 16))
        assert next(iter(m._grid_adjacency.values()))[0] is first

    def test_complex_graph_nodes_raise(self) -> None:
        with pytest.raises(ValueError, match="drop the phase"):
            _model()(torch.rand(1, 10, 5, dtype=torch.complex64))

    def test_more_than_two_grid_channels_raise(self) -> None:
        with pytest.raises(ValueError, match="Combine coils"):
            _model(in_channels=3)(torch.rand(1, 3, 8, 8))


class TestGraphAttention:
    def test_sparse_attention_matches_the_dense_mask(self) -> None:
        torch.manual_seed(0)
        layer = GraphAttentionLayer(6, 4, heads=2)
        x = torch.randn(2, 64, 6)
        adj = knn_adjacency(_lattice(8, 8), k=4)
        sparse = layer(x, adj)
        dense = layer(x, adj.to_dense())
        torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-5)

    def test_full_resolution_by_default(self) -> None:
        # #2391: no silent 64x64 pooling; every output pixel is computed at full res.
        m = _model(backbone_type="gat_unet", heads=2, im_size=(16, 16))
        assert m.gat_max_nodes is None
        x = torch.rand(1, 1, 16, 16, requires_grad=True)
        out = m(x)
        assert out.shape == (1, 1, 16, 16)
        out.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    @pytest.mark.parametrize("bad", [0, -4, 64.0, "4096", True])
    def test_an_invalid_cap_raises(self, bad) -> None:
        with pytest.raises(ValueError, match="gat_max_nodes"):
            _model(backbone_type="gat_unet", heads=2, gat_max_nodes=bad)

    def test_a_declared_cap_pools_and_restores_the_shape(self) -> None:
        m = _model(backbone_type="gat_unet", heads=2, gat_max_nodes=64)
        with torch.no_grad():
            assert m(torch.rand(1, 1, 16, 16)).shape == (1, 1, 16, 16)


class TestTrajectoryNodes:
    """node_source='trajectory': off-grid sample nodes in, one complex sample per node out."""

    def test_an_unknown_node_source_raises(self) -> None:
        with pytest.raises(ValueError, match="node_source"):
            _model(node_source="spiral")

    def test_image_channels_other_than_one_raise(self) -> None:
        with pytest.raises(ValueError, match="out_channels"):
            _model(node_source="trajectory", out_channels=2)

    def test_grid_input_raises(self) -> None:
        m = _model(node_source="trajectory")
        with pytest.raises(ValueError, match=r"\[B, N, 5\]"):
            m(torch.rand(1, 1, 8, 8))

    def test_nodes_return_complex_samples_on_a_cached_knn(self) -> None:
        m = _model(node_source="trajectory")
        nodes = torch.rand(2, 50, 5)
        assert m(nodes).shape == (2, 50, 2)
        first = m._node_adjacency[(50, nodes.device)]
        m(nodes)
        assert m._node_adjacency[(50, nodes.device)] is first

    def test_the_image_probe_and_predict_image_do_not_apply(self) -> None:
        m = _model(node_source="trajectory")
        assert m.synthetic_forward_probe_skip == {"forward"}
        with pytest.raises(NotImplementedError, match="OffGridGraphPath"):
            m.predict_image(torch.rand(1, 10, 2))
