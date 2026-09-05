"""Regression tests for the 2026-05-12 batch of YAML / decorator fixes.

These tests are deliberately lightweight — they read YAMLs and assert
the load-bearing keys are present. No heavy model forward passes.

* **experiment_42c_gat_reconstruction**: image-domain ``graph_unet``
  trained on k-space data. Requires an ``adapters.pre_model`` chain
  that bridges ``kspace → complex_image → image`` (real/imag
  interleaved). The previous mosaic showed a misleading
  ``mat1 4096x2 and mat2 256x1`` error that was actually a
  post-OOM corruption: ``depth=5, hidden_dim=256, gat_max_nodes=4096``
  yielded a dense 4096×4096 attention matrix that exceeded RAM.
  Downsized to ``depth=3, hidden_dim=64, gat_max_nodes=1024``.

* **experiment_20_hybrid_transformer_laplacian_gan**: 8-channel
  ``vit_kan_generator`` for kspace m4raw. The GAN-mode factory route
  does not forward top-level ``model.in_channels`` to the model
  constructor, so the model was built with the default
  ``in_channels=2`` and produced ``Linear(512, 256)`` while the data
  was 8-channel → ``Linear(2048, ...)`` needed. Fix routes through
  the legacy ``channels``/``num_classes`` aliases inside
  ``model_kwargs``.

* **Adapter registry**: ``ifft_kspace_to_image`` and
  ``complex_to_real_imag_interleave`` both gained ``pre_model`` in
  their registered insertion points so they can sit at the front of
  a YAML adapter chain.

* **graph_unet decorator**: ``accepts_complex`` flipped to ``True``
  to match the actual ``_grid_to_graph`` behaviour (line ~1076 splits
  complex into real/imag along the channel dim).
"""

from __future__ import annotations

import yaml

from tests.utils.repo_scripts import require_repo_file


def _load(path: str) -> dict:
    return yaml.safe_load(require_repo_file(path).read_text())


# ---------------------------------------------------------------------------
# experiment_42c_gat_reconstruction
# ---------------------------------------------------------------------------


def test_exp42c_has_kspace_to_image_adapter_chain() -> None:
    cfg = _load("experiments/active/experiment_42c_gat_reconstruction.yaml")
    pre = (cfg.get("adapters") or {}).get("pre_model") or []
    names = [a.get("name") for a in pre]
    assert names == [
        "ifft_kspace_to_image",
        "complex_to_real_imag_interleave",
    ], f"expected the kspace→image bridge chain, got {names}"


def test_exp42c_gat_kwargs_are_memory_safe() -> None:
    """Depth × hidden_dim × node count must keep the dense attention
    matrix tractable. The 2026-05-10 smoke OOM was at depth=5,
    hidden_dim=256, gat_max_nodes=4096 (the default)."""
    cfg = _load("experiments/active/experiment_42c_gat_reconstruction.yaml")
    mkw = (cfg["model"].get("model_kwargs") or {})
    assert mkw["depth"] <= 3, f"depth too large: {mkw['depth']}"
    assert mkw["hidden_dim"] <= 128, f"hidden_dim too large: {mkw['hidden_dim']}"
    # gat_max_nodes must be capped if depth > 1 — otherwise the dense
    # [B, N, N, H] attention matrix blows past 1 GB at N=4096.
    assert mkw.get("gat_max_nodes", 4096) <= 1024, (
        f"gat_max_nodes={mkw.get('gat_max_nodes')} too large for the "
        f"chosen depth/hidden_dim — will OOM at smoke time."
    )


def test_exp42c_no_stale_test_split_key() -> None:
    """v6.0 schema rejects ``data.test_split`` (and ``validation.test_split``).
    The previous YAML had a stale ``data.test_split: 0.1`` that was caught
    in the 2026-05-12 schema sweep."""
    cfg = _load("experiments/active/experiment_42c_gat_reconstruction.yaml")
    assert "test_split" not in cfg.get("data", {})
    assert "test_split" not in cfg.get("validation", {})


# ---------------------------------------------------------------------------
# experiment_20_hybrid_transformer_laplacian_gan
# ---------------------------------------------------------------------------


def test_exp20_vit_channels_override_present() -> None:
    """The GAN-mode factory route does not propagate top-level
    ``model.in_channels`` into ``vit_kan_generator``. The explicit
    ``channels`` / ``num_classes`` keys in ``model_kwargs`` are the
    legacy aliases the model resolves directly — bypassing the
    factory's plumbing question entirely."""
    cfg = _load(
        "experiments/inprogress/transformers/"
        "experiment_20_hybrid_transformer_laplacian_gan.yaml"
    )
    mkw = (cfg["model"].get("model_kwargs") or {})
    assert mkw.get("channels") == cfg["model"]["in_channels"], (
        "model_kwargs.channels must match model.in_channels "
        "(otherwise the GAN factory path constructs the model with "
        "the wrong patch_dim and the smoke crashes with "
        "`mat1 1024x2048 and mat2 512x256`)."
    )
    assert mkw.get("num_classes") == cfg["model"]["out_channels"], (
        "model_kwargs.num_classes must match model.out_channels."
    )


# ---------------------------------------------------------------------------
# Adapter & decorator registration
# ---------------------------------------------------------------------------


def test_ifft_kspace_to_image_supports_pre_model() -> None:
    """The pre_model insertion point was added 2026-05-12 so the
    bridge can sit at the front of an adapter chain."""
    from spectramr.data.adapters import get_adapter_capabilities

    caps = get_adapter_capabilities("ifft_kspace_to_image")
    assert caps is not None, "ifft_kspace_to_image not registered"
    assert "pre_model" in caps.insertion_points, (
        f"ifft_kspace_to_image must support pre_model; "
        f"got insertion_points={caps.insertion_points}"
    )


def test_complex_to_real_imag_interleave_supports_pre_model() -> None:
    """Same fix for the channel adapter that completes the chain."""
    from spectramr.data.adapters import get_adapter_capabilities

    caps = get_adapter_capabilities("complex_to_real_imag_interleave")
    assert caps is not None
    assert "pre_model" in caps.insertion_points


def test_graph_unet_decorator_accepts_complex() -> None:
    """``_grid_to_graph`` (line ~1076 in graph_unet_generator.py)
    handles ``torch.is_complex(grid)`` by splitting real/imag along the
    channel dim, so complex input IS supported. The decorator
    previously claimed ``accepts_complex=False`` and the audit
    rejected pre_model iFFT chains. 2026-05-12 fix."""
    # Force registration by importing the module.
    import spectramr.models.generators.graph_unet_generator  # noqa: F401
    from spectramr.models.registry import get_model_capabilities

    for name in ("graph_unet", "graph_unet_diffusion"):
        caps = get_model_capabilities(name)
        assert caps is not None, f"{name} not registered"
        assert caps.accepts_complex is True, (
            f"{name} must declare accepts_complex=True; "
            f"got {caps.accepts_complex}"
        )
