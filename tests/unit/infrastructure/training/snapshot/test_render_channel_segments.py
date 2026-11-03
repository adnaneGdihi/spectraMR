"""Mixed-domain ``model_input`` renders (experiment_11 ``model_input`` crosshair).

The cold-diffusion strategy builds the tensor the network is actually fed by
concatenating two DIFFERENT domains on the channel axis
(``strategies/diffusion.py``, ``torch.cat([noisy_images, smaps], dim=1)``):

* channels ``0:2C``  -- the degraded k-space ``x_t``, real-stacked
  ``(R0,I0,R1,I1,...)`` and ``log1p``-compressed by the arm's
  ``data.processing.enable_log_scaling``;
* channels ``2C:4C`` -- the coil sensitivity maps, real-stacked in the SAME
  layout but **image domain** and never compressed.

Both halves are even-channel real-stacked float32, so nothing about the tensor
distinguishes them. ``strategies/base.py`` lists ``model_input`` in its
``_canonical`` authoritative-k-space set, which by construction bypasses the
spectrum veto and applies ``expm1`` + ``ifft2c`` to all four coil-map channels
as though they were a spectrum.

The IFFT of a smooth, DC-dominated image-domain field is a separable sinc-like
kernel: a bright centre pixel plus a horizontal and a vertical ridge. After the
RSS combine that cross carries the overwhelming majority of the frame's energy,
and ``render_image_preview``'s closing per-sample min-max then divides the whole
picture by that spike -- so the genuine zero-filled brain in the other half is
crushed to near-black. On ``experiment_11_attention_none`` this rendered
``model_input.png`` as a black frame with a white dot and a crosshair, and read
as "the model input is worse than a zero-filled image" when the k-space actually
handed to the network was fine. The defect is in the RENDER, not the data.

Note what is NOT the fix: narrowing ``log_scaled_keys`` (forbidden outright by
``utils/kspace_view.py``) would leave the wrong ``ifft2c`` in place, and slicing
the recorded tensor down to its k-space half would break the debug-snapshot
contract's requirement that the artifact show the REAL model input
(``docs/debug_snapshot_contract.rst``, CLAUDE.md non-negotiable 14). The tensor
is genuinely 16-channel; only its rendering has to become domain-aware.

**Update (#1327).** The maps half is no longer image-domain: it now reaches the
concat through ``prepare_smaps_for_kspace_conditioning``, which ``fft2c``s it,
RMS-matches it to the k-space half and amplitude-caps it. So it DOES need the
``ifft2c`` now -- but it still never passed through the arm's ``log1p``, and
``expm1`` on an uncompressed spectrum clamps every bin above
``DECOMPRESS_MAGNITUDE_CEILING`` to one value: the magnitude is flattened, phase
survives, and the render is a washed-out ringing artifact. Deriving "was it
compressed" from "is it k-space" therefore stopped working, which is why a
segment carries an optional fourth field. The crosshair story below is the
ORIGIN of the mechanism, not its current shape.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.snapshot.render import (
    render_image_preview,
    split_channel_segments,
)

_SIZE = 64
_COILS = 4


def _real_stack(z: torch.Tensor) -> torch.Tensor:
    """Complex ``(B,C,H,W)`` -> ``(B,2C,H,W)`` interleaved ``R0,I0,R1,I1,...``.

    Mirrors the ``view_as_real -> permute -> reshape`` chain the diffusion
    strategy applies to complex smaps before the concat, so the layout under
    test is the one the run actually produced.
    """
    return (
        torch.view_as_real(z)
        .permute(0, 1, 4, 2, 3)
        .reshape(z.shape[0], -1, *z.shape[2:])
        .contiguous()
    )


def _phantom_and_maps() -> tuple[torch.Tensor, torch.Tensor]:
    """A disc phantom and RSS-normalised complex coil maps, both image-domain."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, _SIZE), torch.linspace(-1, 1, _SIZE), indexing="ij"
    )
    brain = ((xx**2 / 0.6**2 + yy**2 / 0.8**2) < 1.0).float()

    centres = [(-0.7, 0.0), (0.7, 0.0), (0.0, -0.7), (0.0, 0.7)]
    maps = torch.stack(
        [
            torch.complex(
                torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 1.2),
                0.15 * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 1.2),
            )
            for cx, cy in centres
        ]
    ).unsqueeze(0)
    # `_estimate_smaps_from_acs` RSS-normalises the maps; keep that, it is what
    # pins the image-domain half at O(1) while compressed k-space sits near 0.
    maps = maps / torch.sqrt((maps.abs() ** 2).sum(1, keepdim=True) + 1e-8)
    return brain, maps


def _compressed_kspace_half(brain: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    """The ``x_t`` half: coil k-space, ``log1p``-compressed as the arm asks."""
    ksp = fft2c(brain.unsqueeze(0).unsqueeze(0) * maps)
    ksp = torch.polar(torch.log1p(ksp.abs()), torch.angle(ksp))  # enable_log_scaling
    return _real_stack(ksp)


def _mixed_model_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(model_input_16ch, kspace_half_8ch)`` as the strategy did PRE-#1327.

    The maps half is image-domain here. Kept as-is because it is the fixture the
    crosshair regression was found on, and because it is the shape that must
    keep working through the three-tuple segment spelling.
    """
    brain, maps = _phantom_and_maps()
    kspace_half = _compressed_kspace_half(brain, maps)
    smaps_half = _real_stack(maps)
    return torch.cat([kspace_half, smaps_half], dim=1), kspace_half


def _mixed_model_input_post_1327() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(model_input_16ch, kspace_half, smaps_k_half)`` as the strategy builds it NOW.

    ``prepare_smaps_for_kspace_conditioning`` ``fft2c``s the maps before the
    concat, so BOTH halves are k-space -- but only the first went through
    ``log1p``. The maps reach the concat straight from the transform, which is
    the whole reason a segment has to be able to say "k-space, uncompressed".
    """
    brain, maps = _phantom_and_maps()
    kspace_half = _compressed_kspace_half(brain, maps)
    smaps_k_half = _real_stack(fft2c(maps))  # NO log1p -- see the docstring
    return torch.cat([kspace_half, smaps_k_half], dim=1), kspace_half, smaps_k_half


def _crosshair_fraction(img: torch.Tensor) -> float:
    """Share of the frame's energy sitting on the centre row + centre column.

    The signature of an IFFT applied to a smooth image-domain field. A genuine
    brain render puts a percent or two here; the artifact puts most of the frame.
    """
    e = img[:, 0] ** 2
    h, w = e.shape[-2:]
    cross = e[:, h // 2, :].sum(-1) + e[:, :, w // 2].sum(-1) - e[:, h // 2, w // 2]
    return float((cross / e.sum((-2, -1))).mean())


def test_ifft_of_image_domain_smaps_half_produces_the_crosshair() -> None:
    """Characterise the defect: the render is unusable, and it is the smaps half.

    Guards the premise of the fix rather than the fix itself. If this ever stops
    holding, the mixed-domain render stopped being harmful and the special
    handling below can go -- but nothing may quietly assume that.
    """
    mixed, kspace_only = _mixed_model_input()

    good = render_image_preview(kspace_only, in_kspace=True, authoritative=True, log_scaled=True)
    bad = render_image_preview(mixed, in_kspace=True, authoritative=True, log_scaled=True)
    assert good is not None and bad is not None

    # The k-space-only render is a brain: energy spread over the anatomy.
    assert _crosshair_fraction(good) < 0.10
    # Folding the image-domain maps in moves the frame onto the centre cross...
    assert _crosshair_fraction(bad) > 0.50
    # ...and the per-sample min-max then crushes the real content to near-black.
    assert float(bad.mean()) < 0.1 * float(good.mean())


def test_channel_segments_render_each_half_in_its_own_domain() -> None:
    """The fix: declaring the split recovers a readable brain for the k-space half.

    The k-space segment keeps the IFFT *and* the ``expm1`` it genuinely needs;
    the image-domain segment takes neither. Note this is emphatically not
    "narrowing ``log_scaled_keys``" -- that would suppress a decompression the
    k-space half still requires, which ``utils/kspace_view`` forbids for good
    reason. Here the maps are declared image-domain, so ``expm1`` was never the
    correct operation for them at all.

    This is the THREE-tuple spelling, i.e. every declaration written before
    #1327 forced domain and compression apart. It must keep meaning exactly what
    it meant: the segment inherits the parent tensor's compression answer, which
    the helper reports as ``None``.
    """
    mixed, kspace_only = _mixed_model_input()
    segments = [("kspace", kspace_only.shape[1], True), ("smaps", 2 * _COILS, False)]

    parts = split_channel_segments("model_input", mixed, segments)
    assert parts is not None
    assert [n for n, _, _, _ in parts] == ["model_input__kspace", "model_input__smaps"]
    assert [log for _, _, _, log in parts] == [None, None], (
        "a 3-tuple segment must declare NOTHING about compression, so the "
        "render loop falls back to the domain answer as it always did"
    )

    _ks_name, ks_tensor, ks_is_kspace, _ks_log = parts[0]
    assert ks_is_kspace
    torch.testing.assert_close(ks_tensor, kspace_only)

    rendered = render_image_preview(ks_tensor, in_kspace=True, authoritative=True, log_scaled=True)
    reference = render_image_preview(
        kspace_only, in_kspace=True, authoritative=True, log_scaled=True
    )
    assert rendered is not None and reference is not None
    torch.testing.assert_close(rendered, reference)
    # The whole point: the k-space half now renders as a brain, not a crosshair.
    assert _crosshair_fraction(rendered) < 0.10

    # The maps render as magnitude, untransformed — smooth, no central spike.
    sm_render = render_image_preview(
        parts[1][1], in_kspace=False, authoritative=False, log_scaled=False
    )
    assert sm_render is not None
    assert _crosshair_fraction(sm_render) < 0.10


def test_mismatched_segment_widths_are_refused_not_approximated() -> None:
    """A wrong declaration must be loud, never silently re-normalised.

    A split that does not add up is a bug at the emitting call site. Rendering
    an approximation of it would put a confident ``__kspace``/``__smaps``
    filename on a picture built from the wrong channels — precisely the class of
    silent fallback CLAUDE.md non-negotiable 3 rules out.
    """
    mixed, _ = _mixed_model_input()
    bad = [("kspace", 4, True), ("smaps", 4, False)]  # 8 declared, 16 present
    assert split_channel_segments("model_input", mixed, bad) is None


def _image_masked_kspace() -> torch.Tensor:
    """Real-stacked k-space carrying an IMAGE-domain line mask.

    This is precisely what ``_image_domain_mask_fraction`` exists to catch: a
    Cartesian line mask multiplied against IMAGE data and then FFT'd, so the
    tensor is k-space by declaration while the undersampling happened in the
    wrong domain. Rendering it (``ifft2c``) brings the zeroed rows back, which a
    genuine k-space undersampling never does -- that aliases into a dense image.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, _SIZE), torch.linspace(-1, 1, _SIZE), indexing="ij"
    )
    brain = ((xx**2 / 0.6**2 + yy**2 / 0.8**2) < 1.0).float()
    brain = brain * (1.0 + 0.5 * torch.cos(4 * xx) * torch.cos(4 * yy))
    img = brain.unsqueeze(0).unsqueeze(0).to(torch.complex64)
    img[..., ::2, :] = 0.0  # the mask, applied in the IMAGE domain
    return _real_stack(fft2c(img))


def _render_calls(monkeypatch) -> list[tuple[torch.Tensor, bool, bool]]:
    """Record ``(tensor, in_kspace, log_scaled)`` for every preview render.

    The unit under test is the render-unit DERIVATION -- which flags each
    segment earns -- not the pixels it produces. Spying on the call is the only
    way to see a wrong ``expm1`` directly; comparing PNGs would show it only as
    "the picture looks off", which is exactly the reading that took three passes
    to diagnose the first time.
    """
    import spectramr.infrastructure.training.debug_snapshot as ds

    seen: list[tuple[torch.Tensor, bool, bool]] = []
    real = ds.render_image_preview

    def _spy(t, *, in_kspace, authoritative=False, log_scaled=False):
        seen.append((t, in_kspace, log_scaled))
        return real(t, in_kspace=in_kspace, authoritative=authoritative, log_scaled=log_scaled)

    import spectramr.infrastructure.training.snapshot_coil_maps as scm

    real_grid = scm.render_coil_map_grid

    # Coil-map segments render through the per-coil grid, with the same flags.
    def _grid_spy(t, *, in_kspace=False, log_scaled=False):
        seen.append((t, in_kspace, log_scaled))
        return real_grid(t, in_kspace=in_kspace, log_scaled=log_scaled)

    monkeypatch.setattr(ds, "render_image_preview", _spy)
    monkeypatch.setattr(scm, "render_coil_map_grid", _grid_spy)
    return seen


def test_the_writer_renders_each_declared_segment_in_its_own_domain(tmp_path) -> None:
    """The cold-diffusion ``model_input`` split reaches the PNGs: one per segment, and
    the never-compressed maps half is IFFT'd without ``expm1``."""
    import json

    from spectramr.core import snapshot_trace as st
    from spectramr.core.snapshot_trace import View
    from spectramr.infrastructure.training.snapshot.writer import MANIFEST, TraceWriter, _units

    model_input, _ = _mixed_model_input()
    half = model_input.shape[1] // 2
    view = View(
        kspace=True,
        authoritative=True,
        log_scaled=True,
        segments=(("x_t", half, True), ("smaps", half, True, False)),
    )
    units = {name: v for name, _, v in _units("model_input", model_input, view)}
    assert units["model_input__x_t"].log_scaled and not units["model_input__smaps"].log_scaled
    assert units["model_input__smaps"].kspace

    trace = st.SnapshotTrace(phase="train", step=0)
    with st.active(trace):
        st.record(
            "strategy", "model_input", {"model_input": model_input}, views={"model_input": view}
        )
    out = TraceWriter(tmp_path).write(trace, {})
    pngs = json.loads((out / MANIFEST).read_text())["stages"][0]["tensors"][0]["png"]
    assert [p.rsplit("__", 1)[-1] for p in pngs] == ["x_t.png", "smaps.png"]


def test_a_three_tuple_segment_inherits_compression_from_its_domain() -> None:
    from spectramr.core.snapshot_trace import View
    from spectramr.infrastructure.training.snapshot.writer import _units

    t = torch.zeros(1, 4, 8, 8)
    view = View(kspace=True, log_scaled=True, segments=(("k", 2, True), ("img", 2, False)))
    got = {name: v for name, _, v in _units("x", t, view)}
    assert got["x__k"].log_scaled and not got["x__img"].log_scaled and not got["x__img"].kspace
