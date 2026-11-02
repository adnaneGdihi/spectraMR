"""Which non-spatial axes each ``data.dataset_type`` exposes.

The workflow contract can only check "regime *X* needs a ``TEMPORAL`` axis but
this data has none" if it knows what axes a dataset carries. Datasets are
selected in configs by the ``data.dataset_type`` string, so these tables are
keyed off that string: the value that actually picks the data.

Keys are post-normalization values (the canonical-key contract)
==============================================================
``check_workflow_required_axes`` / ``check_workflow_spatial_rank`` read
``config.data.dataset_type`` off a **validated** ``DataConfigSchema``. That
field passes through :meth:`DataConfigSchema.validate_dataset_type`, which
lower-cases the raw YAML value, rewrites aliases to their canonical name, and
raises on anything left over. The only strings that can ever reach a lookup here
are therefore the canonical ones, which makes two kinds of key a dead rule:

* an **alias** key (``2d``, ``3d``, ``paired_nifti``, ``paired_mri``): the value
  is rewritten before the lookup, so the entry is never hit. ``dataset_type:
  2d`` arrives as ``image``.
* an **invalid** key (``slice``, ``volumetric``, ``reconstruction``): such a
  config raises at Tier 0 and never reaches the check at all.

Deleting an alias key costs no coverage, because its canonical target carries
the annotation. ``tests/unit/data/datasets/test_axis_exposure.py`` enforces this
by round-tripping every key of both tables through the real validator and
asserting it survives unchanged, so a dead key cannot be reintroduced silently.

Contract (mirrors the capability invariant): a ``dataset_type`` absent from a
table is **unannotated**, and the rule skips it rather than guessing. Only
annotate a ``dataset_type`` whose exposure is unambiguous. An absent key is
always safe; a wrong one is not.

Two routes, and the weaker one is the fallback
==============================================
Axes reach a rule by one of two routes, and they are not equal:

* **declared** (:func:`declared_axes_for`) -- a per-*arm* fact the config layer
  already validated: ``data.acquisition_axes`` (an explicit list, any
  dataset_type) unioned with ``data.bart.bart_dim_map`` (BART arms). This is the
  stronger claim and it wins.
* **annotated** (:func:`exposed_axes_for`) -- a per-*type* claim about a whole
  corpus, hand-written in ``DATASET_TYPE_AXES`` below, used when the arm
  declares nothing.

Some dataset types cannot be annotated at all, and ``bart_kspace`` is the worked
example: five of its arms declare an ``echo`` axis and three declare ``flip``, so
no single row is true for both. A per-type table cannot express a per-arm fact --
which is why those eight arms were invisible to
``check_workflow_required_axes`` until the declared route existed.

``data.acquisition_axes`` generalises that escape hatch beyond BART. The fact
``bart_dim_map`` encodes is not BART-specific, but its spelling was, so a
non-BART arm whose dataset_type has no truthful row had no way to state what it
acquires -- and simply annotating the row instead would have been the "wrong
key" this contract warns against, because the row generalises over a corpus
whose arms disagree.

Note what this does NOT do: a declaration says the arm's data carries an axis, it
does not make a loader preserve one. ``dataset_type: nifti`` still folds a 4-D
volume into channels and still exposes nothing, and declaring ``temporal`` on
such an arm would be a false claim rather than a fix -- the loader is the thing
to change, which is what ``fmri`` below is for.
"""

from __future__ import annotations

from typing import Any

from spectramr.config.schemas.enums import Axis

#: ``dataset_type`` -> the non-spatial axes it exposes. Absent key = unannotated
#: (skip). Ambiguous / multi-purpose types (``synthetic``,
#: ``preprocessed``, ``pde_synthetic``) are deliberately omitted so the rule
#: never emits a false positive on data whose axes it cannot vouch for.
DATASET_TYPE_AXES: dict[str, frozenset[Axis]] = {
    # Structural single-frame data: no non-spatial encoding axis. ``image``
    # absorbs the 2d / image_folder / folder aliases, ``npy_slice`` the
    # paired_slices / slice_paired / npy_paired ones, and ``nifti_paired`` the
    # paired_nifti / paired_mri ones.
    "image": frozenset(),
    "npy_slice": frozenset(),
    "nifti_paired": frozenset(),
    "contrast_aware_paired": frozenset(),
    # ``NiftiStrategy.load`` folds a 4-D NIfTI's trailing axis into the CHANNEL
    # slot ((W,H,D,C) -> (C,W,H,D)) with no TR, frame order, or echo semantics,
    # so no non-spatial axis survives this route. A 4-D BOLD series read as
    # ``nifti`` is mangled into channels; the temporal route is
    # ``FMRIVolumeDataset``, which no dataset_type selects. Absorbs 3d /
    # 3d_volumetric.
    "nifti": frozenset(),
    # fastMRI k-space (fastmri_kspace / fastmri_knee / fastmri_brain / volume_h5
    # all normalize here). ``FastMRIH5Strategy`` reads exactly two layouts,
    # single-coil ``(Slices, H, W)`` and multi-coil ``(Slices, Coils, H, W)``:
    # static, single-contrast, non-spectral either way. COIL is deliberately NOT
    # claimed, because the single-coil route has no coil axis and both corpora
    # share this one dataset_type. frozenset() is the conservative side: no
    # regime requires COIL today, so withholding it cannot reject a real arm,
    # whereas claiming it would vouch for an axis half the routes lack.
    "kspace": frozenset(),
    # M4Raw 0.3T is 4-coil raw k-space, so COIL is genuine (``coil_processing_mode``
    # may collapse it downstream, but that knob is invisible to a dataset_type-keyed
    # table; the claim here is about the corpus). The NEX repetitions are NOT a
    # temporal axis: they are phase-incoherent re-acquisitions of a static object,
    # differing only by thermal noise and global phase drift, which ``_average_reps``
    # collapses into the target. Annotating TEMPORAL would let a dynamic / functional
    # / perfusion arm pass on static brain data, and temporal_fidelity would then
    # grade the correct answer (the rep mean, which IS the target) as a failure.
    # There is no echo axis; the T1/T2/FLAIR split is Task.CONTRAST_TRANSLATION,
    # not an Axis.
    "m4raw": frozenset({Axis.COIL}),
    # ``CineMRIDataset`` loads 4-D ``[H, W, slices, frames]`` and preserves frame
    # ordering as ``subject["frame_order"]``, so the frame axis is real and usable.
    # CARDIAC_PHASE is not claimed: this is a generic 4-D cine reader (ACDC and
    # FastMRI cine layouts) and it does not certify that frames are cardiac phases.
    "cine": frozenset({Axis.TEMPORAL}),
    # MRIxFields2026 is magnitude-only, MNI-space, paired translation across field
    # strengths. Field strength is not an Axis member and contrast is
    # Task.CONTRAST_TRANSLATION, so no non-spatial axis is exposed; magnitude-only
    # also rules out COIL.
    "mrixfields": frozenset(),
    # ``FMRIBoldSeriesDataset`` is the ONLY loader that keeps a time axis
    # legible: it refuses a 3-D volume outright and carries ``frame_order``,
    # ``num_frames`` and ``tr`` on the Subject beside the tensor. TorchIO's data
    # model is ``[C, H, W, D]``, so the frames sit in the channel slot here just
    # as they would through ``nifti`` -- the layout is NOT the difference. The
    # difference is that ``NiftiStrategy.load`` drops the ordering, the count and
    # the TR, leaving something indistinguishable from coils or contrasts, which
    # is exactly why the ``nifti`` row above claims nothing. This row meets the
    # same standard the ``cine`` row does, and for the same stated reason.
    #
    # Deliberately NOT in ``DATASET_TYPE_RANKS``: the served spatial rank depends
    # on the data (a 2-D+t series and a 3-D+t series both route here), which is
    # the same knob-invisible-to-the-table argument that omits ``nifti`` and
    # ``mrixfields``. ``mri_functional`` declares ``spatial_ranks={3}``, so an
    # honest skip is better than pinning a rank this table cannot see.
    "fmri": frozenset({Axis.TEMPORAL}),
    "flow_synthetic": frozenset({Axis.VELOCITY_ENCODING, Axis.TEMPORAL, Axis.COIL}),
}


#: BART dimension role -> the :class:`Axis` member it evidences. This is the ONE
#: place the two axis vocabularies meet: ``_BART_DIM_ROLES``
#: (``config/schemas/data.py`` -- 11 *acquisition-layout* roles a BART array may
#: carry) and ``Axis`` (8 *non-spatial axes a regime may require*). They are
#: deliberately not merged: the first describes how a file is laid out, the second
#: describes what an experiment needs, and collapsing them would make one of the
#: two lie. An adapter is the honest seam.
#:
#: Every role is either mapped here or listed in ``_BART_ROLES_WITHOUT_AXIS`` with
#: its reason, and ``tests/unit/data/datasets/test_axis_exposure.py`` asserts the
#: two together cover ``_BART_DIM_ROLES`` *exactly*. Without that test a role
#: added to the BART vocabulary later would map to nothing in silence -- the
#: adapter-bypass failure this repo has already hit once, when a checker agreed
#: with a domain adapter by luck rather than by routing through it.
_BART_ROLE_TO_AXIS: dict[str, Axis] = {
    "coil": Axis.COIL,
    "echo": Axis.ECHO,
    # Flip-angle index. Was in ``_BART_ROLES_WITHOUT_AXIS`` with the reason "no
    # regime requires it" -- true when written, and circular once a regime
    # wanted to: ``flip`` had no Axis because no regime required it, and no
    # regime could require it because it had no Axis. Three B1+ arms
    # (Bloch-Siegert, AFI, double-angle) sat in the gap, unambiguously
    # quantitative and undeclarable (#1020).
    "flip": Axis.FLIP_ANGLE,
    # cine / real-time frame. TEMPORAL only, never CARDIAC_PHASE -- the same call
    # the ``cine`` row above makes, for the same reason: a frame index does not
    # certify that the frames are cardiac phases.
    "frame": Axis.TEMPORAL,
}

#: The remaining roles, each with the reason it has no ``Axis`` member. There are
#: two distinct reasons and the distinction is load-bearing, so they are kept
#: apart rather than lumped into one "unsupported" bucket.
_BART_ROLES_WITHOUT_AXIS: dict[str, str] = {
    # (1) SPATIAL encoding directions. ``Axis`` is by construction the *non*-spatial
    #     vocabulary; spatial extent is ``DATASET_TYPE_RANKS``' subject, not this
    #     table's. These are not missing members, they are out of scope.
    "readout": "spatial: frequency-encode direction",
    "phase": "spatial: Cartesian phase-encode direction",
    "phase2": "spatial: second phase-encode direction (3-D)",
    "spoke": "spatial: non-Cartesian readout index",
    "slice": "spatial: 2-D slice index",
    # (2) NON-spatial, but no regime states a rule about them. Adding an ``Axis``
    #     member would create a field whose only possible reader is a rule that
    #     cannot fire -- the exact argument that deleted ``optional_axes`` from
    #     ``WorkflowProfile`` (profiles.py, 2026-07-31). If a regime ever requires
    #     one, the member and the rule arrive together.
    "map": "reconstruction basis (ESPIRiT map set), not an axis of the acquisition",
    # ``repetition`` is NEX / signal averages, and mapping it to TEMPORAL is
    # precisely the error the ``m4raw`` row above refuses: phase-incoherent
    # re-acquisitions of a STATIC object are not a time series. Claiming TEMPORAL
    # would let a dynamic / functional / perfusion arm pass on static data.
    "repetition": "signal averages (NEX) of a static object; NOT Axis.TEMPORAL",
}


def declared_axes_for(data_cfg: Any) -> frozenset[Axis] | None:
    """The axes an arm *declares* it acquired, or ``None`` if it declares none.

    The companion to :func:`exposed_axes_for`, and the stronger of the two.
    ``exposed_axes_for`` reads an annotation about a ``dataset_type`` -- a claim
    about a whole corpus, written by hand in the table above. This reads a
    *per-arm declaration the config layer already validated*:
    ``data.bart.bart_dim_map``, which :class:`~spectramr.config.schemas.data.
    BartConfigSchema` requires to be non-empty when enabled, rejects unknown roles
    in, and refuses to let a non-singleton dimension go unnamed (pitfall #9).

    That difference is why ``bart_kspace`` has no row in ``DATASET_TYPE_AXES`` and
    cannot be given one: its axes are not a property of the dataset_type at all.
    The eight ``bart_kspace`` arms in this repo disagree with each other -- five
    declare ``echo`` (a dual-echo B0 acquisition), three declare ``flip``
    (double-angle B1) -- so any single static annotation would be wrong for one
    group. A per-type table cannot express a per-arm fact, which is what left
    those arms invisible to ``check_workflow_required_axes``: the one rule whose
    whole job is to consume exactly the fact they were declaring.

    Returns ``None`` -- "no declaration, fall back to the type table" -- rather
    than an empty set when the arm declares nothing, preserving the non-breaking
    skip invariant the rest of the axis system uses. An *enabled* dim map that
    happens to name only spatial roles does return ``frozenset()``, and that is
    deliberate: it is a positive declaration that the arm carries no non-spatial
    axis, which should reject a regime requiring one rather than skip.

    Limitation, stated rather than hidden: a role's *presence* is declared, its
    *extent* is not. ``bart_dim_map: {echo: 6}`` says dim 6 carries echoes; it
    cannot say how many. A rule needing "at least two echoes" is a data-level
    check and belongs where the tensor is (see
    ``MultiEchoB0FitStrategy._as_complex_echoes``), not in a config check.
    """
    declared: set[Axis] = set()
    saw_declaration = False

    # (1) The explicit, dataset-agnostic declaration. ``bart_dim_map`` could only
    #     ever serve BART arms, yet the fact it encodes -- "this acquisition
    #     carries these non-spatial axes" -- is not BART-specific. An arm whose
    #     dataset_type has no truthful table row (because the row would have to
    #     generalise over a corpus that disagrees) can now state it directly.
    #     Validated against the Axis enum at the schema, so a typo raises there
    #     rather than resolving to an empty set here.
    axes_field = getattr(data_cfg, "acquisition_axes", None)
    if axes_field is not None:
        saw_declaration = True
        declared |= {Axis(name) for name in axes_field}

    # (2) The BART dim map, unchanged. Both are positive per-arm statements about
    #     the same acquisition, so they UNION rather than override: a BART arm
    #     that also declares an axis its dim map cannot express should keep both.
    bart = getattr(data_cfg, "bart", None)
    if bart is not None and getattr(bart, "enabled", False):
        dim_map = getattr(bart, "bart_dim_map", None) or {}
        if dim_map:
            saw_declaration = True
            declared |= {_BART_ROLE_TO_AXIS[role] for role in dim_map if role in _BART_ROLE_TO_AXIS}

    if not saw_declaration:
        return None
    return frozenset(declared)


def exposed_axes_for(dataset_type: str | None) -> frozenset[Axis] | None:
    """Return the axes a ``dataset_type`` exposes, or ``None`` if unannotated.

    ``dataset_type`` must be a **post-validation** (canonical) value; an alias
    such as ``"2d"`` is unannotated here by construction. ``None`` (unknown /
    ambiguous / unannotated) means "skip the check", the same non-breaking
    invariant the capability system uses.
    """
    if dataset_type is None:
        return None
    return DATASET_TYPE_AXES.get(dataset_type)


def resolve_axes_for(data_cfg: Any) -> frozenset[Axis] | None:
    """The axes an arm carries: declared if it declares any, else annotated.

    The two-route composition, in one place. It was inlined in
    ``check_workflow_required_axes`` while it had a single consumer; the batch
    contract (``TrainingBatch.axes``) is the second, and two hand-written copies
    of "declared wins" is how the audit and the tensor start disagreeing about
    the same arm -- the divergent-sibling shape this module's own docstring
    describes for ``bart_kspace``.

    ``None`` means "cannot vouch for the axes" (unannotated ``dataset_type``,
    no per-arm declaration) and every consumer must SKIP on it rather than treat
    it as "no axes". That is not the same as ``frozenset()``, which is the
    positive claim that the arm carries no non-spatial axis and should REJECT a
    regime requiring one. Preserving that distinction is the whole reason this
    returns an optional set rather than a tuple.
    """
    declared = declared_axes_for(data_cfg)
    if declared is not None:
        return declared
    return exposed_axes_for(getattr(data_cfg, "dataset_type", None))


#: ``dataset_type`` -> the spatial rank it provides. Absent key = unannotated
#: (skip). Only unambiguously-ranked types are listed. ``image`` is omitted
#: because a folder of images can be 2-D slices or 3-D volumes; ``nifti`` is
#: omitted for the same reason, since ``IndexBuilder.build_nifti_index`` emits
#: per-slice records (rank 2) when ``datasets[0].variant == "2d_slices"`` and
#: whole volumes (rank 3) otherwise, a knob this table cannot see; ``kspace`` is
#: omitted because it is a catch-all over several corpora.
DATASET_TYPE_RANKS: dict[str, int] = {
    "npy_slice": 2,
    # M4Raw is a 2-D multi-slice acquisition: every slice is independently 2-D
    # encoded and ``_rss_combine`` runs a per-slice 2-D ``ifft2c`` over (H, W).
    # The leading slice axis is an index, not a 3-D encoding direction.
    "m4raw": 2,
    # The phantom is a 3D Cartesian acquisition served as readout slices of
    # hybrid (x, ky, kz) space; the encoding, and the assembled volume, is 3D.
    "flow_synthetic": 3,
    # ``mrixfields`` is deliberately OMITTED, for the same reason ``nifti`` is: the
    # served rank is decided by a knob this table cannot see. ``MRIxFieldsPairedDataset``
    # takes a ``mrixfields_slice_mode`` — ``central`` / ``all_slices`` serve a 2-D slice,
    # ``volume`` serves the whole ``[C, H, W, D]`` (rank 3). Since the underlying data is
    # a 3-D volume and the mode picks the rank, no single value is honest here; the
    # ``mri_structural`` regime already declares ``spatial_ranks={2, 3}``, so an
    # unannotated mrixfields arm SKIPS the rank check rather than being pinned to one rank.
}


def spatial_rank_for(dataset_type: str | None) -> int | None:
    """Return the spatial rank a ``dataset_type`` provides, or ``None`` if unannotated.

    ``dataset_type`` must be a **post-validation** (canonical) value; an alias
    such as ``"3d"`` is unannotated here by construction.
    """
    if dataset_type is None:
        return None
    return DATASET_TYPE_RANKS.get(dataset_type)


#: ``dataset_type`` -> the signal domain(s) its loader actually materialises. Absent
#: key = unannotated (skip). Values mirror ``spectramr.models.capabilities.Domain`` (bare
#: strings, kept import-free like the tables above).
#:
#: This is the **per-dataset** companion to a regime's ``signal_domains``. A regime
#: declares what its signal *can* be (``mri_structural`` spans
#: ``{image, kspace, complex_image}`` — magnitude arms, complex arms, and raw-k-space
#: recon all live under it). A *dataset* is narrower: mrixfields is magnitude-only MNI
#: images, so it materialises ``{image}`` and nothing else. ``check_workflow_signal_domain``
#: checks a model against the permissive regime set; ``check_workflow_dataset_signal_domain``
#: checks it against THIS set, catching a k-space model pointed at magnitude-only data —
#: a mismatch the regime-level check cannot see because ``kspace`` is a legal structural
#: domain in the abstract.
#:
#: Only annotate a ``dataset_type`` whose loader output domain is unambiguous. Ambiguous /
#: multi-purpose types (``synthetic``, ``preprocessed``, ``pde_synthetic``,
#: ``oracle_bssfp``, ``quantitative``, ``field_ref``) are omitted so the rule never
#: vouches for a domain it cannot guarantee. An absent key is always safe; a wrong one is not.
DATASET_TYPE_SIGNAL_DOMAINS: dict[str, frozenset[str]] = {
    # Magnitude / real-image loaders: they materialise the ``image`` key, never a
    # Fourier or complex representation.
    "image": frozenset({"image"}),
    "nifti": frozenset({"image"}),
    "nifti_paired": frozenset({"image"}),
    "contrast_aware_paired": frozenset({"image"}),
    "dicom": frozenset({"image"}),
    # MRIxFields2026 is magnitude-only [0, 1] MNI images across field strengths — image
    # domain ONLY. NOT ``complex_image`` and NOT ``kspace``: a model declaring an fft /
    # k-space input_domain against this data reads a representation the loader never
    # produces, and there is no honest k-space of a magnitude image.
    "mrixfields": frozenset({"image"}),
    # Raw multi-coil k-space acquisitions: the loader materialises the complex ``kspace``
    # key (real/imag interleaved), which is also a ``complex_image`` after an ifft2c.
    "m4raw": frozenset({"kspace", "complex_image"}),
    "kspace": frozenset({"kspace", "complex_image"}),
    "bart_kspace": frozenset({"kspace", "complex_image"}),
    "ismrmrd_kspace": frozenset({"kspace", "complex_image"}),
    # ``npy_slice`` and ``cine`` are deliberately OMITTED. ``npy_slice`` is ambiguous:
    # 7 of its 8 arms carry it as complex data (``coil_processing_mode: rss`` ->
    # 2 real+imag channels), while another is an 8-channel multi-contrast stack — no
    # single domain is honest, and annotating ``{image}`` would also disagree with
    # ``_IMAGE_DOMAIN_DATASET_TYPES`` (where npy_slice is correctly absent, so those rss
    # arms get 2 channels). ``cine`` (``CineMRIDataset``) is magnitude image frames and
    # would be ``{image}``, but it has zero arms and its channel classification is
    # untested, so it stays unannotated (absent = skip) rather than asserted.
}


#: Coil-processing modes that make the dataset serve IMAGE data whatever its
#: ``dataset_type`` says. ``rss_image`` / ``magnitude`` apply the IFFT inside the
#: dataset's own TorchIO transform pipeline, so a ``dataset_type: kspace`` arm in
#: one of these modes hands the model an image and the k-space row above is not
#: what it materialises.
#:
#: This exact oversight already bit once: ``needs_ifft_for_visualization``
#: ignored the mode until 2026-05-15 and IFFT'd already-image tensors, producing
#: the spectral/tiled-noise aliasing in ten smoke fakes. The constant lives here,
#: in the data layer that owns what a dataset produces, and
#: ``training/utils/domain_inference.py`` imports it rather than keeping a second
#: copy -- two hand-maintained lists of "which modes emit images" is how the
#: audit and the runtime start disagreeing about the same arm.
IMAGE_DOMAIN_COIL_MODES: frozenset[str] = frozenset({"rss_image", "magnitude"})


def _coil_processing_mode(data_cfg: Any) -> str:
    """The arm's coil-processing mode, read from its canonical home.

    There is only one spelling left to read. ``data.coil_processing_mode`` is a
    FOLD record (-> ``data.coils.processing_mode``), so the flat attribute does
    not exist on a loaded config and the fallback that used to sit here returned
    ``None`` for every arm. It read canonical-first, so it was dead weight rather
    than a defect -- but it is the shape
    ``test_renames.py::TestNoStringKeyedReadsOfFoldedNames`` exists to catch,
    because the identical shape reading legacy-FIRST is how a rename goes
    silently wrong (see the `hq_manifest` and `_resolve_probe_timestep` repairs
    in #1016).
    """
    coils = getattr(data_cfg, "coils", None)
    mode = getattr(coils, "processing_mode", None) if coils is not None else None
    return str(mode or "").strip().lower()


def resolve_signal_domains_for(data_cfg: Any) -> frozenset[str] | None:
    """The domains an arm's data actually materialises.

    The per-*arm* companion to :func:`signal_domains_for`, and the same
    declared-beats-annotated shape as :func:`resolve_axes_for`. A ``dataset_type``
    row states what a corpus generally produces; ``coil_processing_mode``
    overrides it for THIS arm by moving the IFFT inside the dataset's transform
    pipeline.

    Without this, ``check_workflow_dataset_signal_domain`` reported 16 arms
    across 10 model families as "the model cannot read what this dataset
    produces" (#1010) -- every one of which sets ``rss_image`` and is served
    images exactly as its model expects. The arms were right and the check was
    reading a row that no longer described them.
    """
    if _coil_processing_mode(data_cfg) in IMAGE_DOMAIN_COIL_MODES:
        return frozenset({"image"})
    return signal_domains_for(getattr(data_cfg, "dataset_type", None))


def signal_domains_for(dataset_type: str | None) -> frozenset[str] | None:
    """Return the signal domain(s) a ``dataset_type`` materialises, or ``None`` if unannotated.

    ``dataset_type`` must be a **post-validation** (canonical) value. ``None`` (unknown /
    ambiguous / unannotated) means "skip the check", the same non-breaking invariant the
    axis and rank tables use.
    """
    if dataset_type is None:
        return None
    return DATASET_TYPE_SIGNAL_DOMAINS.get(dataset_type)


__all__ = [
    "DATASET_TYPE_AXES",
    "DATASET_TYPE_RANKS",
    "DATASET_TYPE_SIGNAL_DOMAINS",
    "IMAGE_DOMAIN_COIL_MODES",
    "declared_axes_for",
    "exposed_axes_for",
    "resolve_signal_domains_for",
    "signal_domains_for",
    "spatial_rank_for",
]
