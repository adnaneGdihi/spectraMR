"""Frozen ``WorkflowProfile`` facts — one per imaging :class:`Regime`.

A ``WorkflowProfile`` is a *declaration of truth* about a regime: what modality
it belongs to, how far the framework can honestly run it
(:class:`~spectramr.config.schemas.enums.Maturity`), what spatial ranks and
non-spatial axes it needs, and which physics forward operator (a key into the
:class:`~spectramr.infrastructure.physics.registry.OperatorRegistry`) reconstructs
it. These are *pure facts* — this module imports nothing from ``infrastructure``
so it stays layer-legal (``domain`` never depends leftward). The registry-walk
that asserts these facts against the live registries lives in
:mod:`spectramr.infrastructure.validation.workflow_ledger`, and the anti-facade
test that ties the two together is
``tests/unit/domain/workflows/test_maturity_ledger.py``.

The maturity contract:

- ``LIVE``      — nothing the regime needs is missing: a registered forward model
  (a linear operator **or** a signal model), a regime-tagged strategy, and
  regime-tagged losses **and** metrics. Trains, and can be graded on its own terms.
- ``PARTIAL``   — runs, but coverage is thin: a tagged strategy, **or** both
  tagged losses and tagged metrics. Audit reports the gap.
- ``EVAL_ONLY`` — metrics exist, zero losses/strategies; ``train`` raises.
- ``STUB``      — nothing exists; every pipeline raises ``WorkflowNotImplementedError``.

A regime declares its forward physics in exactly one of two fields, by whether an
adjoint exists: ``forward_operator`` for a linear measurement operator (``fft2d``,
``phase_contrast``), ``signal_model`` for a nonlinear ``parameters -> signal`` map
(``extended_tofts``, ``adc_monoexp``). A regime may declare both — DWI is
FFT-reconstructed *and* obeys the mono-exponential decay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spectramr.config.schemas.enums import Axis, Maturity, Modality, Regime, Task


@dataclass(frozen=True, slots=True)
class WorkflowProfile:
    """The frozen facts backing one imaging regime.

    ``forward_operator`` and ``signal_model`` are both *keys* into a registry
    (``OperatorRegistry`` and ``SignalModelRegistry` respectively), never
    free-form labels — a profile naming physics the registry does not know is an
    anti-facade failure the maturity-ledger test catches. Both ``None`` means "no
    forward model wired" (STUB, or a regime only evaluated post-hoc).

    The two fields are not interchangeable, and which one a regime uses is a fact
    about the physics rather than a style choice: an ``OperatorRegistry`` entry
    *promises* ``adjoint()``, and generic callers (``OperatorProjectionDataConsistency``,
    ``FISTAMBIRSolver``, ``NullSpaceProjection``) consume that promise assuming
    ``<Ax, y> = <x, A'y>``. A nonlinear map has no adjoint, so it goes in
    ``signal_model`` — registering it as an operator with a fabricated adjoint
    would not crash, it would silently produce wrong results in every solver that
    never thought to ask whether the operator was linear.

    One of the nine fields is declared and read by NOTHING. It is recorded here
    rather than left to look load-bearing, because a field that no consumer reads
    is pitfall #15 wearing a dataclass:

    - ``modality`` — every profile sets it; no consumer reads it. The deferral is
      argued in :mod:`spectramr.domain.workflows.knobs`: every runnable regime is
      MR and the non-MR regimes are ``STUB``, which ``check_workflow_declared``
      rejects before a modality gate could fire, so the gate would be a rule that
      cannot fire. The STUB rows are the seam the first non-MR regime lands on,
      and the gate arrives with it.

    ``optional_axes`` was the second such field and is **gone** (2026-07-31). It
    was inert twice over — no consumer read it *and* no profile populated it —
    which is why deleting it beat wiring it: ``required_axes`` alone drives
    ``check_workflow_required_axes``, and an optional axis has no rule to state,
    since by construction its absence cannot make an arm inadmissible. A field
    whose only possible reader would be a rule that cannot fire is decoration.

    ``tests/unit/domain/workflows/test_maturity_ledger.py`` pins the unread set.
    It is pinned because the claim "the last declared-but-unchecked field" has
    already been made once (of ``signal_domains``, 2026-07-16) while two fields
    were sitting unread — the pin makes the next such claim fail a test rather
    than land in a commit message.
    """

    name: Regime
    modality: Modality
    maturity: Maturity
    spatial_ranks: frozenset[int]
    #: Non-spatial axes the regime needs, read as **at least one of** -- which is
    #: what ``check_workflow_required_axes`` has always PRINTED ("expose none of
    #: them") though it long computed ``required - exposed``, an all-of test. The
    #: two agreed while every profile declared 0 or 1 axis, which was true of all
    #: 14 until ``mri_quantitative`` needed "echo OR flip_angle" (#1020). The
    #: check now matches its own message.
    #:
    #: A future regime needing two axes TOGETHER cannot be expressed here and
    #: should not be faked by adding both: it needs its own field, added with the
    #: rule that reads it.
    required_axes: frozenset[Axis] = field(default_factory=frozenset)
    signal_domains: frozenset[str] = field(default_factory=frozenset)
    #: Key into the physics OperatorRegistry — LINEAR physics, has an adjoint.
    forward_operator: str | None = None
    #: Key into the SignalModelRegistry — NONLINEAR physics, has no adjoint.
    signal_model: str | None = None
    supported_tasks: frozenset[Task] = field(default_factory=frozenset)


# Signal-domain literals mirror spectramr.models.capabilities.Domain. Kept as
# bare strings here so this module has zero cross-layer imports.
_MR_IMAGE_KSPACE = frozenset({"image", "kspace", "complex_image"})

_RECON_FAMILY = frozenset(
    {
        Task.RECONSTRUCTION,
        Task.DENOISING,
        Task.SUPER_RESOLUTION,
    }
)


WORKFLOW_PROFILES: dict[Regime, WorkflowProfile] = {
    Regime.STRUCTURAL: WorkflowProfile(
        name=Regime.STRUCTURAL,
        modality=Modality.MR,
        maturity=Maturity.LIVE,
        spatial_ranks=frozenset({2, 3}),
        signal_domains=_MR_IMAGE_KSPACE,
        forward_operator="fft2d",
        supported_tasks=_RECON_FAMILY
        | frozenset(
            {
                Task.SYNTHESIS,
                Task.CONTRAST_TRANSLATION,
                Task.MOTION_CORRECTION,
                # PILOTStrategy (learned k-space trajectory under slew/gradient
                # constraints) and BALDAcquisitionStrategy (closed-loop line
                # selection) are both structural recon strategies that ALSO
                # design the acquisition, and `acquisition:` is a wired v6.1
                # block. ACQUISITION_DESIGN read as an unsupported Task only
                # because both strategies inherited ReconstructionTrainingStrategy's
                # capabilities instead of declaring their own.
                Task.ACQUISITION_DESIGN,
            }
        ),
    ),
    Regime.QUANTITATIVE: WorkflowProfile(
        name=Regime.QUANTITATIVE,
        modality=Modality.MR,
        # LIVE: MultiEchoB0FitStrategy / OneShotMultiParameterStrategy (Bloch-
        # anchored T1/T2/PD), Bloch-synthesis + Bottomley-prior losses, and
        # b0_field_rmse / geodesic_qmap_error / cross_scanner_t1t2_concordance
        # to grade the maps in physical units.
        maturity=Maturity.LIVE,
        spatial_ranks=frozenset({2, 3}),
        # ANY of these, not all (see WorkflowProfile.required_axes): the regime
        # asks that the acquisition vary a physical parameter along SOME
        # encoding axis, and which one depends on the parameter. Echo time
        # varies for B0 / T2*; flip angle varies for B1+ (double-angle, AFI,
        # Bloch-Siegert). Requiring ECHO alone made three B1+ arms in the ``vf``
        # cohort undeclarable while being unambiguously quantitative (#1020).
        required_axes=frozenset({Axis.ECHO, Axis.FLIP_ANGLE}),
        signal_domains=_MR_IMAGE_KSPACE,
        forward_operator="fft2d",
        supported_tasks=_RECON_FAMILY | frozenset({Task.PARAMETER_MAPPING, Task.FIELD_TRANSLATION}),
    ),
    Regime.FINGERPRINTING: WorkflowProfile(
        name=Regime.FINGERPRINTING,
        modality=Modality.MR,
        # LIVE: ConformalMRFDictlessReconStrategy, the dictionary-matching
        # fingerprint loss, and geodesic_mrf_parameter_error on the Bloch-MRF
        # manifold. Several other MRF-NAMED components are deliberately NOT
        # tagged — tropical_mrf_consistency's reference fan is randn(), and
        # SpatiotemporalMRFReconStrategy's "4D Beltrami" is x.diff(dim=-1). See
        # docs/reference/workflow_backlog.md for the full list and why.
        maturity=Maturity.LIVE,
        spatial_ranks=frozenset({2, 3}),
        required_axes=frozenset({Axis.TRANSIENT}),
        signal_domains=_MR_IMAGE_KSPACE,
        forward_operator="fft2d",
        supported_tasks=frozenset({Task.PARAMETER_MAPPING, Task.RECONSTRUCTION}),
    ),
    Regime.FUNCTIONAL: WorkflowProfile(
        name=Regime.FUNCTIONAL,
        modality=Modality.MR,
        # LIVE, and read `supported_tasks` before reading that as more than it is:
        # this regime supports the RECON family ONLY. The claim is "the framework
        # can reconstruct/denoise a functional series" — BeltramiEPIDistortion
        # (real EPI readout physics), the EPI susceptibility-distortion loss, and
        # tsnr + temporal_fidelity to grade it. It is NOT a claim that BOLD
        # modelling exists: there is no GLM task here, no BOLD task member, and
        # the tagged strategy models the READOUT, not neural activity. LIVE is
        # scoped by supported_tasks, which is why that field is a fact and not
        # decoration.
        maturity=Maturity.LIVE,
        spatial_ranks=frozenset({3}),
        required_axes=frozenset({Axis.TEMPORAL}),
        signal_domains=_MR_IMAGE_KSPACE,
        forward_operator="fft2d",
        supported_tasks=_RECON_FAMILY,
    ),
    Regime.DIFFUSION_WEIGHTED: WorkflowProfile(
        name=Regime.DIFFUSION_WEIGHTED,
        modality=Modality.MR,
        # LIVE, and the one regime that honestly declares BOTH kinds of forward
        # model: DWI is FFT-reconstructed (a linear operator with an adjoint) AND
        # obeys the mono-exponential decay S(b) = S0 exp(-b*ADC) (a nonlinear
        # signal model with none). What carries the claim is adc_monoexp, the
        # dwi_adc_monoexp loss — which consumes the `b_values` key LoadDWIMetadata
        # really does write — and adc_mae, which grades the ADC map in physical
        # units. That is physics in the objective and a metric on its own terms.
        #
        # QSpaceDiffusionStrategy is DWI-tagged, so it satisfies strategies_tagged.
        # Its SH / Laplace-Beltrami term WAS a facade until #350 (2026-07-18): it
        # gated on batch["bvecs"], a key nothing produced, so get() returned None
        # and the strategy silently degraded to vanilla reconstruction under a
        # comment claiming no silent fallback (pitfall #16). Fixed: LoadDWIMetadata
        # now attaches b_vectors as a tensor, the strategy reads that exact key and
        # RAISES if a diffusion batch lacks it. The mechanism now fires — but the
        # maturity claim still does not REST on it: DWI is LIVE on adc_monoexp, the
        # dwi_adc_monoexp loss (consumes b_values) and adc_mae (grades ADC in
        # physical units). The q-space regulariser is honest added coverage, not the
        # load-bearing evidence. DTI/IVIM tensor models remain backlogged.
        maturity=Maturity.LIVE,
        spatial_ranks=frozenset({2, 3}),
        required_axes=frozenset({Axis.DIFFUSION_ENCODING}),
        signal_domains=_MR_IMAGE_KSPACE,
        forward_operator="fft2d",
        signal_model="adc_monoexp",
        supported_tasks=_RECON_FAMILY | frozenset({Task.PARAMETER_MAPPING}),
    ),
    Regime.DYNAMIC: WorkflowProfile(
        name=Regime.DYNAMIC,
        modality=Modality.MR,
        # LIVE: LowRankSparseStrategy (RPCA X = L + S), the SToRM manifold
        # regulariser, and temporal_fidelity — which is the metric that actually
        # grades this regime, because per-frame PSNR cannot see a reconstruction
        # collapsing to the temporal mean.
        maturity=Maturity.LIVE,
        spatial_ranks=frozenset({2, 3}),
        required_axes=frozenset({Axis.TEMPORAL}),
        signal_domains=_MR_IMAGE_KSPACE,
        forward_operator="fft2d",
        supported_tasks=_RECON_FAMILY,
    ),
    Regime.PERFUSION: WorkflowProfile(
        name=Regime.PERFUSION,
        modality=Modality.MR,
        # LIVE via `signal_model`, which is what this regime needed all along.
        # `forward_operator` stays None and that is the CORRECT answer, not a
        # gap: the tracer-kinetic map is nonlinear and has no adjoint, so there
        # is no honest OperatorRegistry entry for it, and a fake `adjoint` would
        # be silently consumed by generic callers assuming <Ax,y> = <x,A'y>
        # (pitfall #16). Perfusion sat PARTIAL for months purely because the LIVE
        # rule asked the wrong question — it demanded a linear operator, so the
        # only way to satisfy it was the facade the ledger exists to prevent.
        #
        # Everything else was already here: real tracer-kinetic physics
        # (extended Tofts / gamma-variate / Parker AIF), PERFUSION-tagged losses
        # and metrics, and a tagged, trainable PerfusionKineticMappingStrategy,
        # which resolves this exact key through the SignalModelRegistry.
        maturity=Maturity.LIVE,
        spatial_ranks=frozenset({2, 3}),
        required_axes=frozenset({Axis.TEMPORAL}),
        signal_domains=_MR_IMAGE_KSPACE,
        forward_operator=None,
        signal_model="extended_tofts",
        supported_tasks=frozenset({Task.PARAMETER_MAPPING}),
    ),
    Regime.SPECTROSCOPIC: WorkflowProfile(
        name=Regime.SPECTROSCOPIC,
        modality=Modality.MR,
        # LIVE via `signal_model`, like perfusion and for the same reason: the
        # Lorentzian sum is nonlinear in frequency, linewidth and phase, and has
        # no adjoint, so `forward_operator=None` is the correct answer rather
        # than a gap. This regime was the last EVAL_ONLY one — it had metrics and
        # nothing else. It now has the AMARES/QUEST time-domain fit
        # (mrs_lorentzian), the self-supervised MRSFIDResidualLoss, and a tagged
        # MRSQuantificationStrategy that resolves this key through the registry.
        #
        # `signal_domains` names "spectrum", NOT the image/kspace family: an MRS
        # acquisition is a time signal read as a spectrum, and it is neither an
        # image nor 2-D k-space. Declaring _MR_IMAGE_KSPACE here would let a
        # model registered for image reconstruction be pointed at an FID.
        maturity=Maturity.LIVE,
        spatial_ranks=frozenset({2, 3}),
        required_axes=frozenset({Axis.SPECTRAL}),
        signal_domains=frozenset({"spectrum"}),
        forward_operator=None,
        signal_model="mrs_lorentzian",
        supported_tasks=frozenset({Task.PARAMETER_MAPPING}),
    ),
    Regime.FLOW: WorkflowProfile(
        name=Regime.FLOW,
        modality=Modality.MR,
        # LIVE: both clauses of the ledger's rule are satisfied honestly — the
        # `phase_contrast` operator is registered (and pc_adjoint is a TRUE
        # inverse within |v| < venc, not a stand-in), and PhaseContrastFlowStrategy
        # is registered in STRATEGY_CLASS_PATHS and FLOW-tagged. Real physics
        # (flow_encoding), FLOW-tagged losses and metrics back it.
        #
        # LIVE is a fact about the CODE, not about any one cluster's data: if
        # maturity tracked local data, this profile would be PARTIAL here and LIVE
        # at a 4D-flow site, which is incoherent for a frozen fact. Data
        # availability is a separate gate, and both halves of what this comment
        # used to say about that gate were wrong.
        #
        # 4D-flow data is NOT absent. The cluster holds osu_4dflow_cine and
        # whole_heart_5d, extracted (TODO/cluster_data_inventory_2026_06_16.md,
        # which retired the "M4Raw-0.3T-magnitude-only" premise the workflow
        # backlog was written under). What is missing is wiring: no dataset_type
        # exposes VELOCITY_ENCODING, so there is no flow arm to run.
        #
        # And that missing annotation does not gate anything, because the guard's
        # polarity is the opposite of the intuitive reading.
        # check_workflow_required_axes rejects an arm only when the dataset_type IS
        # annotated in data/datasets/axis_exposure.py and its declared axes lack
        # VELOCITY_ENCODING. An UNANNOTATED dataset_type returns None from
        # exposed_axes_for, and the check SKIPS it — passed=True, severity=info.
        # An absent annotation is never a rejection, so "no dataset_type exposes
        # it" protects this regime from nothing; only a PRESENT annotation that
        # omits the axis does.
        maturity=Maturity.LIVE,
        spatial_ranks=frozenset({2, 3}),
        required_axes=frozenset({Axis.VELOCITY_ENCODING}),
        signal_domains=_MR_IMAGE_KSPACE,
        forward_operator="phase_contrast",
        supported_tasks=frozenset({Task.PARAMETER_MAPPING, Task.RECONSTRUCTION}),
    ),
    # --- STUBs: typed seams the framework does not implement yet. ---
    Regime.NMR_SPECTROSCOPY: WorkflowProfile(
        name=Regime.NMR_SPECTROSCOPY,
        modality=Modality.MR,
        maturity=Maturity.STUB,
        spatial_ranks=frozenset(),
        required_axes=frozenset({Axis.SPECTRAL}),
        forward_operator=None,
    ),
    Regime.CT: WorkflowProfile(
        name=Regime.CT,
        modality=Modality.XRAY_TRANSMISSION,
        maturity=Maturity.STUB,
        spatial_ranks=frozenset({2, 3}),
        forward_operator=None,
    ),
    Regime.XRAY: WorkflowProfile(
        name=Regime.XRAY,
        modality=Modality.XRAY_TRANSMISSION,
        maturity=Maturity.STUB,
        spatial_ranks=frozenset({2}),
        forward_operator=None,
    ),
    Regime.ULTRASOUND: WorkflowProfile(
        name=Regime.ULTRASOUND,
        modality=Modality.ULTRASOUND,
        maturity=Maturity.STUB,
        spatial_ranks=frozenset({2, 3}),
        forward_operator=None,
    ),
    Regime.OPTICAL: WorkflowProfile(
        name=Regime.OPTICAL,
        modality=Modality.OPTICAL,
        maturity=Maturity.STUB,
        spatial_ranks=frozenset({2, 3}),
        forward_operator=None,
    ),
}


def get_profile(regime: Regime) -> WorkflowProfile:
    """Return the frozen profile for ``regime``.

    Raises:
        KeyError: if the regime has no profile — impossible while the
            maturity-ledger test (``set(WORKFLOW_PROFILES) == set(Regime)``)
            passes, but explicit so a future enum addition fails loudly.
    """
    return WORKFLOW_PROFILES[regime]


__all__ = ["WORKFLOW_PROFILES", "WorkflowProfile", "get_profile"]
