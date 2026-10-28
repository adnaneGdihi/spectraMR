"""Config for QualityMatchingStrategy -- HQ->LQ quality-matched degradation synthesis.

Read via ``config.training.quality_matching``.

``StrictSchema`` (``extra='forbid'``) on purpose. The parent
``TrainingStrategyConfigSchema`` is ``extra='allow'``, so a mistyped knob inside a
permissive block would be absorbed as a raw extra and silently do nothing -- the
mechanism by which ``min_center_fraction`` became inert (issue #550).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from spectramr.config.schemas.renames import reject_renamed_keys
from spectramr.config.schemas.strictness import StrictSchema

__all__ = ["QualityMatchingConfig", "QualityTargetConfig"]


class QualityTargetConfig(StrictSchema):
    """Where the target descriptor comes from, and which attributes are matched."""

    source: Literal["cohort", "literal"] = Field(
        default="cohort",
        description=(
            "'cohort' measures the target across a real low-quality cohort and takes "
            "a robust centre; 'literal' uses the override values verbatim."
        ),
    )
    cohort_manifest: str | None = Field(
        default=None,
        description=(
            "Manifest of the low-quality cohort the target is measured from. Required "
            "when source='cohort'. Must be produced by a committed generator under "
            "scripts/data/ -- a hand-built manifest never reaches the cluster."
        ),
    )
    attributes: list[str] = Field(
        min_length=1,
        description=(
            "Registered no-reference metric names defining the matched quality. Every "
            "entry must be registered and must not require a reference."
        ),
    )
    override: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-attribute literal target values. With source='cohort' these pin "
            "individual attributes so an ablation can vary one at a time; with "
            "source='literal' one is required per attribute."
        ),
    )
    pairing: Literal["paired", "unpaired"] = Field(
        default="unpaired",
        description=(
            "How the source and target cohorts relate.\n\n"
            "'unpaired' (default): different subjects at each quality. The target is "
            "the cohort MEDIAN of the matched attributes, and nothing per-subject can "
            "be checked. This is what a fastMRI/M4Raw pairing forces.\n\n"
            "'paired': the SAME subject appears at both field strengths (a "
            "travelling-volunteer cohort such as MRIxFields2026 Training_prospective). "
            "The target is still the cohort median, but the synthesised volume can "
            "additionally be compared against that subject's REAL low-field scan with "
            "full-reference metrics. That agreement is REPORTED, never optimised -- "
            "folding it into the objective would turn the only independent check of "
            "the chain into a training signal. Requires source_field and target_field."
        ),
    )
    pair_layout: Literal["field_keyed", "explicit"] = Field(
        default="field_keyed",
        description=(
            "How pairs are found in the manifest, when pairing='paired'.\n\n"
            "'field_keyed' (default): each record is ONE field-snapshot and pairs are "
            "formed by grouping subject+contrast across two field strengths (a "
            "travelling-volunteer cohort like MRIxFields2026). Needs source_field_t "
            "and target_field_t.\n\n"
            "'explicit': each record already carries BOTH sides, as a ULF->HF "
            "restoration manifest does. NOTE THE INVERSION: such a manifest names the "
            "ULF scan input_path/primary_path (the model's input) and the HF scan "
            "target_path. Degradation runs the other way, so the high-quality source "
            "is the manifest's TARGET and the low-quality target is its INPUT."
        ),
    )
    source_field_t: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Field strength (Tesla) of the HIGH-quality source, selected from a "
            "multi-field manifest. Required when pairing='paired'."
        ),
    )
    target_field_t: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Field strength (Tesla) of the LOW-quality target, selected from a "
            "multi-field manifest. Required when pairing='paired'."
        ),
    )
    contrast: str | None = Field(
        default=None,
        description=(
            "Restrict both cohorts to one contrast (t1w / t2w / flair). A cross-field "
            "match that mixed contrasts would attribute a CONTRAST difference to "
            "quality; TE and TR set contrast, not noise."
        ),
    )
    spacing_mm: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Target voxel spacing (slice, row, col) in mm, IMPOSED on the synthetic "
            "volume rather than fitted. When omitted with source='cohort' it is "
            "measured as the per-axis median across the cohort's ISMRMRD headers. "
            "Required when source='literal' and match_spacing is on -- there is no "
            "cohort to measure."
        ),
    )

    @field_validator("attributes")
    @classmethod
    def _attributes_are_measurable(cls, v: list[str]) -> list[str]:
        # Delegates to the runtime validator, so an unmeasurable target fails at
        # config load rather than being skipped mid-run (pitfall #15).
        from spectramr.infrastructure.physics.quality_descriptors import (
            validate_attributes,
        )

        validate_attributes(v)
        return v

    @model_validator(mode="after")
    def _check_source_requirements(self) -> QualityTargetConfig:
        if self.source == "cohort" and not self.cohort_manifest:
            raise ValueError(
                "target.source='cohort' requires target.cohort_manifest: the target "
                "must be measured from real data, not assumed."
            )
        if self.source == "literal":
            missing = [a for a in self.attributes if a not in self.override]
            if missing:
                raise ValueError(
                    "target.source='literal' requires an override value for every "
                    f"attribute; missing {missing}."
                )
        if self.pairing == "paired" and self.pair_layout == "field_keyed":
            missing = [n for n in ("source_field_t", "target_field_t") if getattr(self, n) is None]
            if missing:
                raise ValueError(
                    f"pairing='paired' requires {missing}: pairs are formed by "
                    "matching one subject across two field strengths, so both must "
                    "be named."
                )
            if (
                self.source_field_t is not None
                and self.target_field_t is not None
                and self.source_field_t <= self.target_field_t
            ):
                raise ValueError(
                    f"source_field_t ({self.source_field_t} T) must EXCEED "
                    f"target_field_t ({self.target_field_t} T). Degradation goes from "
                    "high field to low; the reverse would ask the chain to synthesise "
                    "signal that was never acquired."
                )
        stray = [k for k in self.override if k not in self.attributes]
        if stray:
            raise ValueError(
                f"override keys {stray} are not a declared attribute of this target "
                f"(declared: {self.attributes}). An override nobody reads is an "
                "unwired knob."
            )
        return self


class QualityMatchingConfig(StrictSchema):
    """Knobs for fitting a compounded degradation chain to a quality target."""

    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training.quality_matching"))  # type: ignore[arg-type]
    )

    axes: list[str] = Field(
        min_length=1,
        description=(
            "DEGRADATION_REGISTRY axes composing the chain, in application order. "
            "Native-only DigitalTwinSimulator axes are rejected -- they cannot be "
            "applied standalone."
        ),
    )
    target: QualityTargetConfig = Field(
        description="Where the matched quality target comes from.",
    )
    max_evals: int = Field(
        default=400,
        ge=10,
        description="Objective-evaluation budget for the derivative-free search.",
    )
    method: Literal["differential_evolution", "nelder_mead"] = Field(
        default="differential_evolution",
        description="scipy optimiser driving the fit.",
    )
    fit_seed: int = Field(
        default=0,
        description="Seed for both the optimiser and the degradation realisations.",
    )
    min_gap_closed: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of the initial descriptor gap the fit must close. Below "
            "this the fit RAISES rather than emitting an uncalibrated chain (the "
            "mechanism-fires guard, pitfall #16)."
        ),
    )
    match_spacing: bool = Field(
        default=True,
        description=(
            "Resample the high-quality volume onto the target voxel grid BEFORE "
            "fitting. This is the GEOMETRIC half of a quality match: resolution is a "
            "header fact, imposed rather than fitted, because fitting it against a "
            "sharpness proxy lets a blur term absorb a geometry error and still "
            "report a good residual. Requires an ISMRMRD header on the source volume "
            "and raises without one -- an assumed 1 mm default would put every "
            "synthetic volume on the wrong grid. Set false only when matching "
            "quality alone is genuinely intended, and say so in metadata.note."
        ),
    )
    enable_acquisition_prior: bool | None = Field(
        default=None,
        description=(
            "Derive the noise axis's warm start from the two cohorts' ISMRMRD "
            "acquisition parameters (field strength, averages, receiver bandwidth) "
            "rather than starting mid-box. This matters because the chain is usually "
            "underdetermined: several severity combinations reach the same measured "
            "quality, and the prior selects the physically plausible one. The "
            "voxel-volume term is deliberately excluded -- match_spacing already "
            "realises it, and counting it twice would hand the noise axis a target "
            "several dB too optimistic. Acquisition is read from an ISMRMRD header (h5) "
            "or from the JSON sidecar beside a NIfTI volume (MagneticFieldStrength, "
            "RepetitionTime and EchoTime in seconds). Requires target.source='cohort'; "
            "raises when the header or sidecar is absent rather than guessing a field "
            "strength. A NIfTI arm whose volumes carry no sidecar sets this to false.\n\n"
            "None (the default) means AUTO: on when target.source='cohort' (the only "
            "source that carries the low-quality field strength), off otherwise. The "
            "resolved value is read via `acquisition_prior_enabled` and stamped into "
            "the calibration artifact, so 'auto' is a documented rule rather than a "
            "silent fallback. Setting it True against a literal target RAISES."
        ),
    )
    fit_seeds: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of fit seeds (fit_seed, fit_seed+1, ...). The chain synthesised and "
            "reported is the lowest-residual seed. The spread of theta across seeds is "
            "reported as theta_mean and theta_sd in the calibration artifact: that spread "
            "is the estimand for the paired-versus-unpaired comparison, not the sampling "
            "error of one fit. Costs one full fit per seed."
        ),
    )
    intensity_percentile: float = Field(
        default=99.0,
        gt=0.0,
        le=100.0,
        description=(
            "Robust percentile of |volume| used as the shared intensity unit. The HQ "
            "fit volume, every cohort volume and both sides of the paired agreement are "
            "divided by their own value at this percentile before anything is measured, "
            "so attributes are compared in one convention. A literal target is therefore "
            "read in these normalised units (the percentile lands at 1)."
        ),
    )
    max_residual: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Absolute ceiling on the weighted residual, beside min_gap_closed. The weights "
            "are 1/target^2, so the residual is a relative error norm: 0.1 means the "
            "achieved attributes lie within about 10% of target. None disables the "
            "ceiling. A fit above it RAISES, because min_gap_closed alone passes any "
            "fit that halves a large gap."
        ),
    )
    held_out_fraction: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Fraction of PAIRS (= subjects) held out of the fit and used only to score "
            "agreement with the real low-field scan. Seeded and subject-level. 0 scores "
            "in-sample. Requires pairing: paired."
        ),
    )
    fit_volumes: int = Field(
        default=1,
        ge=1,
        description=(
            "How many high-quality volumes the fit runs on. They are taken in selection "
            "order, imposed on the target grid, and stacked along the slice axis, so the "
            "chain is fitted to all of them at once. Default 1 fits the first selected "
            "volume whole (not one loader slice). Raises when fewer volumes are selected."
        ),
    )
    synthesise: bool = Field(
        default=True,
        description=(
            "After fitting, apply the chain to every high-quality volume and write "
            "the degraded/clean pairs plus a v4 paired manifest. This is what turns "
            "a calibration into a usable dataset -- with it off the arm's only output "
            "is calibration.yaml and no downstream arm can consume it. Requires "
            "data.index_path; raises without one rather than writing nothing."
        ),
    )
    max_synth_volumes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cap on the number of volumes synthesised. None = the whole manifest. "
            "Set it for a smoke run; a silent cap would make a partial dataset look "
            "complete."
        ),
    )
    output_dir: str = Field(
        default="experiments/results/quality_matching",
        description="Where the calibration artifact and synthetic volumes are written.",
    )

    @field_validator("axes")
    @classmethod
    def _axes_are_applicable(cls, v: list[str]) -> list[str]:
        # Constructing the chain runs the REAL axis validation, so the schema and the
        # runtime can never disagree about which axes are legal.
        from spectramr.infrastructure.physics.degradation_chain import (
            ChainLink,
            DegradationChain,
        )

        DegradationChain(links=tuple(ChainLink(axis=a, theta=0.5) for a in v))
        return v

    @model_validator(mode="after")
    def _severities_are_identified(self) -> QualityMatchingConfig:
        # One attribute constrains one severity. More axes than attributes give a
        # zero-residual curve, not a point: the fit then reports whichever theta the seed
        # lands on (issue #2411). A motion axis therefore needs its own attribute, and
        # until one is declared rigid motion is held at zero by leaving it out.
        if len(self.axes) > len(self.target.attributes):
            raise ValueError(
                f"{len(self.axes)} axes {list(self.axes)} are fitted against "
                f"{len(self.target.attributes)} attribute(s) {self.target.attributes}: the "
                "severities are unidentified and the fit lands on a curve. Drop an axis "
                "(rigid_motion is held at zero by omitting it) or add an attribute that "
                "constrains it."
            )
        return self

    @model_validator(mode="after")
    def _held_out_needs_pairs(self) -> QualityMatchingConfig:
        if self.held_out_fraction > 0.0 and self.target.pairing != "paired":
            raise ValueError(
                "held_out_fraction > 0 requires target.pairing: paired. Only pairs carry "
                "a same-subject reference to score against; an unpaired cohort has none."
            )
        return self

    @model_validator(mode="after")
    def _acquisition_prior_is_resolvable(self) -> QualityMatchingConfig:
        # The prior needs the LOW-quality cohort's field strength, which only a
        # cohort source can supply. Catching it here beats failing after the fit.
        # Guard the EXPLICIT request, not the resolved default: a new validation
        # that rejects the library default would invalidate every existing
        # literal-target config for a knob its author never set.
        if self.enable_acquisition_prior is True and self.target.source != "cohort":
            raise ValueError(
                "enable_acquisition_prior=true requires target.source='cohort': the "
                "prior reads the low-quality cohort's field strength from its own "
                "headers, and a literal target has none. Leave it unset (auto) or "
                "set it false."
            )
        return self

    @property
    def acquisition_prior_enabled(self) -> bool:
        """Resolved value of the auto default. Read this, never the raw field."""
        if self.enable_acquisition_prior is None:
            return self.target.source == "cohort"
        return bool(self.enable_acquisition_prior)

    @model_validator(mode="after")
    def _spacing_is_resolvable(self) -> QualityMatchingConfig:
        # With a literal target there is no cohort whose headers could supply the
        # grid, so an explicit spacing is the only way match_spacing can mean
        # anything. Catching it here beats failing mid-run after the fit.
        if (
            self.match_spacing
            and self.target.source == "literal"
            and self.target.spacing_mm is None
        ):
            raise ValueError(
                "match_spacing is on with target.source='literal', but no "
                "target.spacing_mm was given and there is no cohort to measure one "
                "from. Declare target.spacing_mm, or set match_spacing: false."
            )
        return self
