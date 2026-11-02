"""Knob applicability — which config blocks are meaningful for which regime.

Some config blocks only make sense for a specific imaging regime: an ``mrf:``
trajectory block is inert under ``mri_structural``; a ``data.quantitative``
target-map block is meaningless unless you are mapping parameters. This is the
inverse of scattering that knowledge across each block's validator — one SSOT
table, consumed by ``check_knob_applicability``.

Scope is **regime-gated** here. Modality gating ("coil arrays are meaningless
for ultrasound") is deliberately *not* modelled yet: every runnable regime in
this framework is MR, and the non-MR regimes are ``STUB`` — an arm declaring
one is already rejected by ``check_workflow_declared`` before any knob is
reached, so a modality gate could never fire. It will be added alongside the
first non-MR regime, so the table never carries a rule that cannot fire.
"""

from __future__ import annotations

from dataclasses import dataclass

from spectramr.config.schemas.enums import Regime


@dataclass(frozen=True)
class KnobScope:
    """One config block and the regimes for which it is meaningful.

    Attributes:
        path: Dotted path to the block on ``TrainingSettings`` (e.g. ``"mrf"``,
            ``"data.quantitative"``).
        regimes: Regimes for which the block is applicable. Enabling it under
            any other regime is an INAPPLICABLE error.
        enabled_key: Sub-attribute that gates enablement (e.g. ``"enabled"``).
            ``None`` means the block counts as enabled whenever it is present
            (not ``None``).
    """

    path: str
    regimes: frozenset[Regime]
    enabled_key: str | None = None


KNOB_APPLICABILITY: tuple[KnobScope, ...] = (
    # The mrf: trajectory/dictionary block is fingerprinting-only. Declared
    # under any other regime it is an inert knob that silently does nothing.
    KnobScope("mrf", regimes=frozenset({Regime.FINGERPRINTING})),
    # Per-batch quantitative target maps (T1/T2/T2*/PD/ADC) only make sense
    # when the arm is actually mapping parameters.
    KnobScope(
        "data.quantitative",
        regimes=frozenset({Regime.QUANTITATIVE, Regime.FINGERPRINTING}),
        enabled_key="enabled",
    ),
    # Velocity encoding (venc, 4-point scheme, flux masks) is meaningless
    # outside phase-contrast flow: under any other regime the dataset emits no
    # velocity and the block would silently do nothing.
    KnobScope(
        "data.phase_contrast",
        regimes=frozenset({Regime.FLOW}),
        enabled_key="enabled",
    ),
    # The synthetic phase-contrast phantom emits velocity-encoded k-space only.
    KnobScope(
        "data.flow_synthetic",
        regimes=frozenset({Regime.FLOW}),
        enabled_key="enabled",
    ),
    # The DCE time axis / AIF source only mean something when the arm is
    # actually fitting tracer kinetics.
    KnobScope(
        "data.perfusion",
        regimes=frozenset({Regime.PERFUSION}),
        enabled_key="enabled",
    ),
    # The FID length / dwell time / resonance count only mean something when the
    # arm is actually fitting a spectrum.
    KnobScope(
        "data.spectroscopy",
        regimes=frozenset({Regime.SPECTROSCOPIC}),
        enabled_key="enabled",
    ),
)


__all__ = ["KNOB_APPLICABILITY", "KnobScope"]
