r"""DL-BAE -- dispersion-latent Bloch autoencoder (M4).

An autoencoder whose **decoder is a physical law**. The encoder maps a multi-field
image stack :math:`\{x(B_0^{(m)})\}_{m=1}^M` to a *field-invariant* tissue latent

.. math::

   \xi = \bigl(\rho,\; a_0,\; c_0,\; \{b_k,\tau_{c,k}\}_{k=1}^P\bigr),

and the decoder evaluates the Bloembergen-Purcell-Pound dispersion law
(:mod:`spectramr.infrastructure.physics.dispersion`) at each field to obtain
:math:`(T_1(B_0), T_2(B_0))`, then renders the contrast.

Why the latent is field-invariant *by construction*: it parametrises the
dispersion **law**, not its value at one field. Nothing in :math:`\xi` names a
field, so a latent fit at 0.3 T predicts 3 T without retraining -- the property
that distinguishes DL-BAE from a field-conditioned encoder, which only
interpolates the fields it saw.

Identifiability is a hard constraint, not a hope: the stack gives one magnitude
per field, and the latent has :math:`2P+3` constants per voxel
(:func:`~spectramr.infrastructure.physics.dispersion.magnitude_stack_unknowns`),
so it needs :math:`M \ge 2P+3` distinct fields (#2417). Below that the fit is
rank-deficient and the recovered latent is meaningless, so
:meth:`DispersionBlochAutoencoder.__init__` raises rather than training to a
degenerate optimum (pitfall #9), and the Tier-1 ``dispersion_identifiability``
check catches it at audit time.

Scope (honest): the render below is a spoiled-gradient-echo/spin-echo magnitude
approximation, adequate for the multi-field data-consistency objective this model
is trained under. Sequence-exact rendering routes through the existing
``DifferentiableBlochLayer``; this module deliberately keeps the decoder analytic
so the dispersion gradient stays clean.

References
----------
N. Bloembergen, E. M. Purcell, R. V. Pound, "Relaxation effects in nuclear
magnetic resonance absorption," *Phys. Rev.* 73(7):679-712, 1948.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.dispersion import (
    dispersion_rates_voxelwise,
    magnitude_stack_unknowns,
)
from spectramr.models.registry import register_model


def _softplus_positive(x: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    """Map an unconstrained head to a strictly positive quantity."""
    return nn.functional.softplus(x) + floor


def _per_field_times(name: str, values: object, n_fields: int) -> tuple[float, ...]:
    """Validate a per-field TR or TE declaration: one positive time per field."""
    if values is None:
        raise ValueError(
            f"DL-BAE requires model_kwargs.{name}: one time (s) per entry of "
            "fields_present. The render depends on it, so there is no default (#1712)."
        )
    times = tuple(float(v) for v in values)  # type: ignore[attr-defined]
    if len(times) != n_fields or any(t <= 0.0 for t in times):
        raise ValueError(
            f"{name} must name one positive time (s) per field ({n_fields}); got {times!r}."
        )
    return times


def _as_channel(value: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A scalar or ``[M']`` time as a ``[1, M', 1, 1]`` tensor on ``like``'s device."""
    t = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return t.view(1, -1, 1, 1) if t.ndim == 1 else t


@register_model(
    name="disp_bloch_ae",
    training_mode="dispersion_bloch_ae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    supports_contrast_conditioning=False,
)
class DispersionBlochAutoencoder(nn.Module):
    r"""Encoder to a field-invariant BPP latent; decoder = dispersion + Bloch render.

    Args:
        fields_present: Distinct :math:`B_0` values (Tesla) in the input stack,
            in channel order. Its length is :math:`M`.
        n_pools: Number of BPP relaxation pools :math:`P`.
        hidden_channels: Encoder hidden width.
        depth: Number of encoder conv blocks.
        init_tau_c: Initial correlation time (s), used to bias the tau head.
        tau_c_bounds: Physiological ``(min, max)`` clamp for :math:`\tau_c` (s).
        repetition_times_s: TR (s) of each field's acquisition, in
            ``fields_present`` order. Required: the render depends on it.
        echo_times_s: TE (s) of each field's acquisition, in ``fields_present`` order.

    Raises:
        ValueError: when the arm is under-determined (:math:`M < 2P+3`), when
            ``fields_present`` is empty/non-positive/duplicated, when
            ``tau_c_bounds`` is not a positive ordered interval, or when the
            per-field TR/TE are missing, non-positive or not one per field.
    """

    def __init__(
        self,
        fields_present: tuple[float, ...] | list[float] = (0.05, 0.3, 1.5, 3.0, 7.0),
        n_pools: int = 1,
        hidden_channels: int = 32,
        depth: int = 3,
        init_tau_c: float = 1e-8,
        tau_c_bounds: tuple[float, float] = (1e-11, 1e-6),
        repetition_times_s: tuple[float, ...] | list[float] | None = None,
        echo_times_s: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__()
        fields = tuple(float(b) for b in fields_present)
        if not fields:
            raise ValueError("fields_present must name at least one field strength.")
        if any(b <= 0.0 for b in fields):
            raise ValueError(
                f"fields_present must be positive field strengths (T); got {fields!r}."
            )
        if len(set(fields)) != len(fields):
            raise ValueError(
                "fields_present must be DISTINCT: repeated fields add no rank to "
                f"the dispersion fit. Got {fields!r}."
            )
        n_pools = int(n_pools)
        if n_pools < 1:
            raise ValueError(f"n_pools must be >= 1; got {n_pools}.")
        required = magnitude_stack_unknowns(n_pools)
        if len(fields) < required:
            raise ValueError(
                f"DL-BAE is under-determined: a {n_pools}-pool latent has {required} "
                f"unknowns per voxel and one magnitude per field supplies {len(fields)} "
                f"({fields!r}); it needs M >= {required} distinct fields."
            )
        tr = _per_field_times("repetition_times_s", repetition_times_s, len(fields))
        te = _per_field_times("echo_times_s", echo_times_s, len(fields))
        low, high = (float(tau_c_bounds[0]), float(tau_c_bounds[1]))
        if not (0.0 < low < high):
            raise ValueError(f"tau_c_bounds must satisfy 0 < min < max; got {tau_c_bounds!r}.")

        self.n_pools = n_pools
        self.tau_c_min, self.tau_c_max = low, high
        self.register_buffer("fields", torch.tensor(fields, dtype=torch.float32))
        # [M] protocol per field, broadcast over [B, M, H, W] in decode.
        self.register_buffer("tr_s", torch.tensor(tr, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("te_s", torch.tensor(te, dtype=torch.float32).view(1, -1, 1, 1))

        # Encoder: multi-field stack -> shared trunk -> per-quantity heads.
        layers: list[nn.Module] = []
        c_in = len(fields)
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Conv2d(c_in, hidden_channels, 3, padding=1), nn.GELU()]
            c_in = hidden_channels
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        trunk_out = c_in

        self.head_rho = nn.Conv2d(trunk_out, 1, 3, padding=1)
        self.head_a0 = nn.Conv2d(trunk_out, 1, 3, padding=1)
        self.head_c0 = nn.Conv2d(trunk_out, 1, 3, padding=1)
        self.head_b = nn.Conv2d(trunk_out, n_pools, 3, padding=1)
        self.head_tau = nn.Conv2d(trunk_out, n_pools, 3, padding=1)

        # Bias the tau head so the initial tau_c lands near `init_tau_c` after
        # the sigmoid-in-log-space squash, rather than at an arbitrary corner of
        # the physiological range.
        init_tau = min(max(float(init_tau_c), low * 1.001), high * 0.999)
        frac = (torch.log(torch.tensor(init_tau)) - torch.log(torch.tensor(low))) / (
            torch.log(torch.tensor(high)) - torch.log(torch.tensor(low))
        )
        nn.init.zeros_(self.head_tau.weight)
        nn.init.constant_(self.head_tau.bias, float(torch.logit(frac.clamp(1e-4, 1 - 1e-4))))

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""Map the multi-field stack to the field-invariant latent :math:`\xi`.

        Args:
            x: Multi-field image stack ``[B, M, H, W]`` in ``fields_present`` order.

        Returns:
            Dict with ``rho``/``a0``/``c0`` ``[B, 1, H, W]`` and ``b``/``tau_c``
            ``[B, P, H, W]``. ``tau_c`` is log-uniformly squashed into
            ``tau_c_bounds``, so it is physiological by construction.

        Raises:
            ValueError: when the channel count does not match ``fields_present``.
        """
        expected = int(self.fields.numel())
        if x.ndim != 4 or x.shape[1] != expected:
            raise ValueError(
                f"DL-BAE expects a multi-field stack [B, {expected}, H, W] matching "
                f"fields_present; got {tuple(x.shape)}."
            )
        h = self.trunk(x)
        log_lo = torch.log(torch.tensor(self.tau_c_min, device=x.device, dtype=x.dtype))
        log_hi = torch.log(torch.tensor(self.tau_c_max, device=x.device, dtype=x.dtype))
        tau_c = torch.exp(log_lo + (log_hi - log_lo) * torch.sigmoid(self.head_tau(h)))
        return {
            "rho": _softplus_positive(self.head_rho(h)),
            "a0": _softplus_positive(self.head_a0(h)),
            "c0": _softplus_positive(self.head_c0(h)),
            "b": _softplus_positive(self.head_b(h)),
            "tau_c": tau_c,
        }

    def relaxation_maps(
        self, latent: dict[str, torch.Tensor], fields: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluate :math:`(T_1, T_2)` at ``fields`` from the latent.

        Args:
            latent: Output of :meth:`encode`.
            fields: Field strengths (T) ``[M']``; defaults to ``fields_present``.
                Passing an *unseen* field is the extrapolation the model exists
                for -- nothing in the latent is field-specific.

        Returns:
            ``(t1, t2)``, each ``[B, M', H, W]`` in seconds.
        """
        b0 = self.fields if fields is None else fields.to(self.fields.device)
        r1, r2 = dispersion_rates_voxelwise(
            b0.to(latent["a0"].dtype),
            a0=latent["a0"],
            c0=latent["c0"],
            b=latent["b"],
            tau_c=latent["tau_c"],
        )
        eps = 1e-8
        return 1.0 / (r1 + eps), 1.0 / (r2 + eps)

    def decode(
        self,
        latent: dict[str, torch.Tensor],
        fields: torch.Tensor | None = None,
        *,
        tr_s: float | torch.Tensor | None = None,
        te_s: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Render the multi-field stack from the latent through the dispersion law.

        Spoiled-gradient-echo magnitude:
        :math:`\rho\,(1-e^{-TR/T_1})\,e^{-TE/T_2}`.

        Args:
            latent: Output of :meth:`encode`.
            fields: Field strengths (T) to render at; defaults to ``fields_present``.
            tr_s: Repetition time (s), scalar or ``[M']``. Defaults to the per-field
                TRs declared at construction when ``fields`` is ``None``; required
                with explicit ``fields``, whose protocol the model cannot know.
            te_s: Echo time (s), as ``tr_s``.

        Returns:
            Rendered stack ``[B, M', H, W]``.

        Raises:
            ValueError: when ``fields`` is given without both ``tr_s`` and ``te_s``.
        """
        t1, t2 = self.relaxation_maps(latent, fields)
        if fields is None:
            tr = self.tr_s if tr_s is None else _as_channel(tr_s, t1)
            te = self.te_s if te_s is None else _as_channel(te_s, t1)
        elif tr_s is None or te_s is None:
            raise ValueError(
                "decode() at explicit fields needs tr_s and te_s for those fields; the "
                "declared per-field protocol covers fields_present only."
            )
        else:
            tr, te = _as_channel(tr_s, t1), _as_channel(te_s, t1)
        tr, te = tr.to(t1.dtype), te.to(t1.dtype)
        return latent["rho"] * (1.0 - torch.exp(-tr / t1)) * torch.exp(-te / t2)

    def forward(
        self,
        x: torch.Tensor,
        fields: torch.Tensor | None = None,
        *,
        tr_s: float | torch.Tensor | None = None,
        te_s: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode then decode.

        Args:
            x: Multi-field stack ``[B, M, H, W]``.
            fields: Optional render fields (defaults to ``fields_present``).
            tr_s: Repetition time (s); see :meth:`decode`.
            te_s: Echo time (s); see :meth:`decode`.

        Returns:
            ``(reconstruction, latent)``.
        """
        latent = self.encode(x)
        return self.decode(latent, fields, tr_s=tr_s, te_s=te_s), latent


__all__ = ["DispersionBlochAutoencoder"]
