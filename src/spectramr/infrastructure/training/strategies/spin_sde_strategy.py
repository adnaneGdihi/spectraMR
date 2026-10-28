"""Continuous-Time Spin SDE (ULF-PR-11 / II.8).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-11.

Stochastic Bloch evolution

    dM = (γ B × M − R · M + R · m_eq) dt + g(M) dW

integrated via Euler–Maruyama.  The strategy:

1. Reads (T1, T2, M0, B0) parameter maps from the parent
   :class:`OneShotMultiParameterStrategy` (or directly from the batch
   if a side path provides them).
2. Integrates the SDE forward with those parameters to produce a
   predicted magnetisation trajectory.
3. Computes Bloch-consistency MSE against observed contrasts and adds
   an evidence-lower-bound regulariser on the diffusion coefficient.

Public method :meth:`integrate` is exposed so downstream pipelines can
generate synthetic ULF acquisitions with fitted parameter maps.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE, OwnedParameter

from .multi_param_mapping_strategy import OneShotMultiParameterStrategy

logger = logging.getLogger(__name__)


class SpinSDEStrategy(OneShotMultiParameterStrategy):
    """Stochastic Bloch SDE strategy."""

    # parallel_compat: The diffusion parameter is an owned module, DDP-wrapped, read via call_owned.
    parallel_compat = frozenset({"replicated"})

    n_steps: int = 32
    dt: float = 1e-3
    diffusion: float = 0.05
    lambda_consistency: float = 1.0
    lambda_diffusion_prior: float = 1e-3

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("diffusion", OWNED_MODULE))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Resolve the SDE hyperparameters from the typed training.spin_sde block
        # (pitfall #15: read + validate + stamp); fall back to the class
        # defaults when absent. ``diffusion_init`` seeds the *learnable*
        # diffusion coefficient; n_steps/dt/lambdas are fixed.
        cfg = getattr(self.config.training, "spin_sde", None)
        # Optional explicit per-contrast echo schedule (seconds). When absent the
        # Bloch-consistency readout falls back to evenly-spaced steps; resolved
        # once and cached in _resolve_readout_steps (pitfall #15).
        self.echo_times: tuple[float, ...] | None = None
        self._readout_steps_cache: list[int] | None = None
        if cfg is not None:
            self.n_steps = int(cfg.n_steps)
            self.dt = float(cfg.dt)
            self.lambda_consistency = float(cfg.lambda_consistency)
            self.lambda_diffusion_prior = float(cfg.lambda_diffusion_prior)
            diffusion_init = float(cfg.diffusion_init)
            if cfg.echo_times:
                self.echo_times = tuple(float(t) for t in cfg.echo_times)
        else:
            diffusion_init = float(type(self).diffusion)
        # The diffusion coefficient is a *learnable* SDE parameter. As a plain
        # float its prior term ``self.diffusion**2`` was a constant leaf with
        # requires_grad=False, so the ELBO regulariser (and the SDE noise scale)
        # trained nothing. Promote it to an nn.Parameter on the training device
        # and register it on the generator optimizer so it is actually
        # optimised (the strategy is not an nn.Module, so we add the param
        # group explicitly — mirroring the VF/IB-VF lazy-builder pattern).
        self.diffusion = torch.nn.Parameter(torch.tensor(diffusion_init, device=self.device))
        self.register_owned_module("diffusion", OwnedParameter(self.diffusion))
        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            existing = {p for group in opt.param_groups for p in group["params"]}
            if self.diffusion not in existing:
                opt.add_param_group({"params": [self.diffusion]})

    def integrate(
        self,
        m0: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        B0: torch.Tensor | None = None,
        m_eq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler–Maruyama integration of the Bloch SDE.

        m0:  (B, 3, H, W) initial magnetisation.
        T1, T2: (B, 1, H, W) relaxation maps in seconds.
        B0:    (B, 1, H, W) optional off-resonance field in Hz.
        """
        m, T1, T2, omega, mz_eq = self._prep(m0, T1, T2, B0, m_eq)
        for _ in range(self.n_steps):
            m = self._euler_step(m, T1, T2, omega, mz_eq)
        return m

    def _prep(
        self,
        m0: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        B0: torch.Tensor | None,
        m_eq: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Clamp relaxation maps and build (omega, mz_eq) once per integration.

        Longitudinal recovery is toward the equilibrium magnetisation ``m_eq``
        (= M0 in the loss path), NOT a hardcoded 1.0 — the dz/dt term is
        ``(m_eq - mz)/T1``. Defaults to unit equilibrium for the standalone API.
        """
        T1 = T1.clamp_min(1e-3)
        T2 = T2.clamp_min(1e-3)
        omega = 2.0 * torch.pi * (B0 if B0 is not None else torch.zeros_like(T1))
        mz_eq = m_eq if m_eq is not None else torch.ones_like(T1)
        return m0, T1, T2, omega, mz_eq

    def _euler_step(
        self,
        m: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        omega: torch.Tensor,
        mz_eq: torch.Tensor,
        diffusion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One Euler-Maruyama step of the Bloch SDE; ``diffusion`` defaults to the parameter."""
        mx, my, mz = m[:, 0:1], m[:, 1:2], m[:, 2:3]
        dmx = (-mx / T2 - my * omega) * self.dt
        dmy = (-my / T2 + mx * omega) * self.dt
        dmz = ((mz_eq - mz) / T1) * self.dt
        drift = torch.cat([dmx, dmy, dmz], dim=1)
        # State-dependent diffusion: damp longitudinal more.
        g = (self.diffusion if diffusion is None else diffusion) * torch.tensor(
            [[1.0], [1.0], [0.5]], device=m.device, dtype=m.dtype
        ).view(1, 3, 1, 1)
        noise = g * torch.randn_like(m) * (self.dt**0.5)
        return m + drift + noise

    @staticmethod
    def _transverse(m: torch.Tensor) -> torch.Tensor:
        """Transverse-signal magnitude sqrt(Mx^2 + My^2) -> (B, 1, H, W)."""
        return (m[:, :2] ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()

    def integrate_multicontrast(
        self,
        m0: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        readout_steps: list[int],
        B0: torch.Tensor | None = None,
        m_eq: torch.Tensor | None = None,
        diffusion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-contrast transverse-magnitude readouts ``(B, C, H, W)``.

        ``readout_steps`` is a length-C list of 1-based integration-step indices
        in ``[1, n_steps]``. The SDE is integrated once; the transverse magnitude
        is captured at each requested step and assembled in the given contrast
        order (duplicate steps reuse the same readout tensor). This is the
        physically-faithful Bloch-consistency comparison — contrast ``c`` is the
        transverse signal at echo step ``readout_steps[c]`` — replacing the old
        single final-state readout that was broadcast against every contrast
        (audit 2026-06, design item D2).
        """
        m, T1, T2, omega, mz_eq = self._prep(m0, T1, T2, B0, m_eq)
        want = set(readout_steps)
        captured: dict[int, torch.Tensor] = {}
        for step in range(1, self.n_steps + 1):
            m = self._euler_step(m, T1, T2, omega, mz_eq, diffusion)
            if step in want:
                captured[step] = self._transverse(m)
        return torch.cat([captured[s] for s in readout_steps], dim=1)

    def _resolve_readout_steps(self, n_contrasts: int) -> list[int]:
        """Length-``n_contrasts`` list of 1-based steps to read out for the
        Bloch-consistency loss. Resolved once, then cached.

        Source priority (pitfall #15 — read + validate + stamp):

        1. ``training.spin_sde.echo_times`` (seconds) -> ``round(t / dt)`` clamped
           to ``[1, n_steps]``. Its length MUST equal the observed contrast count
           or a ``ValueError`` is raised (no silent truncation/broadcast).
        2. Single contrast -> the final step (exactly the legacy behaviour).
        3. Otherwise evenly spaced over the trajectory, ending at the final step.
        """
        cached = self._readout_steps_cache
        if cached is not None:
            if len(cached) != n_contrasts:
                raise ValueError(
                    f"spin_sde: cached readout schedule has {len(cached)} steps "
                    f"but the batch now reports {n_contrasts} contrasts — the "
                    "contrast count must be constant across the run."
                )
            return cached

        if self.echo_times is not None:
            if len(self.echo_times) != n_contrasts:
                raise ValueError(
                    f"spin_sde: training.spin_sde.echo_times has "
                    f"{len(self.echo_times)} entries but multi_contrast_signal "
                    f"carries {n_contrasts} contrasts — they must match (no "
                    "silent broadcast; audit 2026-06)."
                )
            steps = [max(1, min(self.n_steps, round(t / self.dt))) for t in self.echo_times]
            source = "echo_times"
        elif n_contrasts == 1:
            steps = [self.n_steps]
            source = "final_state(legacy)"
        else:
            span = self.n_steps - 1
            steps = [
                max(1, min(self.n_steps, round(1 + i * span / (n_contrasts - 1))))
                for i in range(n_contrasts)
            ]
            source = "evenly_spaced"

        self._readout_steps_cache = steps
        logger.info(
            "spin_sde: %d-contrast Bloch-consistency readout at steps=%s "
            "(source=%s, n_steps=%d, dt=%g)",
            n_contrasts,
            steps,
            source,
            self.n_steps,
            self.dt,
        )
        return steps

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # See TODO/audit/smoke_audit_20260516.md §F6.
        base = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if not isinstance(batch, dict):
            return base

        # Diffusion-coefficient prior, which favours low stochasticity. The prior and
        # every Euler step share one call_owned output, so DDP sees one forward per step.
        diffusion = self.call_owned("diffusion")
        prior = diffusion.pow(2) * self.lambda_diffusion_prior
        base["loss_sde_diffusion_prior"] = prior

        T1 = batch.get("T1_map")
        T2 = batch.get("T2_map")
        M0 = batch.get("M0_map")
        signals_obs = batch.get("multi_contrast_signal")
        if T1 is not None and T2 is not None and M0 is not None and signals_obs is not None:
            m0 = torch.cat([torch.zeros_like(T1), torch.zeros_like(T1), M0], dim=1)
            # Per-contrast Bloch consistency (design item D2): read out the
            # transverse magnetisation at ONE integration step per observed
            # contrast and match each to its own channel of signals_obs. The
            # previous code took a single final-state readout (B, 1, H, W) and
            # broadcast it against every contrast (B, C, H, W), so all C
            # contrasts were fit to the same prediction.
            n_contrasts = signals_obs.shape[1] if signals_obs.dim() >= 2 else 1
            readout_steps = self._resolve_readout_steps(n_contrasts)
            sig_pred = self.integrate_multicontrast(
                m0, T1, T2, readout_steps, B0=batch.get("B0_map"), m_eq=M0, diffusion=diffusion
            )
            consistency = self.lambda_consistency * (sig_pred - signals_obs).abs().mean()
            base["loss_bloch_consistency"] = consistency
            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base:
                    base[total_key] = base[total_key] + consistency + prior
                    break
            return base

        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + prior
                break
        return base
