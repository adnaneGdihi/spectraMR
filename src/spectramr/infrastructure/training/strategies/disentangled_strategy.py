"""Disentangled Training Strategy
================================

Training strategy for Disentangled MRI models (Experiment 32a, 50, 51, 52).
Implements cross-cycle consistency loss for unpaired multi-contrast learning.
Supports PCGrad gradient surgery, FNO-based EPG, and zero-padded dipole fields.
"""

import logging
from pathlib import Path
from typing import Any

import torch

from spectramr.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.adversarial import (
    AdversarialMixin,
)
from spectramr.infrastructure.training.strategies.mixins.reconstruction import (
    ReconstructionMixin,
)
from spectramr.infrastructure.training.strategies.mixins.utils import pick_present

logger = logging.getLogger(__name__)


class DisentangledTrainingStrategy(
    AdversarialMixin,
    ReconstructionMixin,
    BaseTrainingStrategy,
):
    """Training strategy for Disentangled MRI synthesis with content-style separation."""

    def __init__(self, *args, **kwargs):
        """__init__."""
        super().__init__(*args, **kwargs)
        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize strategy-specific components following SSOT pattern."""
        self._verify_strategy_config(expected_modes=("disentangled", "reconstruction"))

        recon_settings = self.config.losses.reconstruction if self.config.losses else None

        # [FIX] SSOT: Fail-fast if enabled losses are missing from registry
        if self.env and hasattr(self.env, "losses") and recon_settings:
            enabled_checks = {
                "enable_perceptual": "perceptual",
                "enable_ssim": "ssim",
                "enable_lpips": "lpips",
                "enable_hfen": "hfen",
                "enable_ffl": "ffl",
                "enable_hist": "hist",
                "enable_mind_ssc": "mind_ssc",
            }
            for flag, loss_name in enabled_checks.items():
                if getattr(recon_settings, flag, False) and loss_name not in self.env.losses:
                    raise RuntimeError(
                        f"SSOT VIOLATION: Loss '{loss_name}' is ENABLED in config "
                        f"but MISSING from environment losses."
                    )

        self._in_channels = self.config.model.in_channels
        self._global_step = 0
        self._last_visual_pred: torch.Tensor | None = None
        self._last_visual_target: torch.Tensor | None = None

        model_kwargs = (
            self.config.model.model_kwargs if hasattr(self.config.model, "model_kwargs") else {}
        )
        model_kwargs = model_kwargs or {}
        learnable_flip = model_kwargs.get("learnable_flip_angle", False)
        self._bloch_layer = MultiPhysicsBlochLayer(
            default_flip_angle_deg=10.0,
            learnable_flip_angle=learnable_flip,
            eps=1e-6,
        )

        if self.device is not None:
            self._bloch_layer = self._bloch_layer.to(self.device, non_blocking=True)

        from spectramr.models.losses.computers.unified_disentangled import (
            UnifiedDisentangledLossComputer,
        )
        from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer

        self.loss_computer = UnifiedDisentangledLossComputer(self.config, self.device)
        self.loss_computer = self.loss_computer.to(self.device)

        self.adv_loss_computer = None
        if getattr(self.env, "opt_d", None) is not None:
            self.adv_loss_computer = UnifiedGANLossComputer(self.config, self.device)
            self.adv_loss_computer = self.adv_loss_computer.to(self.device)

        if getattr(self, "env", None) is not None and getattr(self.env, "opt_g", None) is not None:
            existing_params = {p for group in self.env.opt_g.param_groups for p in group["params"]}
            loss_params = list(self.loss_computer.parameters())
            if loss_params and loss_params[0] not in existing_params:
                self.env.opt_g.add_param_group(
                    {"params": loss_params, "lr": 1e-3, "name": "uncertainty_weights"}
                )

        # [FIX] PCGrad: Gradient surgery for physics vs adversarial conflicts
        self._enable_pcgrad = model_kwargs.get("enable_pcgrad", False)
        if self._enable_pcgrad:
            logger.info("PCGrad enabled: gradient surgery for physics/adversarial")

        # [FIX] Dipole kernel z-padding for 2.5D slabs
        self._dipole_pad_z = int(model_kwargs.get("dipole_pad_z", 0))
        if self._dipole_pad_z > 0:
            logger.info("Dipole z-padding: %d slices per side", self._dipole_pad_z)

        # [NEW] FNO EPG surrogate for FSE/FLAIR synthesis
        self._use_fno_epg = model_kwargs.get("use_fno_epg", False)
        self._fno_epg = None
        if self._use_fno_epg:
            try:
                from spectramr.infrastructure.physics.fno_epg import FNOEPGSurrogate

                self._fno_epg = FNOEPGSurrogate().to(self.device)
                logger.info("FNO EPG surrogate initialized")
            except Exception as exc:
                logger.warning("FNO EPG init failed, falling back to analytical EPG: %s", exc)
                self._fno_epg = None

    @property
    def loss_weights(self) -> dict[str, float]:
        """Get loss weights from config (SSOT)."""
        weights = {}
        recon_settings = self.config.losses.reconstruction if self.config.losses else None
        latent_settings = self.config.losses.latent if self.config.losses else None

        if recon_settings:
            weight_mapping = {
                "lambda_recon": "recon",
                "lambda_l1": "l1",
                "lambda_style": "style",
                "lambda_content_consistency": "content",
                "lambda_bloch": "bloch",
                "lambda_anat": "anat",
                "lambda_mind_ssc": "mind_ssc",
                "lambda_latent_consistency": "latent_consistency",
                "lambda_hist": "histogram_consistency",
                "lambda_ffl": "focal_frequency",
                "lambda_hfen": "hfen",
                "lambda_complex_mse": "complex_mse",
                "lambda_perceptual": "perceptual",
                "lambda_lpips": "lpips",
                "lambda_ssim": "ssim",
            }
            for config_key, loss_name in weight_mapping.items():
                if hasattr(recon_settings, config_key):
                    weight = getattr(recon_settings, config_key)
                    if isinstance(weight, (int, float)) and weight > 0:
                        weights[loss_name] = weight

        if latent_settings:
            kl = latent_settings.lambda_kl
            if isinstance(kl, (int, float)) and kl > 0:
                weights["kl"] = kl
        return weights

    def _resolve_prefixed_loss_weight(self, config_key: str, default: float = 0.0) -> float:
        """Resolve a raw ``lambda_``-prefixed weight from this strategy's config.

        Renamed from ``_get_loss_weight`` (2026-07-01): the old name shadowed
        ``BaseTrainingStrategy._get_loss_weight(loss_name, epoch=0, **kwargs)``
        with an incompatible signature — any generic caller would have bound
        ``epoch`` to ``default`` and silently received ``float(epoch)`` as a
        loss weight, with the base seam's iteration warm-up masking lost. This
        helper reads raw *config field names* (``lambda_kl``, ``lambda_bloch``,
        …) from the ``reconstruction``/``latent`` sections; the inherited base
        seam keeps its un-prefixed loss-name contract.
        """
        recon_settings = self.config.losses.reconstruction if self.config.losses else None
        if recon_settings and hasattr(recon_settings, config_key):
            val = getattr(recon_settings, config_key)
            if isinstance(val, (int, float)):
                return float(val)

        latent_settings = self.config.losses.latent if self.config.losses else None
        if latent_settings and hasattr(latent_settings, config_key):
            val = getattr(latent_settings, config_key)
            if isinstance(val, (int, float)):
                return float(val)
        return default

    @accepts_step_io
    def train_step(self, batch: Any, epoch: int, **kwargs: Any) -> list[dict[str, Any]]:
        """Optimized training step using closures."""
        x_a, x_b = self._unpack_batch(batch)
        device = x_a.device
        self._last_step_metrics = {}

        def gen_closure() -> torch.Tensor:
            """gen_closure.

            Returns:
                torch.Tensor: Description.
            """
            results = self._generator_forward_logic(x_a, x_b, batch)
            x_a_target, x_a_recon = x_a, results["x_a_recon"]

            if x_a_recon.shape != x_a_target.shape:
                if x_a_recon.shape[2:] != x_a_target.shape[2:]:
                    import torch.nn.functional as F

                    x_a_target = F.interpolate(
                        x_a_target,
                        size=x_a_recon.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if x_a_recon.shape[1] != x_a_target.shape[1]:
                    if x_a_target.shape[1] == 3 and x_a_recon.shape[1] == 1:
                        x_a_target = x_a_target[:, 1:2, ...]
                    else:
                        x_a_target = x_a_target[:, : x_a_recon.shape[1], ...]

            if x_b is not None and "x_ab" in results:
                x_b_target, x_ab = x_b, results["x_ab"]
                if x_ab.shape != x_b_target.shape:
                    if x_ab.shape[2:] != x_b_target.shape[2:]:
                        import torch.nn.functional as F

                        x_b_target = F.interpolate(
                            x_b_target,
                            size=x_ab.shape[2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    if x_ab.shape[1] != x_b_target.shape[1]:
                        if x_b_target.shape[1] == 3 and x_ab.shape[1] == 1:
                            x_b_target = x_b_target[:, 1:2, ...]
                        else:
                            x_b_target = x_b_target[:, : x_ab.shape[1], ...]
                results["x_b"] = x_b_target

            loss_output = self.loss_computer.compute(
                pred=x_a_recon,
                target=x_a_target,
                epoch=epoch,
                iteration=kwargs.get("iteration", 0),
                **results,
                is_paired=(x_b is not None and x_b is not x_a),
                batch=batch,
                model=self.env.generator,
            )

            # Integrate Adversarial GAN Loss if discriminator exists
            adv_loss_total = 0.0
            gan_config = (
                self.config.losses.gan
                if hasattr(self.config, "losses") and hasattr(self.config.losses, "gan")
                else None
            )

            if (
                gan_config
                and gan_config.enable_adversarial
                and self.adv_loss_computer
                and getattr(self.env, "discriminator", None)
            ):
                adv_output = self.adv_loss_computer.compute_generator_loss(
                    pred=x_a_recon,
                    target=x_a_target,
                    discriminator=self.env.discriminator,
                    epoch=epoch,
                    iteration=kwargs.get("iteration", 0),
                )
                adv_loss_total = adv_output.total

                # Store adversarial components as DETACHED TENSORS — no per-step
                # host sync. The old ``float(v.detach())`` fired a GPU→CPU sync per
                # key EVERY step despite the "no sync: NN#9" comment; get_last_metrics
                # now fuses the conversion into ONE sync when it is read.
                with torch.no_grad():
                    self._last_step_metrics["g_adv_loss"] = (
                        adv_loss_total.detach()
                        if isinstance(adv_loss_total, torch.Tensor)
                        else adv_loss_total
                    )
                    if hasattr(adv_output, "components"):
                        for k, v in adv_output.components.items():
                            self._last_step_metrics[f"g_{k}"] = (
                                v.detach() if isinstance(v, torch.Tensor) else v
                            )

            self._last_step_metrics.update(loss_output.to_dict())
            total_gen_loss = loss_output.total + adv_loss_total
            with torch.no_grad():
                self._last_step_metrics["g_total_loss"] = (
                    total_gen_loss.detach()
                    if isinstance(total_gen_loss, torch.Tensor)
                    else total_gen_loss
                )
            self._last_step_metrics["loss"] = self._last_step_metrics["g_total_loss"]

            self._last_x_a_recon, self._last_x_a_target = x_a_recon, x_a_target
            # Training metrics (PSNR, etc.)
            self._last_step_metrics.update(
                self._compute_training_metrics(
                    x_a_recon, x_a_target, self.config, kwargs.get("iteration", 0)
                )
            )

            if not torch.isfinite(loss_output.total):
                return torch.tensor(0.0, device=device, requires_grad=True)

            # [FIX] PCGrad: If enabled, resolve gradient conflicts between
            # physics/reconstruction losses and adversarial losses.
            if (
                self._enable_pcgrad
                and isinstance(adv_loss_total, torch.Tensor)
                and adv_loss_total.requires_grad
            ):
                from spectramr.infrastructure.optimization.pcgrad import (
                    project_conflicting_gradients,
                )

                params = [p for p in self.env.generator.parameters() if p.requires_grad]
                grad_physics = torch.autograd.grad(
                    loss_output.total, params, retain_graph=True, allow_unused=True
                )
                grad_adv = torch.autograd.grad(
                    adv_loss_total, params, retain_graph=True, allow_unused=True
                )
                grad_projected = project_conflicting_gradients(list(grad_physics), list(grad_adv))
                # [FIX] Accumulate PCGrad projections to preserve
                # gradient_accumulation_steps behavior (= overwrites, += accumulates)
                for p, g_p, g_s in zip(params, grad_physics, grad_projected, strict=False):
                    new_grad = (g_p + g_s) if g_s is not None else g_p
                    if new_grad is not None:
                        if p.grad is None:
                            p.grad = new_grad.clone()
                        else:
                            p.grad += new_grad
                # PCGrad populated ``p.grad`` by hand above, so this closure
                # must hand the executor something it can run backward on
                # WITHOUT disturbing those gradients -- and whose *value* is
                # still the real loss, because the executor logs what it gets
                # (``losses_record[name] = loss.detach()``).
                #
                # ``.detach()`` supplies the value; the zero-weighted parameter
                # term supplies a graph. ``d(0 * p.sum())/dp == 0``, so every
                # ``p.grad`` accumulates exactly 0.0 and keeps its projected
                # value. Attaching to one parameter rather than multiplying the
                # total by zero keeps this O(1): the latter would cost a second
                # full traversal of the loss graph every step, on top of the two
                # ``autograd.grad`` traversals PCGrad already pays.
                #
                # This used to return ``loss_output.total.detach().requires_grad_(True)``,
                # which worked only because ``backward()`` on a leaf is a silent
                # no-op -- the same mechanism that lets a genuinely severed loss
                # train nothing (#1952). The pre-backward graph guard now refuses
                # leaves, and this path is not exempted from it: an exemption
                # would be indistinguishable from the defect.
                return loss_output.total.detach() + 0.0 * params[0].sum()

            return total_gen_loss

        def disc_closure() -> torch.Tensor:
            """disc_closure.

            Returns:
                torch.Tensor: Description.
            """
            gan_cfg = (
                self.config.losses.gan
                if hasattr(self.config, "losses") and hasattr(self.config.losses, "gan")
                else None
            )
            if not (
                gan_cfg
                and gan_cfg.enable_adversarial
                and hasattr(self, "_last_x_a_target")
                and self.adv_loss_computer
                and getattr(self.env, "opt_d", None)
            ):
                return torch.tensor(0.0, device=device, requires_grad=True)

            d_loss_output = self.adv_loss_computer.compute_discriminator_loss(
                real=self._last_x_a_target,
                fake=self._last_x_a_recon.detach(),
                discriminator=self.env.discriminator,
                epoch=epoch,
                iteration=kwargs.get("iteration", 0),
            )

            d_total = d_loss_output.total
            # Store detached TENSORS, not float() — no per-step GPU→CPU sync
            # inside the closure (converted once in get_last_metrics). NN#9.
            with torch.no_grad():
                self._last_step_metrics["d_total_loss"] = (
                    d_total.detach() if isinstance(d_total, torch.Tensor) else d_total
                )
                if hasattr(d_loss_output, "components"):
                    for k, v in d_loss_output.components.items():
                        self._last_step_metrics[f"d_{k}"] = (
                            v.detach() if isinstance(v, torch.Tensor) else v
                        )

            return d_total

        configs = [
            {
                "name": "generator",
                "closure": gen_closure,
                "optimizer": self.env.opt_g,
                "model": self.env.generator,
            }
        ]
        gan_config_final = (
            self.config.losses.gan
            if hasattr(self.config, "losses") and hasattr(self.config.losses, "gan")
            else None
        )
        if (
            gan_config_final
            and gan_config_final.enable_adversarial
            and getattr(self.env, "opt_d", None)
        ):
            configs.append(
                {
                    "name": "discriminator",
                    "closure": disc_closure,
                    "optimizer": self.env.opt_d,
                    "model": getattr(self.env, "discriminator", None),
                }
            )
        return configs

    def get_last_metrics(self) -> dict[str, Any]:
        """Return the metrics collected during the last train_step.

        Values stay **on-device** (#707), matching the other three implementations
        (``base``, ``gan``, ``mixins.adversarial``). This used to fuse the tensor
        entries to floats here, which was already far better than a sync per key --
        but `training_loop` calls this on EVERY iteration, outside the
        ``log_interval`` gate, so a fused transfer per step is still a per-step
        sync. The loop's gated converter fuses them there instead, and this
        strategy no longer disagrees with its siblings about what the method
        returns.

        Non-tensor entries pass through unchanged, as the previous ``.copy()``
        did -- ``loss_output.to_dict()`` may carry non-numeric fields.
        """
        return dict(getattr(self, "_last_step_metrics", {}))

    def _generator_forward_logic(
        self, x_a: torch.Tensor, x_b: torch.Tensor | None, batch: Any
    ) -> dict[str, Any]:
        """Forward pass logic for disentanglement."""
        model = self.env.generator
        results = {"x_a": x_a, "x_b": x_b if x_b is not None else x_a}
        class_idx_a = batch.get("contrast_idx")
        if class_idx_a is not None:
            class_idx_a = class_idx_a.to(x_a.device)

        # [FIX] Complex→Real guard: disentangled encoders use standard nn.Conv2d
        # which cannot handle complex tensors. Convert to magnitude representation.
        def _ensure_real(t: torch.Tensor) -> torch.Tensor:
            if torch.is_complex(t):
                return t.abs()
            return t

        x_a = _ensure_real(x_a)
        if x_b is not None:
            x_b = _ensure_real(x_b)

        results["c_a"] = model.enc_c(x_a)
        style_out_a = model.enc_s(x_a)
        if model.use_vae:
            results["mu_a"], results["logvar_a"] = style_out_a
            results["s_a"] = model.reparameterize(results["mu_a"], results["logvar_a"])
        else:
            results["s_a"] = style_out_a

        s_a_cond = results["s_a"]
        if class_idx_a is not None and getattr(model, "class_embedding", None):
            # [FIX] Clamp indices to valid range to prevent CUDA device-side assert.
            # CANONICAL_CONTRASTS has 6 entries but num_classes may be smaller.
            num_classes = model.class_embedding.num_embeddings
            class_idx_a = class_idx_a.clamp(0, num_classes - 1)
            s_a_cond = s_a_cond + model.class_embedding(class_idx_a)

        x_a_recon_raw = model.gen(results["c_a"], s_a_cond)
        # [FIX] Inject sequence parameters to complete the causal graph:
        # style_encoder → sequence_mlp → Bloch simulator → L1 loss
        if isinstance(x_a_recon_raw, dict) and getattr(model, "sequence_mlp", None):
            seq_params = model.sequence_mlp(s_a_cond)
            x_a_recon_raw.update(seq_params)
        results["x_a_recon"] = (
            self._synthesize_via_bloch(x_a_recon_raw, batch, "a")
            if isinstance(x_a_recon_raw, dict)
            else x_a_recon_raw
        )

        if x_b is not None and x_b is not x_a:
            results["c_b"] = model.enc_c(x_b)
            style_out_b = model.enc_s(x_b)
            results["s_b"] = model.reparameterize(*style_out_b) if model.use_vae else style_out_b

            x_ab_raw = model.gen(results["c_a"], results["s_b"])
            # [FIX] Same injection for cross-translated output
            if isinstance(x_ab_raw, dict) and getattr(model, "sequence_mlp", None):
                seq_params_b = model.sequence_mlp(results["s_b"])
                x_ab_raw.update(seq_params_b)
            results["x_ab"] = (
                self._synthesize_via_bloch(x_ab_raw, batch, "ab")
                if isinstance(x_ab_raw, dict)
                else x_ab_raw
            )

            results["c_a_recon"] = model.enc_c(
                results["x_ab"]
                if results["x_ab"].shape[1] == x_a.shape[1]
                else results["x_ab"].repeat(1, x_a.shape[1], 1, 1)
            )
            style_recon_b = model.enc_s(
                results["x_ab"]
                if results["x_ab"].shape[1] == x_b.shape[1]
                else results["x_ab"].repeat(1, x_b.shape[1], 1, 1)
            )
            results["s_b_recon"] = style_recon_b[0] if model.use_vae else style_recon_b
        return results

    def _compute_losses_impl(self, input_batch, target_batch, epoch, **kwargs) -> dict:
        """Not called directly — DisentangledTrainingStrategy overrides train_step().

        Returns an empty dict to satisfy the base class contract without crashing.
        All loss computation is handled inline within :meth:`train_step`.
        """
        return {}

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Perform a single validation step."""
        batch = (input_batch, target_batch)
        if input_batch is None or target_batch is None:
            if isinstance(batch, dict):
                input_batch = pick_present(
                    batch.get("input"), batch.get("image"), batch.get("kspace")
                )
                target_batch = batch.get("target")
            else:
                lr, hr = self._unpack_batch(batch)
                input_batch, target_batch = (
                    pick_present(input_batch, lr),
                    pick_present(target_batch, hr),
                )

        if input_batch is None:
            return {}
        model = self.env.generator
        model.eval()

        with torch.no_grad():
            from spectramr.infrastructure.training.utils.data_adapters import (
                TorchIOAdapter,
            )

            def _prepare(val, in_ch):
                """_prepare.

                Args:
                    val (Any): Description.
                    in_ch (Any): Description.
                Returns:
                    Any: Description.
                """
                if isinstance(val, dict):
                    val = pick_present(
                        val.get("data"),
                        next(
                            (v for v in val.values() if isinstance(v, torch.Tensor)),
                            val,
                        ),
                    )
                return TorchIOAdapter.to_batch_format(val.to(self.device), in_ch)

            in_ch = self._in_channels
            x_a_raw = _prepare(input_batch, in_ch)

            if target_batch is not None:
                x_b_raw = _prepare(target_batch, in_ch)
            else:
                x_b_raw = x_a_raw

            # Flatten 5D to 4D [B*D, C, H, W]
            def flatten(x):
                """flatten.

                Args:
                    x (Any): Description.
                Returns:
                    Any: Description.
                """
                if x is None:
                    return None
                if x.ndim == 5:
                    # Input: [B, C, D, H, W]
                    # Target: [B*D, C, H, W]
                    b, c, d, h, w = x.shape
                    # (B, C, D, H, W) -> (B, D, C, H, W) -> (B*D, C, H, W)
                    return x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
                return x

            x_a_flat = flatten(x_a_raw)
            x_b_flat = flatten(x_b_raw)
            target_flat = x_b_flat

            if x_a_raw.ndim == 5:
                chunk_size, num_slices = 1, x_a_flat.shape[0]
                total_recon_loss, chunks_count, aggregated_metrics = 0.0, 0, {}

                for i in range(0, num_slices, chunk_size):
                    end = min(i + chunk_size, num_slices)
                    x_a_chunk = x_a_flat[i:end]
                    target_chunk = target_flat[i:end]

                    # Channel adaptation
                    x_a_in = x_a_chunk
                    if x_a_chunk.shape[1] == 2 and in_ch == 1:
                        x_a_in = x_a_chunk[:, 0:1, ...]
                    elif x_a_chunk.shape[1] == 1 and in_ch == 3:
                        x_a_in = x_a_chunk.repeat(1, 3, 1, 1)

                    c_a = model.enc_c(x_a_in)
                    style_out = model.enc_s(x_a_in)
                    s_a = model.reparameterize(*style_out) if model.use_vae else style_out
                    x_a_recon_chunk = model.gen(c_a, s_a)
                    if isinstance(x_a_recon_chunk, dict):
                        x_a_recon_chunk = self._synthesize_via_bloch(x_a_recon_chunk, batch, "a")

                    if i == 0:
                        self._last_visual_pred = x_a_recon_chunk.detach()
                        self._last_visual_target = (
                            target_chunk[:, 0:1, ...]
                            if target_chunk.shape[1] > x_a_recon_chunk.shape[1]
                            else target_chunk
                        )

                    # Loss and metrics
                    import torch.nn.functional as F

                    cur_target = target_chunk
                    if x_a_recon_chunk.shape[1] == 1 and target_chunk.shape[1] == 3:
                        cur_target = target_chunk[:, 1:2, ...]
                    elif x_a_recon_chunk.shape[1] == 1 and target_chunk.shape[1] == 2:
                        cur_target = target_chunk[:, 0:1, ...]

                    loss = F.l1_loss(x_a_recon_chunk, cur_target)
                    total_recon_loss += loss.item() * (end - i)
                    chunks_count += end - i

                    # Map [-1, 1] → [0, 1] for metrics (data uses minmax [-1,1])
                    p_n = torch.nan_to_num(x_a_recon_chunk, nan=0.0)
                    p_n = ((p_n + 1.0) / 2.0).clamp(0, 1)
                    t_n = ((cur_target + 1.0) / 2.0).clamp(0, 1)
                    chunk_m = self.validation_metrics_computer.compute(p_n, t_n)
                    for k, v in chunk_m.items():
                        aggregated_metrics[k] = aggregated_metrics.get(k, 0.0) + v * (end - i)

                metrics = {
                    "val_recon_loss": (total_recon_loss / chunks_count if chunks_count > 0 else 0.0)
                }
                for k, v in aggregated_metrics.items():
                    metrics[k] = v / chunks_count if chunks_count > 0 else 0.0
            else:
                # 4D case — process one sample at a time to avoid OOM
                import torch.nn.functional as F

                x_a, x_b = x_a_flat, x_b_flat
                total_recon_loss, chunks_count, aggregated_metrics = 0.0, 0, {}

                for i in range(x_a.shape[0]):
                    x_a_slice = x_a[i : i + 1]
                    x_b_slice = x_b[i : i + 1]

                    x_a_in = x_a_slice
                    if x_a_slice.shape[1] == 2 and in_ch == 1:
                        x_a_in = x_a_slice[:, 0:1, ...]
                    elif x_a_slice.shape[1] == 1 and in_ch == 3:
                        x_a_in = x_a_slice.repeat(1, 3, 1, 1)

                    # Validation runs in fp32 — Bloch synthesis FFT is unstable in fp16
                    c_a = model.enc_c(x_a_in)
                    style_out = model.enc_s(x_a_in)
                    s_a = model.reparameterize(*style_out) if model.use_vae else style_out
                    x_a_recon = model.gen(c_a, s_a)

                    if isinstance(x_a_recon, dict):
                        x_a_recon = self._synthesize_via_bloch(x_a_recon, batch, "a")

                    # Guard against NaN from early unstable training
                    x_a_recon = torch.nan_to_num(x_a_recon, nan=0.0)

                    cur_target = x_b_slice
                    if x_a_recon.shape[1] == 1 and x_b_slice.shape[1] == 3:
                        cur_target = x_b_slice[:, 1:2, ...]
                    elif x_a_recon.shape[1] == 1 and x_b_slice.shape[1] == 2:
                        cur_target = x_b_slice[:, 0:1, ...]

                    if i == 0:
                        # Cache the GROUND-TRUTH target (x_b), not the source
                        # input (x_a_in): pipelines/train.py saves
                        # ``_last_visual_target`` as the validation REAL image.
                        # Mirrors the 5D branch above (which already caches the
                        # channel-adapted target) so the saved "real" is the
                        # same tensor the loss/metrics grade against.
                        self._last_visual_pred = x_a_recon.detach()
                        self._last_visual_target = cur_target.detach()

                    loss = F.l1_loss(x_a_recon, cur_target)
                    total_recon_loss += loss.item()
                    chunks_count += 1

                    # Map [-1, 1] → [0, 1] for metrics (data uses minmax [-1,1])
                    p_n = ((x_a_recon + 1.0) / 2.0).clamp(0, 1)
                    t_n = ((cur_target + 1.0) / 2.0).clamp(0, 1)
                    chunk_m = self.validation_metrics_computer.compute(p_n, t_n)
                    for k, v in chunk_m.items():
                        aggregated_metrics[k] = aggregated_metrics.get(k, 0.0) + v

                metrics = {
                    "val_recon_loss": (total_recon_loss / chunks_count if chunks_count > 0 else 0.0)
                }
                for k, v in aggregated_metrics.items():
                    metrics[k] = v / chunks_count if chunks_count > 0 else 0.0

            # Logging — save images to TensorBoard and to disk
            if hasattr(self, "logging_service") and self.logging_service:
                try:

                    def to_v(t):
                        """Map [-1,1] to [0,1] for visualization."""
                        return ((torch.abs(t) if torch.is_complex(t) else t) + 1) / 2

                    step_num = self.logging_service.step or 0
                    self.logging_service.log_images("val/input", to_v(x_a_flat[:4]), step_num)
                    self.logging_service.log_images(
                        "val/recon",
                        to_v(self._last_visual_pred[:4]),
                        step_num,
                    )
                except Exception as exc:
                    logger.warning("disentangled val image logging failed: %s", exc)

            # Save validation images to disk
            try:
                import os

                from torchvision.utils import save_image

                # Direct access, not `getattr(self.config, ...)`. `metrics` has
                # a default_factory on the settings model, so a real config
                # ALWAYS carries the block and a defensive getattr buys nothing
                # -- while `test_ssot_compliance` forbids exactly that shape in
                # a strategy (non-negotiable #1: config is nested, read it
                # directly and let a missing block fail loud).
                out_dir = self.config.metrics.output_dir
                # Reject on `(str, Path)`, NOT on `os.PathLike`. A MagicMock
                # SATISFIES PathLike -- its `__fspath__` synthesises a real
                # looking relative path out of the mock's name and id --
                #
                #   os.fspath(mock.config.metrics.output_dir)
                #     -> 'MagicMock/mock.config.metrics.output_dir/1361...'
                #
                # so a test passing a mock config made `makedirs` build that
                # tree in the process CWD, i.e. the repo root, once per run
                # (#917). The `hasattr` chain this replaces could not catch it
                # either: a mock answers every `hasattr` True, so the guard was
                # unfalsifiable for exactly the input that needed guarding.
                if out_dir is not None and not isinstance(out_dir, (str, Path)):
                    raise TypeError(
                        "metrics.output_dir must be str or Path, got "
                        f"{type(out_dir).__name__}. A stand-in config that is "
                        "not a real schema reaches real filesystem code here."
                    )
                if out_dir:
                    img_dir = os.path.join(out_dir, "val_images")
                    os.makedirs(img_dir, exist_ok=True)
                    step_num = getattr(self, "_global_step", 0)

                    # Map [-1,1] -> [0,1] for saving
                    pred_vis = ((self._last_visual_pred[:4] + 1) / 2).clamp(0, 1)
                    target_vis = ((self._last_visual_target[:4] + 1) / 2).clamp(0, 1)

                    save_image(
                        pred_vis,
                        os.path.join(img_dir, f"recon_iter{step_num:06d}.png"),
                        nrow=4,
                        normalize=False,
                    )
                    save_image(
                        target_vis,
                        os.path.join(img_dir, f"target_iter{step_num:06d}.png"),
                        nrow=4,
                        normalize=False,
                    )
            except Exception as exc:
                logger.warning("disentangled val image save failed: %s", exc)
            model.train()
            return metrics

    def _compute_tissue_bounds_loss(self, tp_a, tp_b) -> torch.Tensor:
        """Tissue bounds regularization."""
        total_loss, count = torch.tensor(0.0, device=self.device), 0
        for tp in [tp_a, tp_b]:
            if tp is None:
                continue
            t1, t2 = tp["t1"], tp["t2"]
            # NO ``.item()`` here: it would detach the penalty into a gradient-free
            # constant, so if this regularizer is ever summed into the objective it
            # would contribute exactly zero gradient (a silent dead-loss trap).
            total_loss = total_loss + (
                torch.relu(t2 + 50.0 - t1).mean()
                + torch.relu(t1 - 5000.0).mean()
                + torch.relu(10.0 - t2).mean()
            )
            count += 1
        return total_loss / count if count > 0 else total_loss

    def _prepare_model_inputs(self, batch: dict[str, Any]) -> tuple:
        """Prepare model inputs."""
        x_a, x_b = self._unpack_batch(batch)
        if x_a is None:
            raise ValueError("Input x_a is None")
        x_b = x_b if x_b is not None else x_a

        # Simple MAG conversion if needed
        if self._in_channels == 1:
            if x_a.shape[1] == 2:
                x_a = torch.sqrt(x_a[:, 0:1] ** 2 + x_a[:, 1:2] ** 2 + 1e-8)
            if x_b.shape[1] == 2:
                x_b = torch.sqrt(x_b[:, 0:1] ** 2 + x_b[:, 1:2] ** 2 + 1e-8)

        return (
            x_a.to(self.device),
            x_b.to(self.device),
            None,
            None,
            self.generator_model.use_vae,
            x_b is not x_a,
        )

    def _synthesize_via_bloch(
        self, tissue_params: dict, batch: dict, _source_key: str
    ) -> torch.Tensor:
        """Synthesize MRI from tissue parameters.

        Supports three synthesis pathways:
        1. FNO EPG (if use_fno_epg=True and FNO initialized) for FSE/IR sequences
        2. Analytical EPG (fallback for FSE/IR) with B1+ field modulation
        3. MultiPhysicsBlochLayer for GRE/SPGR/DWI sequences

        Note: Bloch synthesis is detached from the backward pass because
        the physics equations (exp(-TR/T1)) produce gradient norms of 100M+.
        Reconstruction losses on the output image provide sufficient
        gradient signal for learning tissue parameters.
        """
        # validation_step passes a ``(input, target)`` TUPLE as ``batch`` (it
        # never carries the metadata dict), but the sequence-param lookups in
        # _synthesize_via_bloch_fp32 call ``batch.get(...)`` -> AttributeError
        # on a tuple. Coerce a non-dict batch to {} so those lookups are safe;
        # the acquisition params are then resolved from the model prediction
        # (disentangled_mri always emits predicted_{tr,te,ti,alpha}) via
        # ``_resolve_acq_param`` — which fails loud if neither the model nor the
        # batch supplies a value (no silent GRE default, pitfall #9).
        if not isinstance(batch, dict):
            batch = {}
        # [FIX] Force fp32 for Bloch synthesis — FFT, exponentials, and dipole
        # kernel operations overflow in fp16 under AMP, producing NaN.
        # Gradients flow normally; per-sample normalization in _synthesize_via_bloch_fp32
        # bounds signal to [-1, 1] and damps gradients by 1/sig_max.
        with torch.autocast(device_type="cuda", enabled=False):
            tissue_params_fp32 = {
                k: v.float() if isinstance(v, torch.Tensor) else v for k, v in tissue_params.items()
            }
            return self._synthesize_via_bloch_fp32(tissue_params_fp32, batch, _source_key)

    @staticmethod
    def _resolve_acq_param(tissue_params: dict, batch: dict, pred_key: str, batch_key: str) -> Any:
        """Resolve an acquisition timing param from the model prediction, else
        the batch metadata; raise if neither supplies it.

        Previously this fell back to a hardcoded GRE default (TR=8, TE=4,
        alpha=10, TI=0), silently fabricating the contrast and rendering the
        Bloch synthesis physically meaningless whenever the model does not
        predict acquisition params and the batch carries no metadata
        (pitfall #9, silent fallback). The ``disentangled_mri`` generator always
        emits ``predicted_{tr,te,ti,alpha}``, so this only fires for a mis-wired
        model/dataset — fail loud there instead of degrading.
        """
        v = tissue_params.get(pred_key)
        if v is not None:
            return v
        v = batch.get(batch_key)
        if v is not None:
            return v
        raise ValueError(
            f"disentangled Bloch synthesis: acquisition parameter '{batch_key}' "
            f"is absent from both the model output ('{pred_key}') and the batch "
            f"metadata. Use a model that predicts acquisition params (e.g. "
            f"disentangled_mri) or provide '{batch_key}' in the batch — no "
            f"hardcoded default is applied (pitfall #9)."
        )

    def _synthesize_via_bloch_fp32(
        self, tissue_params: dict, batch: dict, _source_key: str
    ) -> torch.Tensor:
        """Inner Bloch synthesis running in fp32."""
        # [FIX] Enforce physics/style coupling: prioritize network predictions
        # over batch metadata, completing the differentiable causal chain.
        metadata = {
            "TR": self._resolve_acq_param(tissue_params, batch, "predicted_tr", "TR"),
            "TE": self._resolve_acq_param(tissue_params, batch, "predicted_te", "TE"),
            "TI": self._resolve_acq_param(tissue_params, batch, "predicted_ti", "TI"),
            "alpha": self._resolve_acq_param(tissue_params, batch, "predicted_alpha", "alpha"),
            "contrast_type": batch.get("contrast_type", "GRE"),
        }

        contrast_idx = batch.get("contrast_idx")
        # 1 and 2 are usually T2w and FLAIR based on standard configs.
        is_fse = (contrast_idx is not None and contrast_idx in [1, 2]) or (
            metadata["contrast_type"] in ["SE", "IR"]
        )

        _bloch_layer_normalized = False
        # Path 1: FNO EPG surrogate (O(N log N) differentiable replacement)
        if is_fse and self._fno_epg is not None:
            signal = self._fno_epg(
                rho=tissue_params["rho"],
                t1=tissue_params["t1"],
                t2=tissue_params["t2"],
                b1_plus=tissue_params.get("b1_plus"),
                te_spacing=10.0,
                echo_train_length=16,
            )

            # If IR/FLAIR, apply inversion recovery scaling
            if (contrast_idx is not None and contrast_idx == 2) or metadata[
                "contrast_type"
            ] == "IR":
                TI_val = metadata["TI"]
                if isinstance(TI_val, torch.Tensor):
                    TI_val = TI_val.view(-1, 1, 1, 1)
                t1 = tissue_params["t1"]
                ir_prep = torch.abs(1.0 - 2.0 * torch.exp(-TI_val / t1))
                signal = signal * ir_prep

        # Path 2: Analytical EPG (simplified differentiable proxy)
        elif is_fse and "b1_plus" in tissue_params:
            from spectramr.infrastructure.physics.epg import (
                simulate_differentiable_epg_fse,
            )

            rho = tissue_params["rho"]
            t1 = tissue_params["t1"]
            t2 = tissue_params["t2"]
            b1_plus_map = tissue_params["b1_plus"]

            # Modulate nominal flip angles by the spatial B1+ transmit field map
            excitation_fa = 90.0 * b1_plus_map
            refocusing_fa = 180.0 * b1_plus_map

            signal = simulate_differentiable_epg_fse(
                rho=rho,
                t1=t1,
                t2=t2,
                excitation_fa=excitation_fa,
                refocusing_fa=refocusing_fa,
                echo_train_length=16,
            )

            # If IR/FLAIR, apply inversion recovery scaling
            if (contrast_idx is not None and contrast_idx == 2) or metadata[
                "contrast_type"
            ] == "IR":
                TI_val = metadata["TI"]
                if isinstance(TI_val, torch.Tensor):
                    TI_val = TI_val.view(-1, 1, 1, 1)
                ir_prep = torch.abs(1.0 - 2.0 * torch.exp(-TI_val / t1))
                signal = signal * ir_prep

        # Path 3: MultiPhysicsBlochLayer (GRE/SPGR/DWI)
        else:
            signal = self._bloch_layer(tissue_params, metadata)
            # Path 3 self-normalizes via _normalize_signal() — skip sig_max below
            _bloch_layer_normalized = True

        if "bias_field" in tissue_params:
            signal = signal * tissue_params["bias_field"]

        # [FIX] Map susceptibility χ to R2' magnitude decay (not phase).
        # The old code |S·e^{-iφ}| = |S| erased all phase information,
        # making the susceptibility_head a dead layer with zero gradients.
        # In magnitude-only MRI, B0 inhomogeneity → intravoxel dephasing (T2* decay).
        if "chi" in tissue_params:
            from spectramr.infrastructure.physics.dipole import apply_3d_dipole_kernel

            # [FIX] Scale O(1) logits to physiological ppm bounds
            chi_map_ppm = tissue_params["chi"] * 1e-6

            is_2d = chi_map_ppm.ndim == 4
            if is_2d:
                chi_map_ppm = chi_map_ppm.unsqueeze(2)

            gamma_bar = 42.58e6  # Hz/T for protons

            # [FIX] Zero-pad z-axis to prevent dipole wrap-around on 2.5D slabs
            delta_B0 = apply_3d_dipole_kernel(
                chi_map_ppm, voxel_sizes=(1.0, 1.0, 1.0), pad_z=self._dipole_pad_z
            )

            # [FIX] R2' (1/s) ≈ γ·|ΔB0| → observable magnitude decay
            r2_prime = gamma_bar * torch.abs(delta_B0)

            # Extract TE, handling scalar or tensor
            te_val = metadata["TE"]
            if isinstance(te_val, torch.Tensor):
                while te_val.dim() < r2_prime.dim():
                    te_val = te_val.unsqueeze(-1)
            else:
                te_val = float(te_val)

            te_sec = te_val / 1000.0

            # Intravoxel dephasing: signal *= exp(-R2' × TE)
            # [FIX] Clamp exponent to max 5.0 to prevent total signal erasure
            # from unconstrained χ (exp(-5) ≈ 0.007 > 0)
            dephasing_exponent = (r2_prime * te_sec).clamp(max=5.0)
            dephasing_factor = torch.exp(-dephasing_exponent)

            if is_2d:
                dephasing_factor = dephasing_factor.squeeze(2)

            if signal.ndim == 4 and dephasing_factor.ndim == 5:
                b, c, d, h, w = dephasing_factor.shape
                dephasing_factor = dephasing_factor.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)

            signal = signal * dephasing_factor

        if self._in_channels == 3 and signal.shape[1] == 1:
            signal = signal.repeat(1, 3, 1, 1)

        # [FIX] Normalize signal to [0, 1] per sample.
        # Detach amax to sever the 1/c Jacobian amplifier in the backward pass.
        # Skip if Path 3 (MultiPhysicsBlochLayer) already self-normalized.
        if not _bloch_layer_normalized:
            sig_max = signal.flatten(1).amax(dim=1).view(-1, 1, 1, 1).detach()
            signal = signal / torch.clamp(sig_max, min=1e-3)

        return signal * 2.0 - 1.0
