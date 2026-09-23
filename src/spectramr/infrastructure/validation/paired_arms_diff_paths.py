"""Canonical allow-list of dotted paths two paired arms may legitimately differ on.

Extracted from :mod:`paired_arms_audit` so that module stays under the 300-LOC
ceiling (non-negotiable 20). This is *data*, not behaviour: the audit logic that
consumes it lives next door and is the only reader.

Every entry is spelled with its CANONICAL path.
:func:`paired_arms_audit._canonical_paths` folds each observed path through the
rename SSOT before matching, so an entry left at a retired spelling can never
match anything -- it does not fail, it silently stops exempting its knob and the
audit then reports a spurious diff on every pair.
``test_paired_arms_diff_paths.py::test_no_allowlist_entry_is_a_retired_path``
pins that.
"""

from __future__ import annotations

# Dotted-path keys that are ALWAYS permitted to differ between arms.
#
# Spell every entry with its CANONICAL path. `_walk` canonicalises each observed
# path through the rename SSOT before matching, so an entry left at a retired
# spelling can never match anything -- it does not fail, it silently stops
# exempting its knob and the audit reports a spurious diff on every pair.
# `test_paired_arms_audit.py::test_no_allowlist_entry_is_a_retired_path` pins it.
# Anything that names an output directory, a checkpoint, a method
# label, an arm letter, or paired-with metadata is automatically
# exempt — those are part of the arm's identity, not its experimental
# specification.
DEFAULT_DIFF_PATHS: frozenset[str] = frozenset(
    {
        "metadata.name",
        "metadata.campaign_arm",
        "metadata.paired_with",
        "metadata.description",
        "metadata.hypothesis",
        "metadata.baseline",
        "metadata.negative_result_plan",
        "metadata.secondary_metrics",
        # Read as a pair with `metadata.baseline`, already exempt above: it names
        # the direction the hypothesis predicts ('comparable' / 'ceiling' /
        # 'floor'), so it differs by construction wherever the arms differ at all.
        "metadata.expected_outcome",
        "metadata.audit_waivers",
        "metadata.tags.type",
        "metadata.tags.domain",
        "metadata.tags.dc_mechanism",
        # Domain / paradigm-locus differences (the FACTOR under test).
        "training.input_domain",
        "training.output_domain",
        "training.output_dir",
        # The equivariance-ablation factor. At 0 the objective reduces exactly to
        # measurement-consistency-only reconstruction, which is the control the
        # self_supervised cohort is read against; the treatment sets 1.0.
        "training.equivariant_imaging.alpha_equivariance",
        "training.diffusion.degradation",
        "training.diffusion.enforce_output_range",
        "training.diffusion.enable_proximal_dc",
        "training.diffusion.proximal_dc_mode",
        "training.diffusion.proximal_dc_weight",
        "model.model_type",
        "model.in_channels",
        "model.out_channels",
        "model.target_domain",
        "model.input_type",
        "model.output_type",
        "model.model_kwargs.degradation_type",
        "model.model_kwargs.channel_mults",
        "model.model_kwargs.dropout",
        "model.model_kwargs.beta_schedule",
        "model.model_kwargs.cond_dim",
        "model.model_kwargs.use_complex_forward",
        "model.model_kwargs.log_scaling",
        "model.model_kwargs.force_pure_kspace",
        "model.model_kwargs.time_embedding_dim",
        "model.model_kwargs.phase_safe_attention_hidden_dim",
        "model.model_kwargs.reflect_padding_bottleneck_layers",
        "model.model_kwargs.process_type",
        "model.model_kwargs.digital_twin_kwargs",
        "model.model_kwargs.forward_operator_kwargs",
        "adapters",
        "losses.policy.output_domain",
        "losses.kspace_losses",
        "physics.kspace.enable_kspace_recon",
        "physics.kspace.data_range",
        "physics.kspace.max_magnitude",
        "data.domain.output",
        "data.collation.log_strategy_selection",
        "data.collation",
        # The NEX-target ablation factor. Before these two the list carried three
        # `data.*` paths and none was a target mode, so no repetition-target
        # ablation could pass however cleanly it was built -- the n2n cohort
        # varies exactly these and was reported as three spurious diffs.
        # `r2r_alpha` rides with it: it is read only under `target_mode: r2r`, so
        # it is absent on every arm that does not select that mode.
        "data.target_mode",
        "data.r2r_alpha",
        "data.r2r_covariance_source",
        "acceleration.center_fraction",
        "acceleration.min_center_fraction",
        "metrics.transform",
        "metrics.compute_kspace_error",
        "metrics.output_dir",
        "validation.scoring.output_transform",
        "validation.scoring.compute",
        "checkpoint.save_dir",
        "checkpoint.checkpoint_dir",
        "logging.sinks.dir",
        "logging.identity.experiment",
        "logging.identity.run",
        "loss_logging.csv_path",
        "loss_logging.output_dir",
        "reporting.method_name",
        "reporting.metrics",
        "optimization.gradient.enable_checkpointing",
    }
)

__all__ = ["DEFAULT_DIFF_PATHS"]
