#!/usr/bin/env bash
# =============================================================================
# gpu_count.sh — SSOT for "how many GPUs did this node actually get?".
#
# Source this; it defines `derive_gpus_per_node` and does nothing else.
#
# WHY THIS EXISTS
# ---------------
# GPUS_PER_NODE feeds `torchrun --nproc_per_node` directly, so it *is* the
# per-node world size. The derivation used to end in `|| echo 1`: any input
# producing no trailing digits — both Slurm variables unset, or a type-only
# value like `ada` — resolved to 1, so a four-GPU allocation forked a single
# rank. Training ran to completion, the checkpoint was valid, the metrics were
# real, and nothing in the logs said three quarters of the requested hardware
# was never touched (#1274). That is non-negotiable 9 in its most expensive
# form, because the evidence is absent rather than wrong.
#
# It lives here rather than inline because TWO launchers now fork ranks from it
# — `train_distributed.sbatch` and `dispatch_experiments.sbatch` — and a
# second copy of a rule that has already cost one silent single-rank run is the
# shape non-negotiable 17 forbids. `tests/unit/scripts/test_train_distributed_sbatch.py`
# executes this file's body against controlled environments.
#
# The count comes from Slurm's own ALLOCATED-GPU count, never from parsing the
# request string, which is why every request spelling works identically:
# `--gres=gpu:<type>:N`, `--gpus=<type>:N`, `--gpus=N`, `--gpus-per-node=N`.
# =============================================================================

# --- gpu-count-derivation (executed verbatim by the paired test) -------------
_gpu_count_from() {  # trailing digits of $1, or nothing at all
    [[ -n "${1:-}" ]] && echo "$1" | grep -oE '[0-9]+$' || true
}

# Set GPUS_PER_NODE from the environment, or exit 1. Honours a caller-supplied
# GPUS_PER_NODE verbatim, so an operator can state the count explicitly.
#
# SLURM_GPUS_ON_NODE is a bare number; SLURM_GPUS_PER_NODE may carry the type
# label (`ada:2`), hence trailing digits rather than the whole value.
derive_gpus_per_node() {
    if [[ -z "${GPUS_PER_NODE:-}" ]]; then
        GPUS_PER_NODE="$(_gpu_count_from "${SLURM_GPUS_ON_NODE:-}")"
        [[ -n "${GPUS_PER_NODE}" ]] || GPUS_PER_NODE="$(_gpu_count_from "${SLURM_GPUS_PER_NODE:-}")"
        # SLURM_GPUS is a JOB TOTAL, not a per-node count -- `--gpus=N` populates
        # only this one, and on a 2-node allocation SLURM_GPUS=8 means 4 per node.
        # Divide, or a correct multi-node launch is read as over-forking. The
        # framework's own idle-device guard splits these into two tuples for the
        # same reason (core/resources.py: ALLOC_GPU_ENV_PER_NODE vs _JOB), and it
        # is the guard this must agree with: it REFUSES a launch whose grant
        # exceeds the ranks started, so a derivation that skips a source the
        # guard reads gets a correct launch rejected on the node.
        if [[ -z "${GPUS_PER_NODE}" ]]; then
            _total="$(_gpu_count_from "${SLURM_GPUS:-}")"
            if [[ -n "${_total}" ]]; then
                GPUS_PER_NODE=$(( _total / ${SLURM_NNODES:-1} ))
                (( GPUS_PER_NODE < 1 )) && GPUS_PER_NODE=""
            fi
        fi
        # Last resort, and deliberately last: nvidia-smi reports the node's
        # PHYSICAL devices, which on a cluster without ConstrainDevices can
        # EXCEED this job's grant. Never promote it above the Slurm sources —
        # it answers "what is on this machine", not "what was allocated to me".
        if [[ -z "${GPUS_PER_NODE}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
            GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
        fi
    fi

    if [[ ! "${GPUS_PER_NODE:-}" =~ ^[0-9]+$ ]] || (( GPUS_PER_NODE < 1 )); then
        echo "FATAL: could not determine GPUs per node (got '${GPUS_PER_NODE:-}')." >&2
        echo "  SLURM_GPUS_ON_NODE='${SLURM_GPUS_ON_NODE:-}'" \
             "SLURM_GPUS_PER_NODE='${SLURM_GPUS_PER_NODE:-}'" >&2
        echo "  Request GPUs (--gres=gpu:<type>:N, --gpus=<type>:N, or SLURM_GPUS=N" \
             "via submit_experiment_array.sh), or export GPUS_PER_NODE=<n> to state" \
             "it explicitly. Refusing to assume 1: that would silently run a single" \
             "rank on a multi-GPU allocation." >&2
        exit 1
    fi
    export GPUS_PER_NODE
}
# --- end gpu-count-derivation ------------------------------------------------
