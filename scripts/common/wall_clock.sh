#!/usr/bin/env bash
# =============================================================================
# wall_clock.sh — SSOT for "when does this allocation end?".
#
# Source this; it defines `derive_wall_clock_deadline` and does nothing else.
#
# WHY THIS EXISTS
# ---------------
# A production arm gets 120 h per job and needs more, so the run must survive
# being cut in half. What was missing was never *resume* — the checkpoint has
# carried model / optimizer / scheduler / scaler / EMA / RNG / counter state all
# along — it was STOPPING IN TIME TO WRITE ONE. A job killed at the wall loses
# everything since the last periodic save, which at the corpus-typical
# `save_interval: 5000` is hours of GPU per link of the chain.
#
# So we export the allocation's end as an absolute epoch second and let the
# training loop stop itself a margin ahead of it
# (`spectramr.core.wall_clock`, which owns the margin and the yield decision —
# this file only answers when the wall is).
#
# WHY A DEADLINE AND NOT `--signal=B:USR1@N`
# ------------------------------------------
# The signal is delivered to the batch step. On the array path that step has
# `exec`'d into `spectramr.cli.manifest_dispatch`, which launches `train` — or
# `torch.distributed.run`, which does not forward SIGUSR1 and whose default
# disposition kills the launcher and orphans its workers. An environment
# variable is inherited by every descendant and needs no forwarding.
#
# It lives here rather than inline because both launchers need it, and a second
# copy of a derivation is the shape non-negotiable 17 forbids (see gpu_count.sh,
# whose duplicate cost one silent single-rank run).
# =============================================================================

# Set SPECTRAMR_WALL_CLOCK_DEADLINE (absolute epoch seconds) from Slurm, or
# leave it unset. Unset is a legitimate answer — no limit, or a Slurm too old to
# report one — and means "train to max_iterations", so this NEVER exits nonzero.
# Which source answered is printed, because a deadline that silently failed to
# derive looks exactly like a job with no limit until it dies at the wall.
derive_wall_clock_deadline() {
    if [[ -n "${SPECTRAMR_WALL_CLOCK_DEADLINE:-}" ]]; then
        echo "[WallClock] deadline stated explicitly: ${SPECTRAMR_WALL_CLOCK_DEADLINE}"
        export SPECTRAMR_WALL_CLOCK_DEADLINE
        return 0
    fi

    # Slurm >= 23.02 hands the end time to the job directly. Cheapest and
    # exact; every fallback below costs an RPC to the controller.
    if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
        export SPECTRAMR_WALL_CLOCK_DEADLINE="${SLURM_JOB_END_TIME}"
        echo "[WallClock] deadline from SLURM_JOB_END_TIME: ${SPECTRAMR_WALL_CLOCK_DEADLINE}"
        return 0
    fi

    if [[ -z "${SLURM_JOB_ID:-}" ]] || ! command -v squeue >/dev/null 2>&1; then
        echo "[WallClock] no Slurm job / no squeue — running without a wall-clock budget."
        return 0
    fi

    # `%all`-free, field-addressed output: `-O EndTime` is stable across the
    # versions here, while `--format=%e` is not. `date -d` parses Slurm's
    # ISO-ish stamp; `UNLIMITED`/`N/A` fail it, which is the correct answer.
    local end_time deadline
    # `|| true`: the launcher runs under `set -euo pipefail`, where a busy
    # controller returning nonzero propagates through the pipe into this
    # assignment and kills the whole 120h job before it trains a step. Not
    # having a deadline is a recoverable state; dying at startup is not.
    end_time="$(squeue -h -j "${SLURM_JOB_ID}" -O EndTime 2>/dev/null | tr -d ' ' || true)"
    if [[ -z "${end_time}" || "${end_time}" == "UNLIMITED" || "${end_time}" == "N/A" ]]; then
        echo "[WallClock] Slurm reports no end time (${end_time:-empty}) — no budget."
        return 0
    fi

    deadline="$(date -d "${end_time}" +%s 2>/dev/null || true)"
    if [[ -z "${deadline}" ]]; then
        echo "[WallClock] could not parse EndTime='${end_time}' — no budget." >&2
        return 0
    fi

    export SPECTRAMR_WALL_CLOCK_DEADLINE="${deadline}"
    echo "[WallClock] deadline from squeue EndTime='${end_time}': ${deadline}" \
         "($(( (deadline - $(date +%s)) / 60 )) min from now)"
}
