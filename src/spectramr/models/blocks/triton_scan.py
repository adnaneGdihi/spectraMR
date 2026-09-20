"""Accelerated selective-scan kernels for ``ContinuousSFCMamba``.

Three execution backends with identical numerics:

* ``python``  — the canonical reference loop. Pure PyTorch, runs on
  any device, used by all unit tests as the ground-truth.
* ``compile`` — wraps the Python loop in :func:`torch.compile`. JIT-
  compiled to a single fused graph; drops per-iteration Python
  overhead. Works on CPU and GPU. ~2–3× speedup over Python in
  practice.
* ``triton``  — explicit Triton kernel; one program per ``(batch,
  state)`` pair, sequential walk along the token axis. Requires CUDA
  and ``triton`` installed. Fastest of the three but only validatable
  on GPU.

A fourth **opt-in** backend, ``mamba_ssm``, routes the scan through the
official ``mamba_ssm.selective_scan_fn`` CUDA kernel. It is NOT
bit-identical to the three above: the kernel forms the decay as
``exp(delta * mean_c A_diag)`` (mean-before-exp) whereas the reference
backends use ``mean_c exp(delta * A_diag)`` (mean-after-exp), which the
kernel cannot express (it takes a constant ``A``, not a precomputed
per-token decay). Because of this it is **never chosen by ``auto``**
(reproducibility) and must be requested explicitly.

The wrapper :func:`selective_scan` picks the backend at call time.
``backend='auto'`` selects ``triton`` when CUDA + triton are
available, ``compile`` on CUDA without triton, ``python`` otherwise
(it never auto-selects ``mamba_ssm``).

The diagonal SSM recurrence the kernels implement is:

.. math::

    \\bar A_t = (\\exp(\\Delta_t \\cdot A_{diag}))\\,.mean(\\text{channel axis})
    \\quad\\in\\mathbb{R}^{S}

    h_t = h_{t-1} \\odot \\bar A_t + \\Delta_t \\cdot B_t
    \\quad\\in\\mathbb{R}^{B \\times S}

This matches the simplified diagonal-SSM contraction used by
``ContinuousSFCMamba``; deviations from the canonical Mamba formula
(per-channel state separation) are deliberate and consistent across
all three backends.
"""

from __future__ import annotations

from typing import Literal

import torch

__all__ = ["selective_scan", "triton_available"]


# Cache the (Triton + CUDA) availability check — cheap but called per
# constructor.
_TRITON_AVAILABLE: bool | None = None


def triton_available() -> bool:
    """Whether the Triton backend is usable in this process.

    Delegates to :mod:`spectramr.core.compile_capability`, which owns the
    question. This module held the original probe and the compile path needs the
    same answer; two copies would disagree the moment one grew a condition
    (non-negotiable 17). The memo stays here because this is the hot caller —
    it is consulted once per ``ContinuousSFCMamba`` constructor.
    """
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        from spectramr.core.compile_capability import (
            triton_available as _probe,
        )

        _TRITON_AVAILABLE = _probe()
    return _TRITON_AVAILABLE


_MAMBA_SSM_AVAILABLE: bool | None = None


def mamba_ssm_available() -> bool:
    """Whether the official ``mamba_ssm`` selective-scan kernel is importable."""
    global _MAMBA_SSM_AVAILABLE
    if _MAMBA_SSM_AVAILABLE is None:
        import importlib.util

        _MAMBA_SSM_AVAILABLE = importlib.util.find_spec("mamba_ssm") is not None
    return _MAMBA_SSM_AVAILABLE


# ── Reference (Python loop) implementation ───────────────────────────


def _python_scan(
    A_diag: torch.Tensor,  # (C, S)
    B_seq: torch.Tensor,  # (B, T, S)
    delta: torch.Tensor,  # (T,)
) -> torch.Tensor:  # returns h_seq: (B, T, S)
    """Reference per-token loop. Used as the ground truth in tests."""
    B, T, S = B_seq.shape
    h = B_seq.new_zeros(B, S)
    out = B_seq.new_empty(B, T, S)
    for t in range(T):
        # ``A_diag.mean(dim=0)`` averages across channels — matches the
        # contraction in ContinuousSFCMamba.forward (consistent with
        # the diagonal-SSM simplification used through this module).
        A_t = torch.exp(delta[t] * A_diag).mean(dim=0)  # (S,)
        B_t = B_seq[:, t] * delta[t]  # (B, S)
        h = h * A_t + B_t
        out[:, t] = h
    return out


# ── torch.compile fast path ──────────────────────────────────────────


_COMPILED_SCAN: callable | None = None


def _compile_scan(*args: torch.Tensor) -> torch.Tensor:
    """``torch.compile``-wrapped Python scan."""
    global _COMPILED_SCAN
    if _COMPILED_SCAN is None:
        # ``torch.compile`` is lazy; first call traces.
        _COMPILED_SCAN = torch.compile(_python_scan, dynamic=True)
    return _COMPILED_SCAN(*args)


# ── Triton kernel ────────────────────────────────────────────────────


def _triton_scan(
    A_diag: torch.Tensor,  # (C, S)  CUDA, fp32
    B_seq: torch.Tensor,  # (B, T, S)
    delta: torch.Tensor,  # (T,)
) -> torch.Tensor:
    """Triton kernel — one program per (batch, state) pair, sequential
    along T.

    Not a true parallel-scan implementation; the operator paper's
    Phase-3 plan reserves the Hillis-Steele variant. This kernel
    instead removes Python-loop overhead by keeping the recurrence
    on-device and avoiding kernel launches per token.

    Numerics are bit-exact-equivalent to ``_python_scan`` modulo
    floating-point reduction order in the ``A_diag.mean()``.
    """
    import triton
    import triton.language as tl

    B_, T_, S_ = B_seq.shape
    C_ = A_diag.shape[0]
    out = torch.empty_like(B_seq)

    # Pre-compute the per-token A across all states so the kernel
    # only does the recurrence (not the exponential / mean). This
    # also matches what `_python_scan` does — the mean is the same
    # across batches.
    # A_per_token: (T, S) with A_per_token[t, s] =
    #   mean_c exp(delta[t] * A_diag[c, s])
    A_per_token = (delta[:, None, None] * A_diag[None, :, :]).exp().mean(dim=1)
    # delta_seq broadcastable (T,)
    delta_dev = delta.contiguous()

    @triton.jit
    def _scan_kernel(
        A_ptr,
        B_ptr,
        dt_ptr,
        out_ptr,
        B_dim,
        T_dim,
        S_dim,
        stride_B_b,
        stride_B_t,
        stride_B_s,
        stride_A_t,
        stride_A_s,
        stride_O_b,
        stride_O_t,
        stride_O_s,
        BLOCK_S: tl.constexpr,
    ):
        # Program id maps to (batch, state-block).
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offsets < S_dim

        h = tl.zeros([BLOCK_S], dtype=tl.float32)
        for t in range(T_dim):
            a_t = tl.load(A_ptr + t * stride_A_t + s_offsets * stride_A_s, mask=s_mask, other=0.0)
            b_t = tl.load(
                B_ptr + pid_b * stride_B_b + t * stride_B_t + s_offsets * stride_B_s,
                mask=s_mask,
                other=0.0,
            )
            dt_t = tl.load(dt_ptr + t)
            h = h * a_t + dt_t * b_t
            tl.store(
                out_ptr + pid_b * stride_O_b + t * stride_O_t + s_offsets * stride_O_s,
                h,
                mask=s_mask,
            )

    BLOCK_S = 16 if S_ <= 16 else 32
    grid = (B_, (S_ + BLOCK_S - 1) // BLOCK_S)
    _scan_kernel[grid](
        A_per_token.contiguous(),
        B_seq.contiguous(),
        delta_dev,
        out,
        B_,
        T_,
        S_,
        B_seq.stride(0),
        B_seq.stride(1),
        B_seq.stride(2),
        A_per_token.stride(0),
        A_per_token.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_S=BLOCK_S,
    )
    return out


# ── Official mamba_ssm kernel ────────────────────────────────────────


def _mamba_ssm_scan(
    A_diag: torch.Tensor,  # (C, S)
    B_seq: torch.Tensor,  # (B, T, S)
    delta: torch.Tensor,  # (T,)
) -> torch.Tensor:  # (B, T, S)
    """Diagonal scan via the official ``mamba_ssm.selective_scan_fn`` kernel.

    Maps the shared-state diagonal recurrence onto ``selective_scan_fn`` by
    treating each of the ``S`` state channels as a 1-state "dim" (``dstate=1``).
    This ``dstate=1`` layout is required because ``selective_scan_fn`` returns
    ``C·h`` (it collapses the state axis), so the only way to recover the full
    hidden vector ``h`` is to put each state component on its own ``dim``
    channel with ``C=1``. The axis-remap is unit-tested on CPU against a
    faithful reference (``test_mamba_ssm_mapping_matches_reference_cpu``); the
    real CUDA kernel (incl. ``dstate=1`` support) is checked by the
    GPU-gated parity test. With ``u = B_seq``, ``B = C = 1`` and
    ``A = mean_c A_diag`` the kernel computes

    .. math::

        h_t = \\exp(\\Delta_t \\cdot \\bar A)\\, h_{t-1} + \\Delta_t \\cdot B_t

    **Numerics deviate from the python/triton/compile backends.**
    ``selective_scan_fn`` forms the decay as ``exp(delta * A)`` for a *constant*
    ``A``, so this uses the **mean-before-exp** decay
    ``exp(delta_t * mean_c A_diag)`` whereas the reference backends use
    **mean-after-exp** ``mean_c exp(delta_t * A_diag)``. The two coincide only
    for channel-uniform ``A`` (Jensen gap otherwise). This backend is therefore
    OPT-IN — ``auto`` never selects it — and is meant for training from scratch
    on the fast CUDA kernel, not for bit-swapping a checkpoint trained under a
    reference backend.
    """
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    B_, T_, S_ = B_seq.shape
    in_dtype = B_seq.dtype
    # dim = S channels, each carrying a 1-dim state.
    A_bar = A_diag.mean(dim=0).to(torch.float32).reshape(S_, 1).contiguous()  # (S, 1)
    u = B_seq.permute(0, 2, 1).contiguous().to(torch.float32)  # (B, S, T)
    dt = delta.to(torch.float32).reshape(1, 1, T_).expand(B_, S_, T_).contiguous()  # (B, S, T)
    ones = u.new_ones(B_, 1, T_)  # B = C = 1
    y = selective_scan_fn(
        u,
        dt,
        A_bar,
        ones,
        ones,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
    )  # (B, S, T)
    return y.permute(0, 2, 1).contiguous().to(in_dtype)  # (B, T, S)


# ── Dispatcher ───────────────────────────────────────────────────────


def selective_scan(
    A_diag: torch.Tensor,
    B_seq: torch.Tensor,
    delta: torch.Tensor,
    backend: Literal["auto", "python", "compile", "triton", "mamba_ssm"] = "auto",
) -> torch.Tensor:
    """Run the selective scan with the requested backend.

    Args:
        A_diag:  ``(C, S)`` per-channel state-matrix diagonal
                 (already negated; the caller passes ``-exp(A_log)``).
        B_seq:   ``(B, T, S)`` per-token input projections.
        delta:   ``(T,)`` per-token timesteps.
        backend: ``'python'``, ``'compile'``, ``'triton'`` or
                 ``'auto'`` (default). ``'auto'`` picks the fastest
                 backend that's actually available.

    Returns:
        ``(B, T, S)`` hidden-state sequence.
    """
    if backend == "auto":
        if A_diag.is_cuda and triton_available():
            backend = "triton"
        elif A_diag.is_cuda:
            backend = "compile"
        else:
            backend = "python"

    if backend == "python":
        return _python_scan(A_diag, B_seq, delta)
    if backend == "compile":
        return _compile_scan(A_diag, B_seq, delta)
    if backend == "triton":
        if not triton_available():
            raise RuntimeError(
                "Triton backend requested but unavailable in this process. "
                "Install 'triton' and use a CUDA-enabled PyTorch build, "
                "or pass backend='python'/'compile'/'auto'."
            )
        if not A_diag.is_cuda:
            raise RuntimeError(
                "Triton backend requires CUDA tensors but A_diag is on "
                f"{A_diag.device}. Move the model to GPU first or pass "
                "backend='python'/'compile'/'auto'."
            )
        return _triton_scan(A_diag, B_seq, delta)
    if backend == "mamba_ssm":
        # Opt-in only (``auto`` never reaches here): the kernel uses a
        # mean-before-exp decay that is NOT bit-identical to the reference
        # backends — see _mamba_ssm_scan.
        if not mamba_ssm_available():
            raise RuntimeError(
                "mamba_ssm backend requested but the mamba_ssm package is not "
                "importable. Install it (pip install -e '.[mamba]' "
                "--no-build-isolation), or pass backend="
                "'python'/'compile'/'triton'/'auto'."
            )
        if not A_diag.is_cuda:
            raise RuntimeError(
                "mamba_ssm backend requires CUDA tensors but A_diag is on "
                f"{A_diag.device}. Move the model to GPU first or pass "
                "backend='python'/'compile'/'auto'."
            )
        return _mamba_ssm_scan(A_diag, B_seq, delta)
    raise ValueError(
        f"Unknown selective_scan backend {backend!r}. "
        f"Choose from {{'auto', 'python', 'compile', 'triton', 'mamba_ssm'}}."
    )
