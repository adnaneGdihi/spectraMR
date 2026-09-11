r"""Matrix-free Magnus / Baker-Campbell-Hausdorff operator exponential.

This module is the single source of truth for the deterministic part of
the identified degradation operator in Proposal 1 (Lie-algebraic
effective-generator identification),

.. math::

    \exp\!\big(\Omega(s)\big),\qquad
    \Omega(s) = \sum_k s_k L_k
              + \tfrac12 \sum_{j<k} s_j s_k [L_j, L_k] + \cdots,

where each :math:`L_k` is a *matrix-free* generator (see
:mod:`spectramr.infrastructure.physics.degradation_generators`) exposed as
a :class:`LinearOperator`. The composite generator :math:`\Omega` is
assembled as a truncated Baker-Campbell-Hausdorff (BCH) series and its
action :math:`y = \exp(\Omega)x` is evaluated by an Arnoldi/Lanczos
Krylov projection — no dense ``N x N`` matrix is ever formed.

Design notes
------------
* **Matrix-free.** A :class:`LinearOperator` stores a ``matvec`` and its
  adjoint ``rmatvec``; composition, addition, scaling and commutators all
  produce new :class:`LinearOperator`\s without materialising matrices.
* **Differentiable.** ``expm_multiply`` is built from autograd-friendly
  torch primitives (the Arnoldi recurrence plus :func:`torch.matrix_exp`
  on the small Hessenberg matrix), so gradients of
  :math:`\exp(\Omega(s))x` w.r.t. the per-mode severities :math:`s`
  flow through autograd. No hand-written :math:`d\exp` is required; the
  right-trivialised differential is recovered automatically.
* **BCH folding.** The multi-generator BCH series truncated to order
  :math:`p\in\{1,2,3\}` is built by *folding* a two-operator truncated
  BCH across the generators, which reproduces
  :math:`\log\big(e^{X_1}\cdots e^{X_K}\big)` to order :math:`p`.

References
----------
W. Magnus, "On the exponential solution of differential equations for a
linear operator," *Commun. Pure Appl. Math.*, 7(4), 1954.

S. Blanes, F. Casas, J. A. Oteo, J. Ros, "The Magnus expansion and some
of its applications," *Phys. Rep.*, 470(5-6), 2009.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

__all__ = [
    "SUPPORTED_BCH_ORDERS",
    "LinearOperator",
    "assemble_omega",
    "bch_two_term",
    "commutator",
    "dense_matrix",
    "expm_multiply",
]

# BCH orders implemented by :func:`bch_two_term` / :func:`assemble_omega`.
# Kept in sync with the ``bch_order_supported`` Tier-1 audit check.
SUPPORTED_BCH_ORDERS: tuple[int, ...] = (1, 2, 3)

_Matvec = Callable[[torch.Tensor], torch.Tensor]


def _reduce_dims(x: torch.Tensor) -> tuple[int, ...]:
    """All dims except the leading batch dim (the per-sample inner-product axes)."""
    return tuple(range(1, x.dim()))


def _inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Batched Hermitian inner product :math:`\langle a, b\rangle = a^H b` → ``[B]``."""
    return (a.conj() * b).sum(dim=_reduce_dims(a))


def _norm(a: torch.Tensor) -> torch.Tensor:
    """Batched Euclidean norm → ``[B]`` (real)."""
    return _inner(a, a).real.clamp_min(0.0).sqrt()


class LinearOperator:
    r"""A matrix-free linear map with an explicit adjoint.

    The operator acts on a tensor ``v`` whose leading axis is the batch
    dimension; ``matvec`` and ``rmatvec`` must be batch-wise linear maps
    of identical shape. ``rmatvec`` implements the Hermitian adjoint
    :math:`L^H` (so that :math:`\langle L u, v\rangle = \langle u, L^H
    v\rangle` for the batched inner product :func:`_inner`).

    Operators compose with ``@`` (operator product), add with ``+``, and
    scale with :meth:`scaled` (the scale may be a Python scalar or a
    per-sample tensor broadcastable over ``v``).
    """

    def __init__(
        self,
        matvec: _Matvec,
        rmatvec: _Matvec | None = None,
        *,
        name: str = "",
    ) -> None:
        self._matvec = matvec
        self._rmatvec = rmatvec
        self.name = name

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self._matvec(v)

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        """Apply :math:`L v`."""
        return self._matvec(v)

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        r"""Apply the Hermitian adjoint :math:`L^H v`."""
        if self._rmatvec is None:
            raise NotImplementedError(f"LinearOperator {self.name!r} has no adjoint defined.")
        return self._rmatvec(v)

    @property
    def H(self) -> LinearOperator:
        """The adjoint operator :math:`L^H`."""
        return LinearOperator(self.adjoint, self._matvec, name=f"{self.name}^H")

    def __matmul__(self, other: LinearOperator) -> LinearOperator:
        """Operator product ``self @ other`` ⇒ ``v ↦ self(other(v))``."""
        return LinearOperator(
            lambda v: self._matvec(other._matvec(v)),
            lambda v: other.adjoint(self.adjoint(v)),
            name=f"({self.name}∘{other.name})",
        )

    def __add__(self, other: LinearOperator) -> LinearOperator:
        return LinearOperator(
            lambda v: self._matvec(v) + other._matvec(v),
            lambda v: self.adjoint(v) + other.adjoint(v),
            name=f"({self.name}+{other.name})",
        )

    def __sub__(self, other: LinearOperator) -> LinearOperator:
        return LinearOperator(
            lambda v: self._matvec(v) - other._matvec(v),
            lambda v: self.adjoint(v) - other.adjoint(v),
            name=f"({self.name}-{other.name})",
        )

    def scaled(self, c: torch.Tensor | float | complex) -> LinearOperator:
        r"""Scalar (or per-sample) scaling :math:`c\,L`.

        ``c`` may be a Python scalar or a tensor broadcastable over the
        operand (e.g. ``[B, 1, 1, 1]`` for a per-sample severity). The
        adjoint scales by :math:`\bar c`.
        """

        def _conj(x: torch.Tensor | float | complex):
            if isinstance(x, torch.Tensor):
                return x.conj()
            if isinstance(x, complex):
                return x.conjugate()
            return x

        cc = _conj(c)
        return LinearOperator(
            lambda v: c * self._matvec(v),
            lambda v: cc * self.adjoint(v),
            name=f"{c}·{self.name}",
        )


def identity_operator(name: str = "I") -> LinearOperator:
    """The identity operator."""
    return LinearOperator(lambda v: v, lambda v: v, name=name)


def commutator(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    r"""The Lie bracket :math:`[A, B] = AB - BA`.

    The adjoint identity :math:`[A,B]^H = [B^H, A^H]` is wired explicitly
    so commutators (and nested commutators) remain adjoint-correct.
    """
    return LinearOperator(
        lambda v: a(b(v)) - b(a(v)),
        lambda v: b.adjoint(a.adjoint(v)) - a.adjoint(b.adjoint(v)),
        name=f"[{a.name},{b.name}]",
    )


def bch_two_term(x: LinearOperator, y: LinearOperator, order: int) -> LinearOperator:
    r"""Two-operator Baker-Campbell-Hausdorff series truncated to ``order``.

    Returns the operator :math:`Z` with
    :math:`e^Z = e^X e^Y + \mathcal{O}(\,\cdot^{order+1})`:

    * order 1: :math:`X + Y`
    * order 2: :math:`X + Y + \tfrac12[X,Y]`
    * order 3: previous :math:`+ \tfrac1{12}\big([X,[X,Y]] + [Y,[Y,X]]\big)`
    """
    if order not in SUPPORTED_BCH_ORDERS:
        raise ValueError(
            f"bch_order must be one of {SUPPORTED_BCH_ORDERS}, got {order!r}. "
            "Higher orders are not implemented."
        )
    z = x + y
    if order >= 2:
        z = z + commutator(x, y).scaled(0.5)
    if order >= 3:
        xxy = commutator(x, commutator(x, y))
        yyx = commutator(y, commutator(y, x))
        z = z + (xxy + yyx).scaled(1.0 / 12.0)
    return z


def assemble_omega(
    generators: Sequence[LinearOperator],
    severities: torch.Tensor,
    order: int = 2,
) -> LinearOperator:
    r"""Assemble the composite generator :math:`\Omega(s)` as a matrix-free operator.

    The composite is the truncated BCH series of the ordered product
    :math:`e^{s_1 L_1}\cdots e^{s_K L_K}`, built by folding
    :func:`bch_two_term` across the (severity-scaled) generators. The
    resulting operator is exact at order 1 (the plain sum), captures the
    leading commutator interaction at order 2, and the dominant nested
    commutators at order 3.

    Args:
        generators: The :math:`K` matrix-free generators :math:`L_k`.
        severities: Per-mode severities. Shape ``[K]`` (shared across the
            batch) or ``[B, K]`` (per-sample). Per-sample severities are
            broadcast over the operand's non-batch axes when scaling.
        order: BCH truncation order in :data:`SUPPORTED_BCH_ORDERS`.

    Returns:
        A :class:`LinearOperator` implementing :math:`v \mapsto \Omega v`.
    """
    if len(generators) == 0:
        raise ValueError("assemble_omega requires at least one generator.")
    if order not in SUPPORTED_BCH_ORDERS:
        raise ValueError(f"bch_order must be one of {SUPPORTED_BCH_ORDERS}, got {order!r}.")

    severities = torch.as_tensor(severities)
    per_sample = severities.dim() == 2

    def _weight(k: int) -> torch.Tensor | float:
        if per_sample:
            # [B] → [B, 1, ..., 1] broadcast handled at apply time via a
            # 1-D tensor that the operand multiplies against; we expand to
            # [B, 1] here and let scaling broadcast through reshape below.
            return severities[:, k]
        return severities[k]

    def _scale_for(weight: torch.Tensor | float, like_rank: int):
        # Build a closure producing a broadcastable scale for a given
        # operand rank when ``weight`` is per-sample.
        if isinstance(weight, torch.Tensor) and weight.dim() == 1:
            view = (-1,) + (1,) * (like_rank - 1)
            return weight.reshape(view)
        return weight

    scaled_gens: list[LinearOperator] = []
    for k, gen in enumerate(generators):
        w = _weight(k)
        if isinstance(w, torch.Tensor) and w.dim() == 1:
            # Wrap so the per-sample weight reshapes to the operand rank.
            def _mk(g: LinearOperator, weight: torch.Tensor) -> LinearOperator:
                return LinearOperator(
                    lambda v: _scale_for(weight, v.dim()) * g(v),
                    lambda v: _scale_for(weight.conj(), v.dim()) * g.adjoint(v),
                    name=f"s·{g.name}",
                )

            scaled_gens.append(_mk(gen, w))
        else:
            scaled_gens.append(gen.scaled(w))

    omega = scaled_gens[0]
    for gen in scaled_gens[1:]:
        omega = bch_two_term(omega, gen, order)
    return omega


def expm_multiply(
    op: LinearOperator,
    x: torch.Tensor,
    *,
    krylov_dim: int = 30,
    hermitian: bool = False,
    scaling_squaring: bool = True,
    norm_threshold: float = 1.0,
) -> torch.Tensor:
    r"""Matrix-free action :math:`y = \exp(\mathrm{op})\,x` via Krylov projection.

    Uses an Arnoldi recurrence (a Lanczos recurrence is mathematically a
    special case for Hermitian ``op``; the symmetric Hessenberg falls out
    automatically) to build an orthonormal Krylov basis, then exponentiates
    the small Hessenberg matrix with :func:`torch.matrix_exp`. Optional
    scaling-and-squaring bounds the spectral radius for stiff generators.

    The whole routine is autograd-differentiable, so gradients of ``y``
    w.r.t. any tensor feeding ``op`` (e.g. the severities inside
    :func:`assemble_omega`) propagate without a hand-coded ``dexp``.

    Args:
        op: The generator :math:`\Omega` as a :class:`LinearOperator`.
        x: Operand ``[B, ...]`` (real or complex). Complex is recommended
            for MRI image data.
        krylov_dim: Maximum Krylov subspace size ``m``. Capped at the
            ambient dimension ``N`` (product of non-batch axes); at
            ``m == N`` the projection is exact up to round-off.
        hermitian: Hint that ``op`` is Hermitian. Unused for correctness
            (Arnoldi handles both) but documents intent.
        scaling_squaring: Enable norm-based scaling and squaring.
        norm_threshold: Target spectral-radius proxy per squaring level.

    Returns:
        ``y`` with the same shape/dtype as ``x``.
    """
    if x.dim() < 1:
        raise ValueError("expm_multiply expects a batched operand [B, ...].")
    n_ambient = 1
    for d in x.shape[1:]:
        n_ambient *= int(d)
    m = max(1, min(int(krylov_dim), n_ambient))

    # ── scaling-and-squaring level ────────────────────────────────────
    squarings = 0
    work_op = op
    if scaling_squaring:
        # One-shot power-iteration proxy for the spectral radius using the
        # operand itself (cheap, differentiation-safe via .detach()).
        with torch.no_grad():
            v = x.detach()
            nv = _norm(v).clamp_min(1e-12)
            v = v / nv.reshape((-1,) + (1,) * (x.dim() - 1))
            av = op(v)
            est = float(_norm(av).max().item())
        if est > norm_threshold:
            import math

            squarings = max(0, math.ceil(math.log2(est / norm_threshold)))
            # Reconstruction below applies exp(op/2^s) a total of 2^s times, so
            # the level is bounded (cap 8 → 256 applications). A tighter scaling
            # would only sharpen the per-step Krylov proxy, not the exact answer.
            squarings = min(squarings, 8)
            scale = 1.0 / float(2**squarings)
            work_op = op.scaled(scale)

    # exp(op) x = [exp(op / 2^s)]^(2^s) x — apply the scaled exponential 2^s
    # times (one initial application below, 2^s − 1 in the loop). The previous
    # `range(squarings)` under-applied it (only s + 1 times), so the squaring
    # path computed exp((s+1)/2^s · op) x for s ≥ 2.
    y = _expm_krylov(work_op, x, m)
    for _ in range(2**squarings - 1):
        y = _expm_krylov(work_op, y, m)
    return y


def _expm_krylov(op: LinearOperator, x: torch.Tensor, m: int) -> torch.Tensor:
    """One Arnoldi-projected ``exp(op) @ x`` with subspace size ``m``."""
    batch = x.shape[0]
    rank = x.dim()
    view = (batch,) + (1,) * (rank - 1)
    eps = 1e-12

    beta = _norm(x)  # [B]
    safe_beta = beta.reshape(view).clamp_min(eps)
    cdtype = x.dtype if x.is_complex() else torch.complex64

    basis: list[torch.Tensor] = [x / safe_beta]
    h = torch.zeros(batch, m, m, dtype=cdtype, device=x.device)

    for j in range(m):
        w = op(basis[j])
        for i in range(j + 1):
            hij = _inner(basis[i], w)  # [B]
            h[:, i, j] = hij.to(cdtype)
            w = w - hij.reshape(view) * basis[i]
        hnext = _norm(w)  # [B]
        if j + 1 < m:
            h[:, j + 1, j] = hnext.to(cdtype)
            basis.append(w / hnext.reshape(view).clamp_min(eps))

    exp_h = torch.matrix_exp(h)  # [B, m, m]
    coeffs = exp_h[:, :, 0]  # [B, m] — first column = exp(H) e_1

    v_stack = torch.stack(basis, dim=1)  # [B, m, ...]
    coeff_view = coeffs.reshape((batch, m) + (1,) * (rank - 1)).to(v_stack.dtype)
    combo = (coeff_view * v_stack).sum(dim=1)  # [B, ...]
    return beta.reshape(view).to(combo.dtype) * combo


def dense_matrix(
    op: LinearOperator,
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    r"""Materialise the ``N x N`` matrix of ``op`` for a single-sample ``shape``.

    Applies ``op`` to each standard basis vector. Intended for tests
    (dense ``torch.matrix_exp`` cross-checks, adjoint dot-product tests),
    never for the training hot path.

    Args:
        op: The operator. Applied with a batch dimension of 1.
        shape: The per-sample operand shape (without batch), e.g.
            ``(C, H, W)``.
        dtype: Complex dtype for the materialised matrix.
        device: Device for the basis vectors.

    Returns:
        Matrix ``M`` of shape ``[N, N]`` with ``N = prod(shape)`` such
        that ``M @ vec(v) == vec(op(v))`` (column-major over ``shape``).
    """
    n = 1
    for d in shape:
        n *= int(d)
    cols = []
    for i in range(n):
        e = torch.zeros(n, dtype=dtype, device=device)
        e[i] = 1.0
        col = op(e.reshape((1, *shape))).reshape(n)
        cols.append(col)
    return torch.stack(cols, dim=1)  # column i is op(e_i)
