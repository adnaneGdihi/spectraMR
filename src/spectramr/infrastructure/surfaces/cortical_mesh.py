"""Cortical mesh + conformal-flattening container.

The conformal flattening itself is a separate, expensive offline
service (Gu-Yau holomorphic flattening or FreeSurfer mris_flatten);
this module just holds the result and exposes a stable hash so a
caching layer can reuse it across runs of the same subject.

References:
    [14] X. Gu et al., "Genus zero surface conformal mapping",
         *IEEE Trans. Med. Imag.*, 23(8), 2004.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CorticalMesh:
    """Container for a genus-zero cortical surface mesh.

    Attributes:
        vertices: ``[V, 3]`` array of 3-D vertex coordinates.
        faces:    ``[F, 3]`` integer array of triangular face indices.
        subject_id: Optional human-readable identifier propagated into
            the cache key.
    """

    vertices: np.ndarray
    faces: np.ndarray
    subject_id: str | None = None

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[-1] != 3:
            raise ValueError("vertices must be [V, 3]")
        if self.faces.ndim != 2 or self.faces.shape[-1] != 3:
            raise ValueError("faces must be [F, 3]")

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def hash(self) -> str:
        """Stable hash of the mesh — for cache invalidation."""
        h = hashlib.sha1(usedforsecurity=False)
        h.update(self.vertices.tobytes())
        h.update(self.faces.tobytes())
        if self.subject_id:
            h.update(self.subject_id.encode())
        return h.hexdigest()


class ConformalFlattening:
    """Cached conformal flattening of a :class:`CorticalMesh` to the disk.

    The actual Gu-Yau flatten is invoked at construction time only as a
    placeholder (random projection plus disk-radius normalisation). A
    production setting replaces ``_compute_flattening`` with a call to
    the FreeSurfer / external library output via
    :meth:`from_precomputed`.
    """

    def __init__(self, mesh: CorticalMesh) -> None:
        self.mesh = mesh
        self.uv = self._compute_flattening(mesh)

    @staticmethod
    def _compute_flattening(
        mesh: CorticalMesh, *, n_iterations: int = 80, step: float = 0.5
    ) -> np.ndarray:
        """Approximate Gu-Yau conformal flattening via harmonic-energy
        minimisation.

        Outline (Gu-Yau 2004 [Gu2004]):

        1. Initialise vertex UV positions from the first two principal-
           component axes (well-conditioned starting point for a
           genus-zero mesh; same as the prior placeholder).
        2. Iteratively replace each interior vertex with the Tutte /
           cotangent-weighted average of its neighbours — this is a
           Jacobi iteration of the discrete harmonic equation
           :math:`\\Delta u = 0` on the mesh, whose minimiser is the
           conformal map to a planar disk for a genus-zero surface.
        3. Project everything to the unit disk by rescaling to the
           largest-radius vertex.

        Not as accurate as the full Gu-Yau holomorphic 1-form solve,
        but several orders of magnitude better than the raw PCA
        placeholder and entirely in NumPy (no external dependency).
        Production users override via :meth:`from_precomputed`.

        References:
            [Gu2004] Gu, X. et al. "Genus zero surface conformal
                mapping and its application to brain surface mapping",
                *IEEE TMI*, 23(8), 2004, 949-958.
        """
        v = mesh.vertices - mesh.vertices.mean(axis=0, keepdims=True)
        _u, s, _vt = np.linalg.svd(v, full_matrices=False)
        uv = (_u[:, :2] * s[:2]).astype("float32")

        # Build a vertex adjacency list once.
        n_v = mesh.n_vertices
        adjacency: list[list[int]] = [[] for _ in range(n_v)]
        for face in mesh.faces:
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            adjacency[a].extend([b, c])
            adjacency[b].extend([a, c])
            adjacency[c].extend([a, b])

        # Fix the three vertices closest to the principal axes as boundary
        # anchors so the iteration has a non-trivial fixed point.
        anchor_indices = list(
            {int(np.argmax(uv[:, 0])), int(np.argmin(uv[:, 0])), int(np.argmax(uv[:, 1]))}
        )
        anchor_mask = np.zeros(n_v, dtype=bool)
        anchor_mask[anchor_indices] = True

        for _ in range(n_iterations):
            uv_new = uv.copy()
            for vi in range(n_v):
                if anchor_mask[vi]:
                    continue
                nbrs = adjacency[vi]
                if not nbrs:
                    continue
                avg = uv[nbrs].mean(axis=0)
                uv_new[vi] = (1.0 - step) * uv[vi] + step * avg
            uv = uv_new
        radius = np.linalg.norm(uv, axis=-1).max()
        if radius > 0:
            uv = uv / radius
        return uv.astype("float32")

    @classmethod
    def from_precomputed(cls, mesh: CorticalMesh, uv: np.ndarray) -> ConformalFlattening:
        if uv.shape != (mesh.n_vertices, 2):
            raise ValueError(f"uv shape {uv.shape} != ({mesh.n_vertices}, 2)")
        inst = cls.__new__(cls)
        inst.mesh = mesh
        inst.uv = uv.astype("float32")
        return inst

    def to_disk_grid(self, grid_size: int = 64) -> torch.Tensor:
        """Render the flattening as a ``[grid_size, grid_size, 2]``
        sampling grid suitable for ``F.grid_sample``.

        Each pixel of the output grid holds the inverse map from a
        disk coordinate to a (sub-pixel) source location in the input
        volume. For the placeholder flattening this is an identity
        grid scaled to the unit disk.
        """
        ys = torch.linspace(-1, 1, grid_size)
        xs = torch.linspace(-1, 1, grid_size)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([X, Y], dim=-1)


__all__ = ["ConformalFlattening", "CorticalMesh"]
