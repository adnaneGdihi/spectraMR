"""Content hash of a cortical mesh."""

import hashlib

import numpy as np

from spectramr.infrastructure.surfaces.cortical_mesh import CorticalMesh


def test_mesh_hash_is_a_non_security_sha1(monkeypatch):
    real = hashlib.sha1
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("usedforsecurity"))
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha1", spy)
    vertices = np.eye(3)
    faces = np.array([[0, 1, 2]])
    digest = CorticalMesh(vertices=vertices, faces=faces).hash()
    assert seen == [False]
    assert len(digest) == 40
