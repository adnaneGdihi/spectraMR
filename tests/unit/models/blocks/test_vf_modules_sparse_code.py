"""MarkerDictionaryLearner._sparse_code: batched matching pursuit equals the per-patch loop."""

import torch
import torch.nn.functional as F

from spectramr.models.blocks.vf_modules import MarkerDictionaryLearner


def test_sparse_code_matches_per_patch_loop() -> None:
    torch.manual_seed(0)
    m = MarkerDictionaryLearner(patch_size=4, num_atoms=12, sparsity=3)
    patches = torch.randn(7, 16)
    d_norm = F.normalize(m.dictionary, dim=1)
    coeffs = torch.zeros(7, 12)
    residual = patches.clone()
    for _ in range(3):
        corr = residual @ d_norm.T
        best = corr.abs().argmax(dim=1)
        for n in range(7):
            k = int(best[n])
            coeffs[n, k] += corr[n, k]
            residual[n] -= corr[n, k] * d_norm[k]
    torch.testing.assert_close(m._sparse_code(patches), coeffs, rtol=1e-5, atol=1e-6)
