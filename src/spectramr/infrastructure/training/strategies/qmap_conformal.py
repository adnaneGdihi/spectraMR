"""Subject-level split-conformal intervals for a held-out contrast's synthesis.

For a held-out contrast, the prediction is ``g_hat * render`` with the gain predicted from
the available contrasts only (``qmap_gauge_prior.prior_predicted_gains``), so the interval
never sees the label it covers. Subjects, not voxels, are the exchangeable unit: each
subject's score is the (1 - beta) quantile of its absolute residual over its signal voxels,
pooled over its slices, and the threshold is the ceil((n+1)(1-alpha))-th smallest score of
the calibration subjects. The guarantee reads: for a new subject, with probability at
least 1 - alpha, at least 1 - beta of its signal voxels lie within +-q of the prediction.

Validation subjects also drive early stopping, so coverage reported during training is a
diagnostic. The claim is the post-training run on held-out test subjects.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch

#: Residual histogram in noise units: 0.25-unit bins to 256, then one overflow bin.
_BIN = 0.25
_N_BINS = 1024


def subject_half(subject: str, split: int) -> int:
    """0 = calibration, 1 = evaluation; stable across processes and runs."""
    return hashlib.sha1(f"{split}:{subject}".encode(), usedforsecurity=False).digest()[0] & 1


class SubjectResiduals:
    """Per (subject, contrast) histogram of absolute residuals on signal voxels."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, int], torch.Tensor] = {}

    def add(
        self, subjects: Sequence[str], k: int, abs_residual: torch.Tensor, mask: torch.Tensor
    ) -> None:
        # On the residual's device, with unmasked voxels sent to a discard bin: boolean
        # indexing ran the histogram on the CPU, and a weighted CUDA bincount is not
        # deterministic. A non-finite residual lands in the overflow bin, as unbounded.
        r = abs_residual.detach().float().nan_to_num(nan=math.inf)
        idx = (r / _BIN).floor().clamp(0, _N_BINS).long()
        idx = torch.where(mask.bool(), idx, _N_BINS + 1)
        for i, subject in enumerate(subjects):
            counts = torch.bincount(idx[i].flatten(), minlength=_N_BINS + 2)[: _N_BINS + 1]
            key = (str(subject), k)
            self._counts[key] = self._counts.get(key, 0) + counts

    @staticmethod
    def _quantile(counts: torch.Tensor, level: float) -> float:
        cum = counts.cpu().cumsum(0)  # once per subject, at the end of the pass
        b = int(torch.searchsorted(cum, level * float(cum[-1])))
        return math.inf if b >= _N_BINS else (b + 1) * _BIN

    def summary(
        self, *, alpha: float, beta: float, n_splits: int, names: Sequence[str]
    ) -> dict[str, float]:
        """Threshold q, evaluation coverage (and its spread over splits) per contrast."""
        out: dict[str, float] = {}
        for k, name in enumerate(names):
            scores = {
                s: self._quantile(c, 1 - beta) for (s, kk), c in self._counts.items() if kk == k
            }
            if not scores:
                continue
            qs, covers, n_cal = [], [], []
            for split in range(n_splits):
                cal = sorted(v for s, v in scores.items() if subject_half(s, split) == 0)
                ev = [v for s, v in scores.items() if subject_half(s, split) == 1]
                if not cal or not ev:
                    continue
                rank = math.ceil((len(cal) + 1) * (1 - alpha))
                q = math.inf if rank > len(cal) else cal[rank - 1]
                qs.append(q)
                covers.append(sum(v <= q for v in ev) / len(ev))
                n_cal.append(len(cal))
            if not qs:
                continue
            mean_cover = sum(covers) / len(covers)
            out[f"val_conformal_q_{name}"] = sorted(qs)[len(qs) // 2]
            out[f"val_conformal_cover_{name}"] = mean_cover
            out[f"val_conformal_cover_spread_{name}"] = (
                sum((c - mean_cover) ** 2 for c in covers) / len(covers)
            ) ** 0.5
            out[f"val_conformal_ncal_{name}"] = sum(n_cal) / len(n_cal)
        return out

    def reset(self) -> None:
        self._counts = {}
