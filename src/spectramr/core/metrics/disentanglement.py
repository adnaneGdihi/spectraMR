"""Does a content/style factorisation separate what it claims to separate?

An MSE-graded disentangled arm cannot show that its codes factorise: a decoder that ignores
the style code reconstructs just as well. Two readings do, over a validation pass that
carries a contrast label per row:

* **Linear probes.** A closed-form ridge classifier predicts the contrast label from the
  style code (it should succeed) and from the content code (it should sit at chance). The
  content probe reported as accuracy minus chance is the *leak*.
* **Swap distances.** Decode image *i*'s content with image *j*'s style, render it, and
  encode the result again: the content should come back unchanged and the style should be
  *j*'s. Both are cosine distances, 0 when the factorisation holds.

These are not registry metrics: the registry contract is ``(pred, target)`` images, and
neither reading can be formed from a pair of images. The strategy owns a
:class:`DisentanglementAccumulator` across the validation pass and reports the summary from
``finalize_validation``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _train_mask(n: int, groups: Sequence[str] | None) -> torch.Tensor:
    """Rows that train: the first half of the subjects, else the first half of the rows."""
    if groups is None:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[: n // 2] = True
        return mask
    if len(groups) != n:
        raise ValueError(f"{len(groups)} group labels for {n} rows")
    order = list(dict.fromkeys(groups))  # subjects in order of first appearance
    if len(order) < 2:
        raise ValueError(f"a subject-disjoint probe needs at least 2 subjects, got {order}")
    train_subjects = set(order[: len(order) - len(order) // 2])
    return torch.tensor([g in train_subjects for g in groups], dtype=torch.bool)


def linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    ridge: float = 1.0,
    groups: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Held-out accuracy of a one-vs-rest ridge classifier, and the chance level.

    With ``groups`` (a subject id per row) the first half of the subjects trains and the
    rest tests, so no subject's slices sit on both sides: neighbouring slices share a
    volume, and a probe that has seen a volume recognises it and reads its label back,
    which is a fingerprint and not a contrast. Without ``groups`` the first half of the
    rows trains, which keeps neighbouring slices together only if the order is by volume.

    Args:
        features: ``(N, D)`` codes, one row per validation sample.
        labels: ``(N,)`` integer contrast labels.
        ridge: Ridge penalty on the standardised features.
        groups: Optional subject id per row.

    Returns:
        ``(accuracy, chance)`` on the held-out rows; ``chance`` is the share of the most
        frequent label there, which a constant predictor reaches.

    Raises:
        ValueError: with fewer than 4 rows, fewer than 2 labels, fewer than 2 subjects, or
            a training side that holds a single label (no classifier can be fitted).
    """
    n = labels.numel()
    if n < 4:
        raise ValueError(f"a linear probe needs at least 4 rows, got {n}")
    if features.shape[0] != n:
        raise ValueError(f"{features.shape[0]} feature rows for {n} labels")
    classes = torch.unique(labels)
    if classes.numel() < 2:
        raise ValueError(f"a linear probe needs at least 2 contrast labels, got {classes.tolist()}")
    train = _train_mask(n, groups)
    x = features.detach().to(torch.float64)
    train_x, test_x = x[train], x[~train]
    train_y, test_y = labels[train], labels[~train]
    seen = torch.unique(train_y)
    if seen.numel() < 2:
        raise ValueError(
            f"the training side holds one contrast label ({seen.tolist()}); the validation "
            "order cannot support a held-out probe"
        )
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-8)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    ones_train = torch.ones(train_x.shape[0], 1, dtype=x.dtype)
    ones_test = torch.ones(test_x.shape[0], 1, dtype=x.dtype)
    a = torch.cat([train_x, ones_train], dim=1)
    onehot = (train_y.unsqueeze(1) == seen.unsqueeze(0)).to(x.dtype)
    penalty = ridge * torch.eye(a.shape[1], dtype=x.dtype)
    penalty[-1, -1] = 0.0  # the bias is not shrunk
    weights = torch.linalg.solve(a.T @ a + penalty, a.T @ onehot)
    predicted = seen[(torch.cat([test_x, ones_test], dim=1) @ weights).argmax(dim=1)]
    accuracy = float((predicted == test_y).to(torch.float64).mean())
    chance = float(torch.bincount(test_y.long()).max()) / test_y.numel()
    return accuracy, chance


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean ``1 - cos`` between matching rows of ``a`` and ``b``, flattened per row."""
    if a.shape != b.shape:
        raise ValueError(
            f"cosine_distance needs equal shapes, got {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=1, eps=1e-8)
    return float((1.0 - cos).mean())


def relative_error(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Mean per-row ``||estimate - reference|| / ||reference||``, flattened per row.

    0 when the code came back unchanged and about 1 when it was lost (a zero estimate).
    Unlike a cosine it sees scale, which is all a one-dimensional style code has.
    """
    if estimate.shape != reference.shape:
        raise ValueError(
            f"relative_error needs equal shapes, got {tuple(estimate.shape)} vs {tuple(reference.shape)}"
        )
    num = (estimate - reference).flatten(1).norm(dim=1)
    den = reference.flatten(1).norm(dim=1).clamp_min(1e-8)
    return float((num / den).mean())


class DisentanglementAccumulator:
    """Rows of one validation pass, kept for the probes and for the swap test.

    Args:
        per_label_swap_rows: How many rows per contrast label are kept whole (content map,
            style code, image) for the swap test; the probes see every row.
    """

    def __init__(self, per_label_swap_rows: int = 2) -> None:
        self._per_label = per_label_swap_rows
        self._content: list[torch.Tensor] = []
        self._style: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._groups: list[str] = []
        self._batches_with_groups = 0
        self._kept: dict[int, list[Any]] = {}

    def add(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        labels: torch.Tensor,
        groups: Sequence[str] | None = None,
    ) -> None:
        """Record a batch: ``content`` ``(B, C, H, W)``, ``style`` ``(B, ...)``, ``labels`` ``(B,)``.

        ``groups`` is the subject id of each row. Either every batch of a pass carries it
        or none does: a pass that mixed the two could not be split consistently.
        """
        if not content.shape[0] == style.shape[0] == labels.numel():
            raise ValueError(
                f"{content.shape[0]} content rows, {style.shape[0]} style rows, "
                f"{labels.numel()} labels"
            )
        if groups is not None and len(groups) != labels.numel():
            raise ValueError(f"{len(groups)} group labels for {labels.numel()} rows")
        if (groups is None) != (self._batches_with_groups == 0) and self._labels:
            raise ValueError("subject ids must accompany every batch of a pass, or none")
        # Mean and spread per channel: a content code can carry the contrast in its spatial
        # structure with every channel mean equal, and means alone would read that at chance.
        flat = content.detach().flatten(2)
        self._content.append(
            torch.cat([flat.mean(dim=2), flat.std(dim=2, unbiased=False)], dim=1).cpu()
        )
        self._style.append(style.detach().flatten(1).cpu())
        self._labels.append(labels.detach().view(-1).cpu())
        if groups is not None:
            self._groups.extend(str(g) for g in groups)
            self._batches_with_groups += 1

    def wants(self, label: int) -> bool:
        """Whether another whole row of ``label`` is still wanted for the swap test."""
        return len(self._kept.get(label, ())) < self._per_label

    def keep(self, label: int, row: Any) -> None:
        self._kept.setdefault(label, []).append(row)

    @property
    def n_rows(self) -> int:
        return sum(t.numel() for t in self._labels)

    def probes(self, include_style: bool = True) -> dict[str, float]:
        """``val_style_contrast_acc`` (high is good) and ``val_content_contrast_leak`` (low).

        ``include_style=False`` for a model whose render takes no style code (a declared
        protocol owns the contrast): a style reading would then describe the encoder alone.
        """
        if not self._labels:
            return {}
        labels = torch.cat(self._labels)
        groups = self._groups or None
        content_acc, content_chance = linear_probe(torch.cat(self._content), labels, groups=groups)
        out = {
            "val_content_contrast_leak": content_acc - content_chance,
            # Which split produced the numbers: 1 = subject-disjoint, 0 = row halves.
            "val_probe_split_by_subject": float(groups is not None),
        }
        if include_style:
            style_acc, style_chance = linear_probe(torch.cat(self._style), labels, groups=groups)
            out["val_style_contrast_acc"] = style_acc
            out["val_style_contrast_chance"] = style_chance
        return out

    def swap_pairs(self, max_pairs: int) -> list[tuple[Any, Any]]:
        """Ordered ``(content_source, style_source)`` rows of different labels, at most ``max_pairs``."""
        # Round-robin over the ordered contrast pairs, so a cap leaves every pair of contrasts
        # represented; truncating the sorted list kept only the first labels as content.
        labels = sorted(self._kept)
        groups = [
            [(ri, rj) for ri in self._kept[li] for rj in self._kept[lj]]
            for li in labels
            for lj in labels
            if li != lj
        ]
        pairs: list[tuple[Any, Any]] = []
        depth = 0
        while len(pairs) < max_pairs and any(depth < len(g) for g in groups):
            for g in groups:
                if depth < len(g) and len(pairs) < max_pairs:
                    pairs.append(g[depth])
            depth += 1
        return pairs
