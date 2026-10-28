"""Independent domain-A / domain-B draws for unpaired training.

The cyclegan and cut strategies read ``batch["input"]`` from domain A and
``batch["target"]`` from domain B with no correspondence between them. This
dataset builds that batch from one paired manifest: domain A is every record's
primary (ULF) volume, domain B is every record's target (HF) volume, and each
sample draws one of each independently.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import torch
import torchio as tio
from loguru import logger

from spectramr.data.datasets.contrast_aware import ContrastAwareSubjectBuilder, ContrastConfig
from spectramr.data.datasets.field_strength_pair import stamp_field_strength_pair


class UnpairedDomainDataset(torch.utils.data.Dataset):
    """Yield ``tio.Subject(input=<domain A>, target=<domain B>)`` with no pairing.

    The epoch walks the domain-A pool once (index ``idx % len(A)``); the domain-B
    volume is drawn uniformly at random per sample. Each side goes through the
    transform separately because ULF and HF volumes do not share a grid.

    Draws come from ``random.Random`` seeded by the DataLoader worker (or by
    ``torch.initial_seed()`` in the main process), so they follow the run seed.
    The stream is not checkpointed: a resumed run does not replay the same
    A/B pairing sequence.
    """

    def __init__(
        self,
        index: list[dict],
        input_contrast: ContrastConfig,
        target_contrast: ContrastConfig,
        transform: tio.Transform | None = None,
        *,
        disjoint_subjects: bool = False,
        exclude_subjects: Iterable[str] = (),
        io_strategy: str = "nifti",
        max_skip_attempts: int = 10,
        expose_field_strength_pair: bool = False,
    ):
        self.transform = transform
        self.expose_field_strength_pair = expose_field_strength_pair
        self.max_skip_attempts = max_skip_attempts
        self.builder = ContrastAwareSubjectBuilder(
            input_contrast=input_contrast,
            target_contrast=target_contrast,
            io_strategy=io_strategy,
            verify_contrast=True,
            normalize=False,
        )
        self.a_pool, self.b_pool = self._build_pools(
            index, disjoint_subjects, {str(s) for s in exclude_subjects}
        )
        self._rng: random.Random | None = None
        logger.info(
            f"[UnpairedDomainDataset] A pool={len(self.a_pool)} records, "
            f"B pool={len(self.b_pool)} records, disjoint_subjects={disjoint_subjects}"
        )

    @staticmethod
    def _build_pools(
        index: list[dict], disjoint: bool, exclude: set[str]
    ) -> tuple[list[dict], list[dict]]:
        def subject(rec: dict) -> str:
            return str(rec.get("subject_id"))

        leaked = sorted({subject(r) for r in index if subject(r) in exclude})
        if leaked:
            raise ValueError(
                f"unpaired training index contains held-out subjects {leaked}; "
                "they must be excluded from both domain pools."
            )
        a_subjects: set[str] | None = None
        b_subjects: set[str] | None = None
        if disjoint:
            # Alternating halves of the sorted subject ids: deterministic and balanced.
            ordered = sorted({subject(r) for r in index})
            a_subjects, b_subjects = set(ordered[0::2]), set(ordered[1::2])
        a_pool = [
            r
            for r in index
            if r.get("primary_path") and (a_subjects is None or subject(r) in a_subjects)
        ]
        b_pool = [
            r
            for r in index
            if r.get("target_path") and (b_subjects is None or subject(r) in b_subjects)
        ]
        if not a_pool or not b_pool:
            raise ValueError(
                "unpaired training needs both domains: got "
                f"{len(a_pool)} domain-A (primary) and {len(b_pool)} domain-B (target) "
                "records in the train split."
            )
        return a_pool, b_pool

    def __len__(self) -> int:
        return len(self.a_pool)

    def _draw_b(self) -> dict:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            seed = info.seed if info is not None else torch.initial_seed()
            self._rng = random.Random(seed)
        return self.b_pool[self._rng.randrange(len(self.b_pool))]

    def __getitem__(self, idx: int) -> tio.Subject:
        a_idx = idx % len(self.a_pool)
        for _ in range(self.max_skip_attempts):
            a_rec = self.a_pool[a_idx]
            a_subject = self.builder.build({**a_rec, "target_path": None})
            # Single-side record: only the domain-B volume is read.
            b_rec = self._draw_b()
            b_subject = self.builder.build(
                {"target_path": b_rec["target_path"], "primary_path": None},
                swap=True,
            )
            if not (
                torch.isnan(a_subject["input"].data).any()
                or torch.isnan(b_subject["input"].data).any()
            ):
                if self.transform is not None:
                    a_subject = self.transform(a_subject)
                    b_subject = self.transform(b_subject)
                out = tio.Subject(input=a_subject["input"], target=b_subject["input"])
                if self.expose_field_strength_pair:
                    stamp_field_strength_pair(out, a_rec, b_rec)
                return out
            a_idx = (a_idx + 1) % len(self.a_pool)
        raise RuntimeError(f"no NaN-free unpaired draw after {self.max_skip_attempts} attempts")
