"""Where an M4Raw dataset gets its coil maps: the precomputed file, or the recipe on a miss.

The recipe, the key and the file format belong to
:mod:`~spectramr.data.datasets.m4raw_coil_maps`; this module only decides, per
load, whether to read a file or to estimate, and refuses the read when the file
was made by a different calibration or recipe than the consumer declares.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from spectramr.data.datasets.m4raw_coil_map_recipe import DEFAULT_RECIPE, check_recipe
from spectramr.data.datasets.m4raw_coil_maps import (
    _CALIBRATION_FIELDS,
    estimate_group_maps,
    load_coil_maps,
    map_path,
)

logger = logging.getLogger(__name__)

__all__ = ["VALID_MISSING_POLICIES", "CoilMapSource"]

#: What a consumer does when ``<root>/<key>.pt`` does not exist. ``raise`` is the
#: default because a silent recompute would hide an unpopulated cache for a
#: whole run; ``estimate`` runs :func:`estimate_group_maps` in the loader.
VALID_MISSING_POLICIES: frozenset[str] = frozenset({"raise", "estimate"})


class CoilMapSource:
    """Where one dataset gets its maps: ``<root>/<key>.pt``, or the recipe on a miss.

    Args:
        root: directory of map files.
        missing: one of :data:`VALID_MISSING_POLICIES`.
        calibration: the consumer's ESPIRiT settings, checked against every file
            and used by the ``estimate`` fallback.
        recipe: one of :data:`~spectramr.data.datasets.m4raw_coil_map_recipe.VALID_RECIPES`, checked against every file and
            used by the ``estimate`` fallback.
    """

    def __init__(
        self,
        root: str | Path,
        missing: str,
        calibration: dict[str, Any],
        recipe: str = DEFAULT_RECIPE,
    ) -> None:
        if missing not in VALID_MISSING_POLICIES:
            raise ValueError(
                f"coil_maps_missing={missing!r}; valid: {sorted(VALID_MISSING_POLICIES)}."
            )
        unknown = set(_CALIBRATION_FIELDS) - set(calibration)
        if unknown:
            raise ValueError(f"calibration lacks {sorted(unknown)}.")
        self.root = Path(root)
        self.missing = missing
        self.calibration = {k: calibration[k] for k in _CALIBRATION_FIELDS}
        self.recipe = check_recipe(recipe)
        self._reported_missing = False

    @classmethod
    def declared(
        cls,
        root: str | Path,
        missing: str,
        *,
        recipe: str = DEFAULT_RECIPE,
        acs_size: int | None = None,
    ) -> CoilMapSource:
        """The source an arm's ``data.coil_maps_*`` keys declare.

        The calibration is the per-scan ESPIRiT default with ``acs_size``
        replacing its ACS width when declared. The loader and ``predict`` both
        build through here, so a file written under another ACS width or recipe
        raises in both rather than only in the one that passed it (#2688).
        """
        from spectramr.infrastructure.physics.scan_coil_calibration import DEFAULT_CALIBRATION

        calibration = dict(DEFAULT_CALIBRATION)
        if acs_size is not None:
            if int(acs_size) < 1:
                raise ValueError(f"[M4Raw] coil_maps_acs_size must be >= 1, got {acs_size!r}.")
            calibration["acs_size"] = int(acs_size)
        return cls(root, missing, calibration, recipe)

    def refuse_absent_maps(self, file_ids: Iterable[str]) -> None:
        """Raise now when ``missing`` is ``raise`` and a group's map file is not on disk.

        ``maps_for`` raises the same error, but only when the first sample of the absent group
        is drawn, which can be hours into an epoch. The groups are known when the dataset is
        built, so the arm that declared a map root without its maps fails there instead.
        ``estimate`` never refuses: absence is what it was asked to handle.
        """
        if self.missing != "raise":
            return
        wanted = {map_path(self.root, fid) for fid in file_ids}
        absent = sorted(p for p in wanted if not p.is_file())
        if not absent:
            return
        shown = ", ".join(p.name for p in absent[:5]) + (", ..." if len(absent) > 5 else "")
        raise FileNotFoundError(
            f"{len(absent)} coil-map file(s) missing under {self.root} and "
            f"data.coil_maps_missing is 'raise' (e.g. {shown}). The maps are not estimated in the "
            "loader on this setting, so the run would fail at the first sample of an affected "
            "group. Run scripts/precompute_m4raw_coil_maps.py, or declare "
            "coil_maps_missing: estimate."
        )

    def maps_for(
        self,
        file_id: str,
        kspace_reps: Sequence[torch.Tensor],
        *,
        slice_index: int | None = None,
    ) -> tuple[torch.Tensor, str]:
        """``(maps, source)`` for the group ``file_id`` belongs to.

        ``kspace_reps`` are the loaded repetitions (the input first); their
        shape is what the file must describe, and they are the estimate's data.
        ``source`` is ``"file:<path>"`` or ``"estimated"``.
        """
        path = map_path(self.root, file_id)
        expect = tuple(kspace_reps[0].shape) if kspace_reps else ()
        if len(expect) == 3:
            expect = (1, *expect)
        if path.is_file():
            maps, _ = load_coil_maps(
                path,
                expect_shape=expect,
                calibration=self.calibration,
                slice_index=slice_index,
                recipe=self.recipe,
            )
            return maps, f"file:{path}"
        if self.missing == "raise":
            raise FileNotFoundError(
                f"no precomputed coil maps at {path} and data.coil_maps_missing is 'raise'. "
                "Run scripts/precompute_m4raw_coil_maps.py, or declare "
                "coil_maps_missing: estimate to estimate them in the loader."
            )
        if not self._reported_missing:
            self._reported_missing = True
            logger.warning(
                "[M4Raw] coil maps absent under %s (first miss: %s); estimating them in "
                "the loader with recipe %s, at 0.8-2.3 s per slice. Reported once per "
                "worker.",
                self.root,
                path.name,
                self.recipe,
            )
        maps, _ = estimate_group_maps(kspace_reps, self.calibration, self.recipe)
        return maps, "estimated"
