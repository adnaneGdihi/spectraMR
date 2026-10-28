r"""Pattern-C acquisition FiLM on a contrast embedding (LM-20).

The latent ULF->HF generator conditions on ``contrast_embedder(contrast_idx)``.
This block modulates that embedding with the contrast's physical acquisition
parameters,

.. math::

   e' = (1 + \gamma(a_c))\, e + \beta(a_c), \qquad a_c = \mathrm{Enc}(TE_c, TR_c, TI_c, FA_c),

where ``Enc`` is :class:`AcquisitionEmbedding` (the encoder owner) and
:math:`\gamma, \beta` come from one zero-initialised linear head, so the block
starts as the identity.

The table holds one row per contrast, bound from
``data.multi_contrast.acquisition_params`` and ordered by ``contrast_map``. It is
a buffer, so a checkpoint carries the table it trained with. B0 is not encoded:
the per-contrast block declares one field strength for every row, and a constant
column carries no information.

Because the table is constant per contrast, the modulation is a function of the
contrast id. FiLM on versus off therefore tests the physics parameterisation of
the conditioning, not information the id lacks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from spectramr.models.blocks.acquisition_conditioning import AcquisitionEmbedding

__all__ = ["ACQUISITION_FILM_KEYS", "AcquisitionTableFiLM", "acquisition_table"]

#: The acquisition scalars encoded per contrast, in table-column order.
ACQUISITION_FILM_KEYS: tuple[str, ...] = ("TE", "TR", "TI", "FA")


def acquisition_table(
    contrast_map: Mapping[str, int],
    rows: Sequence[Mapping[str, object]],
    num_contrasts: int,
) -> torch.Tensor:
    """``[num_contrasts, 4]`` table of (TE, TR, TI, FA), row ``i`` = contrast id ``i``.

    Raises when the map is not exactly ``0..num_contrasts-1``, or when the rows'
    names do not match the map's contrasts one to one.
    """
    ids = sorted(contrast_map.values())
    if ids != list(range(num_contrasts)):
        raise ValueError(
            f"contrast_map ids {ids} must be exactly 0..{num_contrasts - 1} "
            f"(num_contrasts={num_contrasts}) for an acquisition table."
        )
    by_name: dict[str, Mapping[str, object]] = {}
    for row in rows:
        name = str(row["name"])
        if name in by_name:
            raise ValueError(f"acquisition_params lists {name!r} twice.")
        by_name[name] = row
    missing = sorted(set(contrast_map) - set(by_name))
    extra = sorted(set(by_name) - set(contrast_map))
    if missing or extra:
        raise ValueError(
            "acquisition_params must name exactly the contrast_map contrasts; "
            f"missing {missing}, not in contrast_map {extra}."
        )
    table = torch.zeros(num_contrasts, len(ACQUISITION_FILM_KEYS))
    for name, idx in contrast_map.items():
        row = by_name[name]
        table[idx] = torch.tensor([float(row[k]) for k in ACQUISITION_FILM_KEYS])  # type: ignore[arg-type]
    return table


class AcquisitionTableFiLM(nn.Module):
    """FiLM the contrast embedding by its contrast's acquisition parameters."""

    def __init__(self, num_contrasts: int, embed_dim: int) -> None:
        super().__init__()
        self.num_contrasts = int(num_contrasts)
        self.encoder = AcquisitionEmbedding(
            embed_dim=embed_dim, n_contrasts=0, param_keys=ACQUISITION_FILM_KEYS
        )
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * embed_dim)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.register_buffer("table", torch.zeros(num_contrasts, len(ACQUISITION_FILM_KEYS)))
        self.register_buffer("table_bound", torch.zeros((), dtype=torch.bool))
        self._bound = False
        self.register_load_state_dict_post_hook(_sync_bound)

    def bind(self, contrast_map: Mapping[str, int], rows: Sequence[Mapping[str, object]]) -> None:
        """Bind the per-contrast table (once, at strategy setup)."""
        table = acquisition_table(contrast_map, rows, self.num_contrasts)
        self.table.copy_(table.to(self.table.device))
        self.table_bound.fill_(True)
        self._bound = True

    def forward(self, embedding: torch.Tensor, contrast_idx: torch.Tensor) -> torch.Tensor:
        if not self._bound:
            raise RuntimeError(
                "acquisition_film is on but no acquisition table was bound: the training "
                "strategy binds data.multi_contrast.acquisition_params at setup, and a "
                "checkpoint carries it. Neither happened."
            )
        params = self.table[contrast_idx.long()]
        enc = self.encoder({k: params[:, i] for i, k in enumerate(ACQUISITION_FILM_KEYS)})
        gamma, beta = self.to_gamma_beta(enc).chunk(2, dim=-1)
        return embedding * (1.0 + gamma) + beta


def _sync_bound(module: nn.Module, incompatible_keys: object) -> None:
    """After ``load_state_dict``, mirror the loaded flag (one host read, at load)."""
    module._bound = bool(module.table_bound)
