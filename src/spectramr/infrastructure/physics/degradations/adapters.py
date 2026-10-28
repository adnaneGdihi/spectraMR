"""Run a 2-D degradation kernel on a ``(B, C, D, H, W)`` volume by folding slices into a 4-D batch."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any

from torch import Tensor

from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.degradations.spec import ModeImpl

__all__ = ["Fold", "per_slice"]

Kernel2D = Callable[..., Tensor]


class Fold(StrEnum):
    """Where the slice axis goes when a volume is folded to 4-D.

    ``INDEPENDENT`` folds slices into the batch, so every slice gets its own draw.
    ``SHARED`` folds them into the channels, so a subject gets one realisation; a
    kernel that pools over C then pools over the whole volume.
    """

    INDEPENDENT = "batch"
    SHARED = "channel"


def _fold_batch(v: Tensor) -> Tensor:
    b, c, d, h, w = v.shape
    return v.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)


def _unfold_batch(y: Tensor, b: int, c: int, d: int) -> Tensor:
    return y.reshape(b, d, c, *y.shape[-2:]).permute(0, 2, 1, 3, 4)


def _fold_channel(v: Tensor) -> Tensor:
    b, c, d, h, w = v.shape
    return v.reshape(b, c * d, h, w)


def _unfold_channel(y: Tensor, b: int, c: int, d: int) -> Tensor:
    return y.reshape(b, c, d, *y.shape[-2:])


_FOLDS: dict[Fold, tuple[Callable[[Tensor], Tensor], Callable[..., Tensor]]] = {
    Fold.INDEPENDENT: (_fold_batch, _unfold_batch),
    Fold.SHARED: (_fold_channel, _unfold_channel),
}


def _fold_volume_kwarg(key: str, v: Tensor, fold: Fold, b: int, c: int, d: int) -> Tensor:
    if v.dim() != 5 or v.shape[0] != b or v.shape[1] not in (1, c) or v.shape[2] != d:
        raise ValueError(
            f"volume kwarg {key!r} must be (B, 1|C, D, H, W) = ({b}, 1|{c}, {d}, ...), "
            f"got {tuple(v.shape)}"
        )
    if fold is Fold.SHARED and v.shape[1] != c:
        # The channel fold is C-major, so a size-1 channel must be expanded before it.
        v = v.expand(b, c, *v.shape[2:])
    return _FOLDS[fold][0](v)


def per_slice(kernel2d: Kernel2D, fold: Fold, *, volume_kwargs: Iterable[str] = ()) -> ModeImpl:
    """Wrap ``kernel2d(x2d, theta, *, seed, **kwargs)`` as a 5-D mode implementation.

    Tensor kwargs named in ``volume_kwargs`` are folded the same way as ``x``; any
    other 5-D tensor kwarg raises, since passing it unfolded would misalign it.
    """
    if not isinstance(fold, Fold):
        raise ValueError(f"fold must be a Fold, got {fold!r}")
    named = frozenset(volume_kwargs)
    fold_x, unfold = _FOLDS[fold]

    def impl(
        x: Tensor, theta: float, seed: int, geometry: AcquisitionGeometry, **kwargs: Any
    ) -> Tensor:
        if x.dim() != 5:
            raise ValueError(f"per_slice expects a 5-D (B, C, D, H, W) input, got {x.dim()}-D")
        b, c, d = x.shape[:3]
        folded: dict[str, Any] = {}
        for key, v in kwargs.items():
            if key in named:
                if not isinstance(v, Tensor):
                    raise ValueError(f"volume kwarg {key!r} must be a tensor, got {type(v)}")
                folded[key] = _fold_volume_kwarg(key, v, fold, b, c, d)
            elif isinstance(v, Tensor) and v.dim() == 5:
                raise ValueError(
                    f"5-D tensor kwarg {key!r} is not declared in volume_kwargs {sorted(named)}"
                )
            else:
                folded[key] = v
        x2d = fold_x(x)
        y = kernel2d(x2d, theta, seed=seed, **folded)
        if y.shape[:2] != x2d.shape[:2]:
            raise ValueError(
                f"kernel {getattr(kernel2d, '__name__', kernel2d)!r} changed the folded "
                f"(batch, channel) shape {tuple(x2d.shape[:2])} to {tuple(y.shape[:2])}"
            )
        return unfold(y, b, c, d)

    impl.__name__ = f"per_slice_{fold.name.lower()}_{getattr(kernel2d, '__name__', 'kernel')}"
    return impl
