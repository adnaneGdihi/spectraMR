"""MRIQC-style image-quality metrics from one magnitude image.

Computed from the saved prediction/target images, so they describe a single slice:
the entropy focus criterion (EFC), foreground-to-background energy ratio (FBER), SNR
against the background noise and that noise's standard deviation. There is no time
series here, so FD, DVARS and tSNR are not computed.
"""

from __future__ import annotations

import numpy as np

_BINS = 256


def otsu_threshold(image: np.ndarray) -> float:
    """Otsu's threshold of ``image`` (maximises between-class variance)."""
    values = np.asarray(image, dtype=np.float64).ravel()
    hist, edges = np.histogram(values, bins=_BINS)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(np.argmax(between))])


def foreground_mask(image: np.ndarray) -> np.ndarray:
    """Boolean mask of the head: above the Otsu threshold of the magnitude image."""
    img = np.asarray(image, dtype=np.float64)
    return img > otsu_threshold(img)


def efc(image: np.ndarray) -> float:
    """Entropy focus criterion in [0, 1]; higher means more ghosting or blur."""
    x = np.abs(np.asarray(image, dtype=np.float64)).ravel()
    n = x.size
    b_max = np.sqrt(np.sum(x**2))
    if b_max == 0:
        return float("nan")
    efc_max = n * (1.0 / np.sqrt(n)) * np.log(1.0 / np.sqrt(n))
    ratio = x / b_max
    return float(np.sum(ratio * np.log(ratio + 1e-16)) / efc_max)


def iqms(image: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """``{efc, fber, snr, bg_sigma}`` of ``image`` with the foreground ``mask``.

    Returns NaN for a quantity whose region is empty or noise-free rather than guessing.
    """
    img = np.asarray(image, dtype=np.float64)
    fg = foreground_mask(img) if mask is None else np.asarray(mask, dtype=bool)
    bg = ~fg
    out = {"efc": efc(img), "fber": float("nan"), "snr": float("nan"), "bg_sigma": float("nan")}
    if fg.sum() < 2 or bg.sum() < 2:
        return out
    sigma = float(np.std(img[bg]))
    out["bg_sigma"] = sigma
    bg_energy = float(np.mean(img[bg] ** 2))
    if bg_energy > 0:
        out["fber"] = float(np.mean(img[fg] ** 2) / bg_energy)
    if sigma > 0:
        out["snr"] = float(np.mean(img[fg]) / sigma)
    return out


__all__ = ["efc", "foreground_mask", "iqms", "otsu_threshold"]
