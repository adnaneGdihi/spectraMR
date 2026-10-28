"""Figures drawn from saved validation images: a pass-by-pass strip and an image embedding."""

from . import image_embedding, progress_strip

FIGURES = {
    "fig_4_3_progress_strip": progress_strip.make,
    "fig_4_4_image_embedding": image_embedding.make,
}
