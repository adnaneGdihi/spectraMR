import matplotlib.pyplot as plt
import pytest

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_slice_df
from spectramr.infrastructure.reporting.plotters.distributions import _frame


@pytest.fixture
def slice_df():
    return make_slice_df()


@pytest.fixture
def figures(monkeypatch):
    """Figures a plotter saved, so a test can inspect what was drawn."""
    seen = []
    real = _frame.save_plot

    def spy(fig, *args, **kwargs):
        seen.append(fig)
        return real(fig, *args, **kwargs)

    monkeypatch.setattr(_frame, "save_plot", spy)
    yield seen
    plt.close("all")


@pytest.fixture
def drawn_texts():
    """Texts placed on the axes of a figure (panel letters are one character, so skipped)."""

    def read(fig):
        return [t.get_text() for ax in fig.axes for t in ax.texts if len(t.get_text()) > 1]

    return read
