"""
================================
Plot (:mod:`fklab.plot`)
================================

.. currentmodule:: fklab.plot

Tools for data visualization.

.. automodule:: fklab.plot.plotting

.. automodule:: fklab.plot.neuralynx

.. automodule:: fklab.plot.open_ephys

.. automodule:: fklab.plot.artists

.. automodule:: fklab.plot.interaction

.. automodule:: fklab.plot.utilities

"""

from .plotting import plot_signals, plot_events, plot_segments, plot_raster, plot_spectrogram

from . import artists
from . import neuralynx
from . import open_ephys

__all__ = [_s for _s in dir() if not _s.startswith('_')]
