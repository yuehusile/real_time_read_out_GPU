"""
================================
Signals (:mod:`fklab.signals`)
================================

.. currentmodule:: fklab.signals

Functions for digital signal processing, including filtering, smoothing,
rate conversions, spectral analysis, etc. The functions in the basic_algorithms
module are also directly available in the top level signals package.

.. automodule:: fklab.signals.basic_algorithms

.. automodule:: fklab.signals.kernelsmoothing

.. automodule:: fklab.signals.filter

.. automodule:: fklab.signals.multirate

.. automodule:: fklab.signals.multitaper

.. automodule:: fklab.signals.ripple

.. automodule:: fklab.signals.ica

"""

from .basic_algorithms import (detect_mountains, zerocrossing, localmaxima, 
    localminima, localextrema, remove_artefacts, extract_data_windows,
    extract_trigger_windows, generate_windows, generate_trigger_windows,
    event_triggered_average)

__all__ = [_s for _s in dir() if not _s.startswith('_')]
