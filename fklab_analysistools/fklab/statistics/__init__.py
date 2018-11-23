"""
===============================================
Statistical functions (:mod:`fklab.statistics`)
===============================================

.. currentmodule:: fklab.statistics

Collection of statistical functions.

.. automodule:: fklab.statistics.circular

.. automodule:: fklab.statistics.information

.. automodule:: fklab.statistics.bootstrap

.. automodule:: fklab.statistics.correlation
   
"""

import circular
import information
import bootstrap
import correlation

__all__ = [s for s in dir() if not s.startswith('_')]
