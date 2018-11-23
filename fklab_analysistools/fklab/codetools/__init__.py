"""
=========================================
FKLab code tools (:mod:`fklab.codetools`)
=========================================

.. currentmodule:: fklab.codetools

Tools for code management, including deprecations warnings.

.. automodule:: fklab.codetools.general


"""

from .general import *

__all__ = [s for s in dir() if not s.startswith('_')]
