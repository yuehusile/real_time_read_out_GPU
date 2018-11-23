"""
=============================================
Geometrical functions (:mod:`fklab.geometry`)
=============================================

.. currentmodule:: fklab.geometry

Collection of geometrical functions The functions in the utilities ans
transforms module are also directly available in the top level geometry 
package.

.. automodule:: fklab.geometry.utilities

.. automodule:: fklab.geometry.transforms

.. automodule:: fklab.geometry.shapes
    
"""

from .utilities import *
from .transforms import *

import utilities
import transforms

import shapes

__all__ = [s for s in dir() if not s.startswith('_')]
