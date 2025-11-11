"""
PathSim-Chem: Chemical Engineering Blocks for PathSim

A toolbox providing specialized blocks for chemical engineering simulations
in the PathSim framework.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]

#for direct block import from main package
from .tritium import *
