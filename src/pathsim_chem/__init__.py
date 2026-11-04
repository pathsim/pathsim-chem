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
from .thermodynamics import *
from .process import *
from .neutronics import *


#blocks that were removed -> what to use instead
_REMOVED = {
    "Polynomial": "pathsim.blocks.Polynomial with the coefficients in descending order (numpy.polyval)",
}


def __getattr__(name):
    """Deprecated aliases for the blocks that moved, see 'tritium', and an
    error that names the replacement for the blocks that were removed.
    """
    if name in _REMOVED:
        raise ImportError(f"'{name}' was removed from pathsim-chem, use {_REMOVED[name]}")
    return getattr(tritium, name)
