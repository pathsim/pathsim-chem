from .residencetime import *
# from .tcap import *


#blocks that moved to pathsim core and to pathsim-fusion, still accessible
#from here as deprecated aliases -> (module, package)
_MOVED = {
    "Splitter": ("pathsim.blocks", "pathsim"),
    "GLC": ("pathsim_fusion", "pathsim-fusion"),
    "Bubbler4": ("pathsim_fusion", "pathsim-fusion"),
    "IonisationChamber": ("pathsim_fusion", "pathsim-fusion"),
}


def __getattr__(name):
    """Deprecated aliases for the blocks that moved, they are resolved
    on access and are not registered as blocks of this package.
    """
    import importlib
    import warnings

    if name not in _MOVED:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module, package = _MOVED[name]
    warnings.warn(
        f"'{name}' has moved to '{module}', import it from there",
        DeprecationWarning,
        stacklevel=2
        )

    try:
        return getattr(importlib.import_module(module), name)
    except ImportError as e:
        raise ImportError(f"'{name}' has moved to {package}: pip install {package}") from e
