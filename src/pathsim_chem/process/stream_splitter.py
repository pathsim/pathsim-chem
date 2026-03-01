#########################################################################################
##
##                              Stream Splitter Block
##
#########################################################################################

# IMPORTS ===============================================================================

from pathsim.blocks.function import Function

# BLOCKS ================================================================================

class StreamSplitter(Function):
    """Algebraic stream splitter that divides one stream into two by a fixed ratio.

    Named ``StreamSplitter`` to avoid collision with ``tritium.Splitter``.

    Mathematical Formulation
    -------------------------
    .. math::

        F_1 = \\mathrm{split} \\cdot F_{in}, \\quad F_2 = (1 - \\mathrm{split}) \\cdot F_{in}

    .. math::

        T_1 = T_2 = T_{in}

    Parameters
    ----------
    split : float
        Fraction of feed sent to first outlet (0 to 1). Default 0.5.
    """

    input_port_labels = {
        "F_in": 0,
        "T_in": 1,
    }

    output_port_labels = {
        "F_1": 0,
        "T_1": 1,
        "F_2": 2,
        "T_2": 3,
    }

    def __init__(self, split=0.5):
        if not 0.0 <= split <= 1.0:
            raise ValueError(f"'split' must be in [0, 1] but is {split}")

        self.split = split
        super().__init__(func=self._eval)

    def _eval(self, F_in, T_in):
        F_1 = self.split * F_in
        F_2 = (1.0 - self.split) * F_in
        return (F_1, T_in, F_2, T_in)
