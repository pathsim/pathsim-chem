#########################################################################################
##
##                              2-Stream Mixer Block
##
#########################################################################################

# IMPORTS ===============================================================================

from pathsim.blocks.function import Function

# BLOCKS ================================================================================

class Mixer(Function):
    """Algebraic 2-stream mixer with mass and energy balance.

    Mixes two streams by mass balance (additive flows) and energy balance
    (flow-weighted temperature mixing). No phase change or reaction.

    Mathematical Formulation
    -------------------------
    .. math::

        F_{out} = F_1 + F_2

    .. math::

        T_{out} = \\frac{F_1 \\, T_1 + F_2 \\, T_2}{F_{out}}

    Parameters
    ----------
    None — this is a purely algebraic block.
    """

    input_port_labels = {
        "F_1": 0,
        "T_1": 1,
        "F_2": 2,
        "T_2": 3,
    }

    output_port_labels = {
        "F_out": 0,
        "T_out": 1,
    }

    def __init__(self):
        super().__init__(func=self._eval)

    def _eval(self, F_1, T_1, F_2, T_2):
        F_out = F_1 + F_2
        if F_out > 0:
            T_out = (F_1 * T_1 + F_2 * T_2) / F_out
        else:
            T_out = 0.5 * (T_1 + T_2)
        return (F_out, T_out)
