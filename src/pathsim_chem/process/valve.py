#########################################################################################
##
##                              Pressure-Drop Valve Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function

# BLOCKS ================================================================================

class Valve(Function):
    """Algebraic pressure-drop valve with standard flow equation.

    Models an isenthalpic valve (no temperature change for liquids)
    with flow proportional to the square root of the pressure drop.

    Mathematical Formulation
    -------------------------
    .. math::

        F = C_v \\sqrt{|P_{in} - P_{out}|} \\cdot \\mathrm{sign}(P_{in} - P_{out})

    .. math::

        T_{out} = T_{in}

    Parameters
    ----------
    Cv : float
        Valve flow coefficient. Must be positive.
    """

    input_port_labels = {
        "P_in":  0,
        "P_out": 1,
        "T_in":  2,
    }

    output_port_labels = {
        "F":     0,
        "T_out": 1,
    }

    def __init__(self, Cv=1.0):
        if Cv <= 0:
            raise ValueError(f"'Cv' must be positive but is {Cv}")

        self.Cv = Cv
        super().__init__(func=self._eval)

    def _eval(self, P_in, P_out, T_in):
        dP = P_in - P_out
        F = self.Cv * np.sqrt(np.abs(dP)) * np.sign(dP)
        return (F, T_in)
