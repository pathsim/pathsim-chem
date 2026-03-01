#########################################################################################
##
##                              Duty-Specified Heater/Cooler Block
##
#########################################################################################

# IMPORTS ===============================================================================

from pathsim.blocks.function import Function

# BLOCKS ================================================================================

class Heater(Function):
    """Algebraic duty-specified heater/cooler with no thermal mass.

    Raises or lowers the stream temperature by a specified heat duty.
    Flow passes through unchanged.

    Mathematical Formulation
    -------------------------
    .. math::

        T_{out} = T_{in} + \\frac{Q}{F \\, \\rho \\, C_p}

    .. math::

        F_{out} = F_{in}

    Parameters
    ----------
    rho : float
        Fluid density [kg/m³].
    Cp : float
        Fluid heat capacity [J/(kg·K)].
    """

    input_port_labels = {
        "F":   0,
        "T_in": 1,
        "Q":   2,
    }

    output_port_labels = {
        "F_out": 0,
        "T_out": 1,
    }

    def __init__(self, rho=1000.0, Cp=4184.0):
        if rho <= 0:
            raise ValueError(f"'rho' must be positive but is {rho}")
        if Cp <= 0:
            raise ValueError(f"'Cp' must be positive but is {Cp}")

        self.rho = rho
        self.Cp = Cp
        super().__init__(func=self._eval)

    def _eval(self, F, T_in, Q):
        if F > 0:
            T_out = T_in + Q / (F * self.rho * self.Cp)
        else:
            T_out = T_in
        return (F, T_out)
