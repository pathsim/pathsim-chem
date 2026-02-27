#########################################################################################
##
##                IK-CAPE Corrections (Chapters 5 and 6)
##
##    Poynting correction and Henry's law constant for vapor-liquid
##    equilibrium calculations.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function


# CONSTANTS =============================================================================

R = 8.314462  # gas constant [J/(mol*K)]


# BLOCKS ================================================================================

class PoyntingCorrection(Function):
    r"""Poynting pressure correction factor (IK-CAPE Chapter 5).

    Accounts for the effect of total system pressure on the fugacity of a
    liquid component when the system pressure :math:`P` differs from the
    pure component saturation pressure :math:`P_i^s`. The Poynting factor
    is used in rigorous VLE calculations via the modified Raoult's law:

    .. math::

        y_i P = x_i \gamma_i P_i^s \phi_i^s F_{p,i} / \phi_i^V

    At moderate pressures (below ~10 bar), the Poynting correction is
    very close to unity and can often be neglected. It becomes significant
    at high pressures or for components with large liquid molar volumes.

    .. math::

        \ln F_{p,i} = \frac{v_i^L \, (P - P_i^s)}{R T}

    **Input ports:**

    - ``T`` -- temperature [K]
    - ``P`` -- total system pressure [Pa]
    - ``Psat`` -- pure component saturation pressure [Pa] (e.g., from Antoine block)
    - ``vL`` -- liquid molar volume [m^3/mol] (e.g., from Rackett block)

    **Output port:** ``Fp`` -- dimensionless Poynting correction factor [-].

    Parameters
    ----------
    None. All values are received as dynamic inputs from upstream blocks.
    """

    input_port_labels = {"T": 0, "P": 1, "Psat": 2, "vL": 3}
    output_port_labels = {"Fp": 0}

    def __init__(self):
        super().__init__(func=self._eval)

    def _eval(self, T, P, Psat, vL):
        return np.exp(vL * (P - Psat) / (R * T))


class HenryConstant(Function):
    r"""Temperature-dependent Henry's law constant (IK-CAPE Chapter 6).

    Computes the Henry's law constant :math:`H_{i,j}` for a dissolved gas
    species *i* in solvent *j* as a function of temperature. Henry's law
    relates the partial pressure of a dilute gas above a solution to its
    mole fraction in the liquid:

    .. math::

        p_i = H_{i,j} \, x_i

    The temperature dependence follows the four-parameter correlation:

    .. math::

        \ln H_{i,j} = a + \frac{b}{T} + c \, \ln T + d \, T

    Coefficients are specific to each gas-solvent pair and are available
    from standard databases (e.g., DECHEMA, NIST).

    **Input port:** ``T`` -- temperature [K].

    **Output port:** ``H`` -- Henry's law constant [Pa].

    Parameters
    ----------
    a : float
        Constant term in :math:`\ln H`.
    b : float, optional
        Coefficient of :math:`1/T`. Controls the temperature sensitivity
        (related to enthalpy of dissolution). Default: 0.
    c : float, optional
        Coefficient of :math:`\ln T`. Default: 0.
    d : float, optional
        Coefficient of :math:`T`. Default: 0.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"H": 0}

    def __init__(self, a, b=0.0, c=0.0, d=0.0):
        self.coeffs = (a, b, c, d)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a, b, c, d = self.coeffs
        return np.exp(a + b / T + c * np.log(T) + d * T)
