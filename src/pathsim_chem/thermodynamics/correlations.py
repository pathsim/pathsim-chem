#########################################################################################
##
##               IK-CAPE Pure Component Property Correlations
##
##    Temperature-dependent correlation functions for thermodynamic properties
##    as defined in the IK-CAPE Thermodynamics Module, Chapter 2.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function


# BLOCKS ================================================================================

class Polynomial(Function):
    r"""Polynomial correlation (IK-CAPE code: POLY).

    .. math::

        f(T) = \sum_{i=0}^{n} a_i \, T^i

    General-purpose polynomial for temperature-dependent properties.

    Parameters
    ----------
    coeffs : list or tuple of float
        Polynomial coefficients :math:`[a_0, a_1, \ldots, a_n]` (up to 10).
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, coeffs):
        self.coeffs = tuple(coeffs)
        super().__init__(func=self._eval)

    def _eval(self, T):
        return sum(a * T**i for i, a in enumerate(self.coeffs))


class ExponentialPolynomial(Function):
    r"""Exponential polynomial correlation (IK-CAPE code: EPOL).

    .. math::

        f(T) = 10^{\sum_{i=0}^{n} a_i \, T^i}

    General-purpose correlation for properties that vary over orders of magnitude.

    Parameters
    ----------
    coeffs : list or tuple of float
        Polynomial exponent coefficients :math:`[a_0, a_1, \ldots, a_n]` (up to 10).
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, coeffs):
        self.coeffs = tuple(coeffs)
        super().__init__(func=self._eval)

    def _eval(self, T):
        exponent = sum(a * T**i for i, a in enumerate(self.coeffs))
        return 10**exponent


class Watson(Function):
    r"""Watson correlation (IK-CAPE code: WATS).

    .. math::

        f(T) = a_0 \, (a_2 - T)^{a_1} + a_3

    Commonly used for heat of vaporization.

    Parameters
    ----------
    a0, a1, a2, a3 : float
        Watson correlation coefficients. :math:`a_2` is typically the
        critical temperature.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1, a2, a3=0.0):
        self.coeffs = (a0, a1, a2, a3)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3 = self.coeffs
        return a0 * (a2 - T)**a1 + a3


class Antoine(Function):
    r"""Antoine correlation (IK-CAPE code: ANTO).

    .. math::

        \ln f(T) = a_0 - \frac{a_1}{T + a_2}

    Standard three-parameter vapor pressure correlation.

    Parameters
    ----------
    a0, a1, a2 : float
        Antoine coefficients. For natural-log form with T in Kelvin.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1, a2):
        self.coeffs = (a0, a1, a2)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2 = self.coeffs
        return np.exp(a0 - a1 / (T + a2))


class ExtendedAntoine(Function):
    r"""Extended Antoine correlation (IK-CAPE code: ANT1).

    .. math::

        \ln f(T) = a_0 + \frac{a_1}{T + a_2} + a_3 \, T + a_4 \, \ln(T) + a_5 \, T^{a_6}

    Extended form for wider temperature range vapor pressure.

    Parameters
    ----------
    a0, a1, a2, a3, a4, a5, a6 : float
        Extended Antoine coefficients.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1, a2, a3=0.0, a4=0.0, a5=0.0, a6=0.0):
        self.coeffs = (a0, a1, a2, a3, a4, a5, a6)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3, a4, a5, a6 = self.coeffs
        ln_f = a0 + a1 / (T + a2) + a3 * T + a4 * np.log(T) + a5 * T**a6
        return np.exp(ln_f)


class Kirchhoff(Function):
    r"""Kirchhoff correlation (IK-CAPE code: KIRC).

    .. math::

        \ln f(T) = a_0 - \frac{a_1}{T} - a_2 \, \ln(T)

    Three-parameter vapor pressure correlation.

    Parameters
    ----------
    a0, a1, a2 : float
        Kirchhoff coefficients.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1, a2):
        self.coeffs = (a0, a1, a2)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2 = self.coeffs
        return np.exp(a0 - a1 / T - a2 * np.log(T))


class ExtendedKirchhoff(Function):
    r"""Extended Kirchhoff correlation (IK-CAPE code: KIR1).

    .. math::

        \ln f(T) = a_0 + \frac{a_1}{T} + a_2 \, \ln(T) + a_3 \, T^{a_4}

    Extended form with additional power term.

    Parameters
    ----------
    a0, a1, a2, a3, a4 : float
        Extended Kirchhoff coefficients.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1, a2, a3=0.0, a4=0.0):
        self.coeffs = (a0, a1, a2, a3, a4)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3, a4 = self.coeffs
        return np.exp(a0 + a1 / T + a2 * np.log(T) + a3 * T**a4)


class Sutherland(Function):
    r"""Sutherland correlation (IK-CAPE code: SUTH).

    .. math::

        f(T) = \frac{a_0 \, \sqrt{T}}{1 + a_1 / T}

    Used for gas-phase viscosity estimation.

    Parameters
    ----------
    a0, a1 : float
        Sutherland coefficients. :math:`a_1` is the Sutherland constant.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1):
        self.coeffs = (a0, a1)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1 = self.coeffs
        return a0 * np.sqrt(T) / (1 + a1 / T)


class Wagner(Function):
    r"""Wagner correlation (IK-CAPE code: WAGN).

    .. math::

        \ln f(T) = \ln(a_1) + \frac{1}{T_r} \left( a_2 \, \tau + a_3 \, \tau^{1.5}
        + a_4 \, \tau^3 + a_5 \, \tau^6 \right)

    where :math:`T_r = T / a_0` and :math:`\tau = 1 - T_r`.

    High-accuracy vapor pressure correlation.

    Parameters
    ----------
    Tc : float
        Critical temperature :math:`a_0` [K].
    Pc : float
        Critical pressure :math:`a_1` [Pa].
    a2, a3, a4, a5 : float
        Wagner correlation coefficients.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, Tc, Pc, a2, a3, a4, a5):
        self.coeffs = (Tc, Pc, a2, a3, a4, a5)
        super().__init__(func=self._eval)

    def _eval(self, T):
        Tc, Pc, a2, a3, a4, a5 = self.coeffs
        Tr = T / Tc
        tau = 1 - Tr
        ln_f = np.log(Pc) + (1 / Tr) * (a2 * tau + a3 * tau**1.5 + a4 * tau**3 + a5 * tau**6)
        return np.exp(ln_f)


class LiquidHeatCapacity(Function):
    r"""Liquid heat capacity correlation (IK-CAPE code: CPL).

    .. math::

        f(T) = a_0 + a_1 \, T + a_2 \, T^2 + a_3 \, T^3 + \frac{a_4}{T^2}

    Five-parameter liquid heat capacity correlation.

    Parameters
    ----------
    a0, a1, a2, a3, a4 : float
        Liquid Cp coefficients.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1=0.0, a2=0.0, a3=0.0, a4=0.0):
        self.coeffs = (a0, a1, a2, a3, a4)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3, a4 = self.coeffs
        return a0 + a1 * T + a2 * T**2 + a3 * T**3 + a4 / T**2


class ExtendedLiquidHeatCapacity(Function):
    r"""Extended liquid heat capacity correlation (IK-CAPE code: ICPL).

    .. math::

        f(T) = a_0 + a_1 \, T + a_2 \, T^2 + a_3 \, T^3 + a_4 \, T^4 + \frac{a_5}{T}

    Six-parameter extended liquid heat capacity correlation.

    Parameters
    ----------
    a0, a1, a2, a3, a4, a5 : float
        Extended liquid Cp coefficients.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1=0.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0):
        self.coeffs = (a0, a1, a2, a3, a4, a5)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3, a4, a5 = self.coeffs
        return a0 + a1 * T + a2 * T**2 + a3 * T**3 + a4 * T**4 + a5 / T


class DynamicViscosity(Function):
    r"""Dynamic viscosity correlation (IK-CAPE code: VISC).

    .. math::

        f(T) = a_0 \, \exp\!\left(\frac{a_1}{T}\right) + a_2

    Simple two/three-parameter liquid viscosity correlation.

    Parameters
    ----------
    a0, a1 : float
        Pre-exponential factor and activation energy parameter.
    a2 : float, optional
        Additive constant (default 0).
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1, a2=0.0):
        self.coeffs = (a0, a1, a2)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2 = self.coeffs
        return a0 * np.exp(a1 / T) + a2


class Rackett(Function):
    r"""Rackett correlation (IK-CAPE code: RACK).

    .. math::

        f(T) = \frac{a_0}{a_1^{\left(1 + (1 - T/a_2)^{a_3}\right)}}

    Saturated liquid density correlation.

    Parameters
    ----------
    a0 : float
        Scaling parameter (related to critical volume).
    a1 : float
        Rackett parameter (base of exponent).
    a2 : float
        Critical temperature [K].
    a3 : float
        Exponent parameter (often 2/7).
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1, a2, a3):
        self.coeffs = (a0, a1, a2, a3)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3 = self.coeffs
        return a0 / a1**(1 + (1 - T / a2)**a3)


class AlyLee(Function):
    r"""Aly-Lee correlation (IK-CAPE code: ALYL).

    .. math::

        f(T) = a_0 + a_1 \left(\frac{a_2 / T}{\sinh(a_2 / T)}\right)^2
        + a_3 \left(\frac{a_4 / T}{\cosh(a_4 / T)}\right)^2

    Ideal gas heat capacity correlation.

    Parameters
    ----------
    a0, a1, a2, a3, a4 : float
        Aly-Lee coefficients.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1, a2, a3, a4):
        self.coeffs = (a0, a1, a2, a3, a4)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3, a4 = self.coeffs
        x = a2 / T
        y = a4 / T
        return a0 + a1 * (x / np.sinh(x))**2 + a3 * (y / np.cosh(y))**2


class DIPPR4(Function):
    r"""DIPPR Equation 4 correlation (IK-CAPE code: DIP4).

    .. math::

        f(T) = a_1 \, (1 - T_r)^h, \quad
        h = a_2 + a_3 \, T_r + a_4 \, T_r^2 + a_5 \, T_r^3

    where :math:`T_r = T / a_0`.

    Used for heat of vaporization and surface tension.

    Parameters
    ----------
    Tc : float
        Critical temperature :math:`a_0` [K].
    a1, a2, a3, a4, a5 : float
        DIPPR-4 coefficients.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, Tc, a1, a2, a3=0.0, a4=0.0, a5=0.0):
        self.coeffs = (Tc, a1, a2, a3, a4, a5)
        super().__init__(func=self._eval)

    def _eval(self, T):
        Tc, a1, a2, a3, a4, a5 = self.coeffs
        Tr = T / Tc
        h = a2 + a3 * Tr + a4 * Tr**2 + a5 * Tr**3
        return a1 * (1 - Tr)**h


class DIPPR5(Function):
    r"""DIPPR Equation 5 correlation (IK-CAPE code: DIP5).

    .. math::

        f(T) = \frac{a_0 \, T^{a_1}}{1 + a_2 / T + a_3 / T^2}

    Used for vapor viscosity and thermal conductivity.

    Parameters
    ----------
    a0, a1, a2, a3 : float
        DIPPR-5 coefficients.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"value": 0}

    def __init__(self, a0, a1, a2=0.0, a3=0.0):
        self.coeffs = (a0, a1, a2, a3)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3 = self.coeffs
        return a0 * T**a1 / (1 + a2 / T + a3 / T**2)
