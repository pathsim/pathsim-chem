#########################################################################################
##
##                    IK-CAPE Calculation of Averages (Chapter 3)
##
##    Mixing rules for computing mixture properties from pure component values
##    and composition data.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function


# BLOCKS ================================================================================

class MolarAverage(Function):
    r"""Molar average mixing rule (IK-CAPE code: MOLA).

    .. math::

        \text{average} = \sum_i x_i \, \text{value}_i

    Parameters
    ----------
    x : array_like
        Mole fractions for each component. Determines the number of input ports.
    """

    output_port_labels = {"average": 0}

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        super().__init__(func=self._eval)

    def _eval(self, *values):
        return np.dot(self.x, values)


class MassAverage(Function):
    r"""Mass fraction average mixing rule (IK-CAPE code: MASS).

    .. math::

        \text{average} = \frac{\sum_i w_i \, \text{value}_i}{\sum_i w_i}

    Parameters
    ----------
    w : array_like
        Mass fractions for each component. Determines the number of input ports.
    """

    output_port_labels = {"average": 0}

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)
        super().__init__(func=self._eval)

    def _eval(self, *values):
        return np.dot(self.w, values) / np.sum(self.w)


class LogMolarAverage(Function):
    r"""Logarithmic molar average mixing rule (IK-CAPE code: MOLG).

    .. math::

        \ln(\text{average}) = \sum_i x_i \, \ln(\text{value}_i)

    Parameters
    ----------
    x : array_like
        Mole fractions for each component. Determines the number of input ports.
    """

    output_port_labels = {"average": 0}

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        super().__init__(func=self._eval)

    def _eval(self, *values):
        return np.exp(np.dot(self.x, np.log(values)))


class LogMassAverage(Function):
    r"""Logarithmic mass fraction average mixing rule (IK-CAPE code: MALG).

    .. math::

        \ln(\text{average}) = \frac{\sum_i w_i \, \ln(\text{value}_i)}{\sum_i w_i}

    Parameters
    ----------
    w : array_like
        Mass fractions for each component. Determines the number of input ports.
    """

    output_port_labels = {"average": 0}

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)
        super().__init__(func=self._eval)

    def _eval(self, *values):
        return np.exp(np.dot(self.w, np.log(values)) / np.sum(self.w))


class LambdaAverage(Function):
    r"""Thermal conductivity average for gaseous mixtures (IK-CAPE code: LAMB).

    .. math::

        \lambda_m = 0.5 \left( \sum_i x_i \, \lambda_i
        + \frac{1}{\sum_i x_i / \lambda_i} \right)

    Arithmetic-harmonic mean of pure component conductivities.

    Parameters
    ----------
    x : array_like
        Mole fractions for each component. Determines the number of input ports.
    """

    output_port_labels = {"average": 0}

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        super().__init__(func=self._eval)

    def _eval(self, *values):
        v = np.asarray(values)
        return 0.5 * (np.dot(self.x, v) + 1.0 / np.dot(self.x, 1.0 / v))


class ViscosityAverage(Function):
    r"""Viscosity average for gaseous mixtures (IK-CAPE code: VISC).

    .. math::

        \mu_m = \frac{\sum_i x_i \, \sqrt{M_i} \, \mu_i}
                     {\sum_i x_i \, \sqrt{M_i}}

    Molecular-weight-weighted average.

    Parameters
    ----------
    x : array_like
        Mole fractions for each component. Determines the number of input ports.
    M : array_like
        Molecular weights [kg/kmol] for each component.
    """

    output_port_labels = {"average": 0}

    def __init__(self, x, M):
        self.x = np.asarray(x, dtype=float)
        self.M = np.asarray(M, dtype=float)
        self.weights = self.x * np.sqrt(self.M)
        super().__init__(func=self._eval)

    def _eval(self, *values):
        return np.dot(self.weights, values) / np.sum(self.weights)


class VolumeAverage(Function):
    r"""Volume-based density average (IK-CAPE code: VOLU).

    .. math::

        \rho_m = \frac{1}{\sum_i x_i / \rho_i}

    Harmonic mean weighted by mole fractions, used for mixture density.

    Parameters
    ----------
    x : array_like
        Mole fractions for each component. Determines the number of input ports.
    """

    output_port_labels = {"average": 0}

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        super().__init__(func=self._eval)

    def _eval(self, *values):
        return 1.0 / np.dot(self.x, 1.0 / np.asarray(values))


class WilkeViscosity(Function):
    r"""Wilke mixing rule for gas viscosity (IK-CAPE code: WILK).

    .. math::

        \mu_m = \sum_i \frac{y_i \, \mu_i}{\sum_j y_j \, F_{ij}}

    .. math::

        F_{ij} = \frac{\left(1 + \sqrt{\mu_i / \mu_j}
        \, \sqrt[4]{M_j / M_i}\right)^2}
        {\sqrt{8 \left(1 + M_i / M_j\right)}}

    Parameters
    ----------
    y : array_like
        Mole fractions (vapor phase) for each component.
    M : array_like
        Molecular weights [kg/kmol] for each component.
    """

    output_port_labels = {"average": 0}

    def __init__(self, y, M):
        self.y = np.asarray(y, dtype=float)
        self.M = np.asarray(M, dtype=float)
        n = len(self.M)

        # precompute the molecular weight ratios
        self._Mr = np.zeros((n, n))
        self._denom = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self._Mr[i, j] = (self.M[j] / self.M[i])**0.25
                self._denom[i, j] = np.sqrt(8 * (1 + self.M[i] / self.M[j]))

        super().__init__(func=self._eval)

    def _eval(self, *values):
        v = np.asarray(values)
        n = len(v)
        result = 0.0
        for i in range(n):
            denom_sum = 0.0
            for j in range(n):
                F_ij = (1 + np.sqrt(v[i] / v[j]) * self._Mr[i, j])**2 / self._denom[i, j]
                denom_sum += self.y[j] * F_ij
            result += self.y[i] * v[i] / denom_sum
        return result


class WassiljewaMasonSaxena(Function):
    r"""Wassiljewa-Mason-Saxena mixing rule for gas thermal conductivity
    (IK-CAPE code: WAMA).

    .. math::

        \lambda_m = \sum_i \frac{y_i \, \lambda_i}{\sum_j y_j \, F_{ij}}

    where :math:`F_{ij}` uses viscosity values :math:`\eta`:

    .. math::

        F_{ij} = \frac{\left(1 + \sqrt{\eta_i / \eta_j}
        \, \sqrt[4]{M_j / M_i}\right)^2}
        {\sqrt{8 \left(1 + M_i / M_j\right)}}

    The first N inputs are thermal conductivities, the next N are viscosities.

    Parameters
    ----------
    y : array_like
        Mole fractions (vapor phase) for each component.
    M : array_like
        Molecular weights [kg/kmol] for each component.
    """

    output_port_labels = {"average": 0}

    def __init__(self, y, M):
        self.y = np.asarray(y, dtype=float)
        self.M = np.asarray(M, dtype=float)
        self.n = len(self.M)

        # precompute molecular weight ratios
        n = self.n
        self._Mr = np.zeros((n, n))
        self._denom = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self._Mr[i, j] = (self.M[j] / self.M[i])**0.25
                self._denom[i, j] = np.sqrt(8 * (1 + self.M[i] / self.M[j]))

        super().__init__(func=self._eval)

    def _eval(self, *inputs):
        n = self.n
        lam = np.asarray(inputs[:n])   # thermal conductivities
        eta = np.asarray(inputs[n:])   # viscosities

        result = 0.0
        for i in range(n):
            denom_sum = 0.0
            for j in range(n):
                F_ij = (1 + np.sqrt(eta[i] / eta[j]) * self._Mr[i, j])**2 / self._denom[i, j]
                denom_sum += self.y[j] * F_ij
            result += self.y[i] * lam[i] / denom_sum
        return result


class DIPPRSurfaceTension(Function):
    r"""DIPPR surface tension average (IK-CAPE code: DIST).

    .. math::

        \text{average} = \left(
        \frac{\sum_i x_i \, V_i \, \text{value}_i^{1/4}}
        {\sum_i x_i \, V_i} \right)^4

    Parameters
    ----------
    x : array_like
        Mole fractions for each component. Determines the number of input ports.
    V : array_like
        Molar volumes [m^3/kmol] for each component.
    """

    output_port_labels = {"average": 0}

    def __init__(self, x, V):
        self.x = np.asarray(x, dtype=float)
        self.V = np.asarray(V, dtype=float)
        self.xV = self.x * self.V
        super().__init__(func=self._eval)

    def _eval(self, *values):
        v = np.asarray(values)
        return (np.dot(self.xV, v**0.25) / np.sum(self.xV))**4
