#########################################################################################
##
##                    IK-CAPE Fugacity Coefficients (Chapter 8)
##
##    Fugacity coefficients from cubic equations of state and the
##    virial equation for vapor-liquid equilibrium calculations.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function

from .equations_of_state import _solve_cubic_eos


# CONSTANTS =============================================================================

R = 8.314462  # gas constant [J/(mol*K)]


# BLOCKS ================================================================================

class FugacityRKS(Function):
    r"""Fugacity coefficients from the Soave-Redlich-Kwong EoS (IK-CAPE Chapter 8.1).

    Computes the fugacity coefficient of each component in a vapor or liquid
    phase using the Soave-Redlich-Kwong cubic equation of state. The fugacity
    coefficient :math:`\phi_i` relates the fugacity of a real gas to its
    partial pressure and is essential for rigorous VLE calculations:

    .. math::

        f_i = \phi_i \, y_i \, P

    For a pure component the fugacity coefficient is:

    .. math::

        \ln \phi_i = Z_i - 1 - \ln(Z_i - B_i)
        - \frac{A_i}{B_i} \ln\!\left(1 + \frac{B_i}{Z_i}\right)

    For component *i* in a mixture:

    .. math::

        \ln \phi_i = \frac{b_i}{b_m}(Z_m - 1) - \ln(Z_m - B_m)
        + \frac{A_m}{B_m}\left(\frac{b_i}{b_m}
        - \frac{2\sum_j x_j a_{ij}}{a_m}\right)
        \ln\!\left(1 + \frac{B_m}{Z_m}\right)

    **Input ports:** ``T`` -- temperature [K], ``P`` -- pressure [Pa].

    **Output ports:** ``phi_0``, ``phi_1``, ... -- fugacity coefficient [-]
    for each component (one output per component).

    Parameters
    ----------
    Tc : float or array_like
        Critical temperature(s) [K].
    Pc : float or array_like
        Critical pressure(s) [Pa].
    omega : float or array_like
        Acentric factor(s) [-].
    x : array_like, optional
        Mole fractions [N]. Required for mixtures, omit for pure components.
    k : array_like, optional
        Binary interaction parameters [N x N]. Default: zero for all pairs.
    phase : str, optional
        ``"vapor"`` (default) or ``"liquid"`` -- selects the EoS root.
    """

    input_port_labels = {"T": 0, "P": 1}

    def __init__(self, Tc, Pc, omega, x=None, k=None, phase="vapor"):
        self.Tc = np.atleast_1d(np.asarray(Tc, dtype=float))
        self.Pc = np.atleast_1d(np.asarray(Pc, dtype=float))
        self.omega = np.atleast_1d(np.asarray(omega, dtype=float))
        self.nc = len(self.Tc)
        self.phase = phase

        if x is None:
            self.x = np.ones(1) if self.nc == 1 else None
        else:
            self.x = np.asarray(x, dtype=float)

        if k is None:
            self.k = np.zeros((self.nc, self.nc))
        else:
            self.k = np.asarray(k, dtype=float).reshape(self.nc, self.nc)

        # constant pure component parameters (SRK)
        self.m = 0.48 + 1.574 * self.omega - 0.176 * self.omega**2
        self.a_c = 0.42748 * R**2 * self.Tc**2 / self.Pc
        self.b_i = 0.08664 * R * self.Tc / self.Pc

        super().__init__(func=self._eval)

    def _eval(self, T, P):
        nc = self.nc
        x = self.x

        # temperature-dependent a_i
        alpha = (1 + self.m * (1 - np.sqrt(T / self.Tc)))**2
        a_i = self.a_c * alpha

        # cross parameters a_ij
        a_ij = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                a_ij[i, j] = np.sqrt(a_i[i] * a_i[j]) * (1 - self.k[i, j])

        # mixing rules
        if nc == 1:
            a_m = a_i[0]
            b_m = self.b_i[0]
        else:
            a_m = 0.0
            for i in range(nc):
                for j in range(nc):
                    a_m += x[i] * x[j] * a_ij[i, j]
            b_m = np.dot(x, self.b_i)

        # dimensionless parameters
        A_m = a_m * P / (R**2 * T**2)
        B_m = b_m * P / (R * T)

        # solve cubic: Z^3 - Z^2 + (A-B-B^2)*Z - AB = 0
        coeffs = [1, -1, A_m - B_m - B_m**2, -A_m * B_m]
        Z = _solve_cubic_eos(coeffs, self.phase)

        # fugacity coefficients for each component
        ln_phi = np.zeros(nc)
        for i in range(nc):
            # sum_j x_j * a_ij
            sum_xa = np.dot(x, a_ij[i, :])
            ln_phi[i] = (self.b_i[i] / b_m * (Z - 1)
                         - np.log(Z - B_m)
                         + A_m / B_m * (self.b_i[i] / b_m
                                        - 2.0 * sum_xa / a_m)
                         * np.log(1 + B_m / Z))

        return tuple(np.exp(ln_phi))


class FugacityPR(Function):
    r"""Fugacity coefficients from the Peng-Robinson EoS (IK-CAPE Chapter 8.2).

    Computes the fugacity coefficient of each component in a vapor or liquid
    phase using the Peng-Robinson cubic equation of state. The PR EoS
    provides improved liquid density predictions compared to SRK and is
    widely used for hydrocarbon and industrial fluid systems.

    For a pure component:

    .. math::

        \ln \phi_i = Z_i - 1 - \ln(Z_i - B_i)
        - \frac{A_i}{2\sqrt{2}\,B_i}
        \ln\frac{Z_i + (1+\sqrt{2})B_i}{Z_i + (1-\sqrt{2})B_i}

    For component *i* in a mixture:

    .. math::

        \ln \phi_i = \frac{b_i}{b_m}(Z_m - 1) - \ln(Z_m - B_m)
        - \frac{A_m}{2\sqrt{2}\,B_m}
        \left(\frac{2\sum_j x_j a_{ij}}{a_m} - \frac{b_i}{b_m}\right)
        \ln\frac{Z_m + (1+\sqrt{2})B_m}{Z_m + (1-\sqrt{2})B_m}

    **Input ports:** ``T`` -- temperature [K], ``P`` -- pressure [Pa].

    **Output ports:** ``phi_0``, ``phi_1``, ... -- fugacity coefficient [-]
    for each component.

    Parameters
    ----------
    Tc : float or array_like
        Critical temperature(s) [K].
    Pc : float or array_like
        Critical pressure(s) [Pa].
    omega : float or array_like
        Acentric factor(s) [-].
    x : array_like, optional
        Mole fractions [N]. Required for mixtures, omit for pure components.
    k : array_like, optional
        Binary interaction parameters [N x N]. Default: zero for all pairs.
    phase : str, optional
        ``"vapor"`` (default) or ``"liquid"`` -- selects the EoS root.
    """

    input_port_labels = {"T": 0, "P": 1}

    def __init__(self, Tc, Pc, omega, x=None, k=None, phase="vapor"):
        self.Tc = np.atleast_1d(np.asarray(Tc, dtype=float))
        self.Pc = np.atleast_1d(np.asarray(Pc, dtype=float))
        self.omega = np.atleast_1d(np.asarray(omega, dtype=float))
        self.nc = len(self.Tc)
        self.phase = phase

        if x is None:
            self.x = np.ones(1) if self.nc == 1 else None
        else:
            self.x = np.asarray(x, dtype=float)

        if k is None:
            self.k = np.zeros((self.nc, self.nc))
        else:
            self.k = np.asarray(k, dtype=float).reshape(self.nc, self.nc)

        # constant pure component parameters (PR)
        self.m = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        self.a_c = 0.45724 * R**2 * self.Tc**2 / self.Pc
        self.b_i = 0.07780 * R * self.Tc / self.Pc

        super().__init__(func=self._eval)

    def _eval(self, T, P):
        nc = self.nc
        x = self.x
        sqrt2 = np.sqrt(2.0)

        # temperature-dependent a_i
        alpha = (1 + self.m * (1 - np.sqrt(T / self.Tc)))**2
        a_i = self.a_c * alpha

        # cross parameters a_ij
        a_ij = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                a_ij[i, j] = np.sqrt(a_i[i] * a_i[j]) * (1 - self.k[i, j])

        # mixing rules
        if nc == 1:
            a_m = a_i[0]
            b_m = self.b_i[0]
        else:
            a_m = 0.0
            for i in range(nc):
                for j in range(nc):
                    a_m += x[i] * x[j] * a_ij[i, j]
            b_m = np.dot(x, self.b_i)

        # dimensionless parameters
        A_m = a_m * P / (R**2 * T**2)
        B_m = b_m * P / (R * T)

        # solve PR cubic: Z^3 - (1-B)Z^2 + (A-3B^2-2B)Z - (AB-B^2-B^3) = 0
        coeffs = [1,
                  -(1 - B_m),
                  A_m - 3 * B_m**2 - 2 * B_m,
                  -(A_m * B_m - B_m**2 - B_m**3)]
        Z = _solve_cubic_eos(coeffs, self.phase)

        # fugacity coefficients for each component
        ln_phi = np.zeros(nc)
        log_term = np.log((Z + (1 + sqrt2) * B_m) / (Z + (1 - sqrt2) * B_m))

        for i in range(nc):
            sum_xa = np.dot(x, a_ij[i, :])
            ln_phi[i] = (self.b_i[i] / b_m * (Z - 1)
                         - np.log(Z - B_m)
                         - A_m / (2 * sqrt2 * B_m)
                         * (2.0 * sum_xa / a_m - self.b_i[i] / b_m)
                         * log_term)

        return tuple(np.exp(ln_phi))


class FugacityVirial(Function):
    r"""Fugacity coefficients from the truncated virial equation (IK-CAPE Chapter 8.3).

    Computes fugacity coefficients using the second virial equation of state,
    which is valid at low to moderate pressures (typically below 10-15 bar).
    The virial EoS is theoretically exact in the limit of low density and is
    particularly useful for gases at moderate conditions where cubic EoS may
    be unnecessarily complex.

    For a pure component:

    .. math::

        \phi_i^0 = \frac{\exp(2 B_{ii} / v)}{1 + B_{ii}/v}

    where :math:`v` is the molar volume from:

    .. math::

        v = \frac{RT}{2P}\left(1 + \sqrt{1 + \frac{4 P B_{ii}}{RT}}\right)

    For component *i* in a mixture:

    .. math::

        \phi_i^v = \frac{\exp\!\left(\frac{2}{v}\sum_j y_j B_{ij}\right)}
        {1 + B_m/v}

    where :math:`B_m = \sum_i \sum_j y_i y_j B_{ij}` and
    :math:`Z_m = 1 + B_m/v`.

    **Input ports:** ``T`` -- temperature [K], ``P`` -- pressure [Pa].

    **Output ports:** ``phi_0``, ``phi_1``, ... -- fugacity coefficient [-]
    for each component.

    Parameters
    ----------
    B : array_like
        Second virial coefficients [N x N] in [m^3/mol]. For a pure component,
        pass a scalar or 1x1 array. For a mixture, pass the full symmetric
        matrix including cross-coefficients :math:`B_{ij}`.
        These are typically functions of temperature, but here assumed constant
        at the evaluation temperature.
    y : array_like, optional
        Vapor-phase mole fractions [N]. Required for mixtures.
    """

    input_port_labels = {"T": 0, "P": 1}

    def __init__(self, B, y=None):
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        self.nc = self.B.shape[0]

        if y is None:
            self.y = np.ones(1) if self.nc == 1 else None
        else:
            self.y = np.asarray(y, dtype=float)

        super().__init__(func=self._eval)

    def _eval(self, T, P):
        nc = self.nc
        y = self.y
        B_mat = self.B

        # mixture second virial coefficient
        B_m = 0.0
        for i in range(nc):
            for j in range(nc):
                B_m += y[i] * y[j] * B_mat[i, j]

        # molar volume from truncated virial
        v = R * T / (2 * P) * (1 + np.sqrt(1 + 4 * P * B_m / (R * T)))

        # fugacity coefficients
        ln_phi = np.zeros(nc)
        for i in range(nc):
            sum_yB = np.dot(y, B_mat[i, :])
            ln_phi[i] = 2.0 * sum_yB / v - np.log(1 + B_m / v)

        return tuple(np.exp(ln_phi))
