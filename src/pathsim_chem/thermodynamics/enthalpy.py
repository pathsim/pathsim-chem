#########################################################################################
##
##                         IK-CAPE Enthalpy (Chapter 9)
##
##    Excess enthalpy models (9.6) and isothermal pressure dependency
##    of the enthalpy in the vapor phase (9.7).
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function

from .equations_of_state import _solve_cubic_eos


# CONSTANTS =============================================================================

R = 8.314462  # gas constant [J/(mol*K)]


# EXCESS ENTHALPY MODELS (9.6) ==========================================================

class ExcessEnthalpyNRTL(Function):
    r"""Excess enthalpy from the NRTL model (IK-CAPE Chapter 9.6.1).

    Computes the molar excess enthalpy :math:`h^E` of a liquid mixture
    using the NRTL (Non-Random Two-Liquid) model. The excess enthalpy
    is obtained from the temperature derivative of the excess Gibbs energy:

    .. math::

        h^E = -T^2 \frac{\partial (g^E / T)}{\partial T}
        = -R T^2 \sum_i x_i \frac{A'_i B_i - A_i B'_i}{B_i^2}

    where:

    .. math::

        A_i = \sum_j x_j G_{j,i} \tau_{j,i}, \quad
        A'_i = \sum_j x_j (G'_{j,i} \tau_{j,i} + G_{j,i} \tau'_{j,i})

        B_i = \sum_j x_j G_{j,i}, \quad
        B'_i = \sum_j x_j G'_{j,i}

    Temperature derivatives:

    .. math::

        \tau'_{j,i} = -b_{j,i}/T^2 + e_{j,i}/T + f_{j,i}

        G'_{j,i} = -G_{j,i}(d_{j,i} \tau_{j,i} + S_{j,i} \tau'_{j,i})

    **Input port:** ``T`` -- temperature [K].

    **Output port:** ``hE`` -- molar excess enthalpy [J/mol].

    Parameters
    ----------
    x : array_like
        Mole fractions [N]. Fixed mixture composition.
    a : array_like
        Constant part of :math:`\tau` [N x N]. Diagonal should be zero.
    b : array_like, optional
        Coefficient of :math:`1/T` in :math:`\tau` [N x N]. Default: zeros.
    c : array_like, optional
        Constant part of non-randomness :math:`\alpha` [N x N].
        Default: 0.3 off-diagonal.
    d : array_like, optional
        Temperature-dependent part of :math:`\alpha` [N x N]. Default: zeros.
    e : array_like, optional
        Coefficient of :math:`\ln T` in :math:`\tau` [N x N]. Default: zeros.
    f : array_like, optional
        Coefficient of :math:`T` in :math:`\tau` [N x N]. Default: zeros.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"hE": 0}

    def __init__(self, x, a, b=None, c=None, d=None, e=None, f=None):
        self.x = np.asarray(x, dtype=float)
        self.n = len(self.x)
        n = self.n

        self.a = np.asarray(a, dtype=float).reshape(n, n)
        self.b = np.zeros((n, n)) if b is None else np.asarray(b, dtype=float).reshape(n, n)
        self.e = np.zeros((n, n)) if e is None else np.asarray(e, dtype=float).reshape(n, n)
        self.f = np.zeros((n, n)) if f is None else np.asarray(f, dtype=float).reshape(n, n)

        if c is None:
            self.c = np.full((n, n), 0.3)
            np.fill_diagonal(self.c, 0.0)
        else:
            self.c = np.asarray(c, dtype=float).reshape(n, n)

        self.d = np.zeros((n, n)) if d is None else np.asarray(d, dtype=float).reshape(n, n)

        super().__init__(func=self._eval)

    def _eval(self, T):
        n = self.n
        x = self.x

        # tau and its derivative
        tau = self.a + self.b / T + self.e * np.log(T) + self.f * T
        tau_prime = -self.b / T**2 + self.e / T + self.f

        # alpha (S) and its derivative
        S = self.c + self.d * (T - 273.15)
        S_prime = self.d

        # G and its derivative
        G = np.exp(-S * tau)
        G_prime = -G * (S_prime * tau + S * tau_prime)

        # h^E = -RT^2 * sum_i x_i * (A'_i * B_i - A_i * B'_i) / B_i^2
        hE = 0.0
        for i in range(n):
            A_i = np.dot(x, G[:, i] * tau[:, i])
            A_prime_i = np.dot(x, G_prime[:, i] * tau[:, i] + G[:, i] * tau_prime[:, i])
            B_i = np.dot(x, G[:, i])
            B_prime_i = np.dot(x, G_prime[:, i])

            hE += x[i] * (A_prime_i * B_i - A_i * B_prime_i) / B_i**2

        hE = -R * T**2 * hE
        return hE


class ExcessEnthalpyUNIQUAC(Function):
    r"""Excess enthalpy from the UNIQUAC model (IK-CAPE Chapter 9.6.2).

    Computes the molar excess enthalpy :math:`h^E` of a liquid mixture
    using the UNIQUAC model. Only the residual (interaction) part
    contributes to the excess enthalpy since the combinatorial part
    depends only on composition, not temperature:

    .. math::

        h^E = R T^2 \sum_i q'_i x_i \left(
        \frac{A'_i}{A_i} + \sum_j F'_j
        \frac{\frac{\partial \tau_{i,j}}{\partial T} A_j
        - \tau_{i,j} A'_j}{A_j^2}
        \right)

    where :math:`A_i = \sum_j F'_j \tau_{j,i}` and
    :math:`\tau_{j,i} = \exp(a_{j,i} + b_{j,i}/T + c_{j,i}\ln T + d_{j,i}T)`.

    **Input port:** ``T`` -- temperature [K].

    **Output port:** ``hE`` -- molar excess enthalpy [J/mol].

    Parameters
    ----------
    x : array_like
        Mole fractions [N].
    r : array_like
        Van der Waals volume parameters [N].
    q : array_like
        Van der Waals surface area parameters [N].
    a : array_like
        Constant part of the :math:`\\tau` exponent [N x N].
    q_prime : array_like, optional
        Modified surface area for residual part [N]. Defaults to ``q``.
    b : array_like, optional
        Coefficient of :math:`1/T` [N x N]. Default: zeros.
    c : array_like, optional
        Coefficient of :math:`\\ln T` [N x N]. Default: zeros.
    d : array_like, optional
        Coefficient of :math:`T` [N x N]. Default: zeros.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"hE": 0}

    def __init__(self, x, r, q, a, q_prime=None, b=None, c=None, d=None):
        self.x = np.asarray(x, dtype=float)
        self.n = len(self.x)
        n = self.n

        self.r = np.asarray(r, dtype=float)
        self.q = np.asarray(q, dtype=float)
        self.q_prime = self.q.copy() if q_prime is None else np.asarray(q_prime, dtype=float)

        self.a = np.asarray(a, dtype=float).reshape(n, n)
        self.b_param = np.zeros((n, n)) if b is None else np.asarray(b, dtype=float).reshape(n, n)
        self.c_param = np.zeros((n, n)) if c is None else np.asarray(c, dtype=float).reshape(n, n)
        self.d_param = np.zeros((n, n)) if d is None else np.asarray(d, dtype=float).reshape(n, n)

        super().__init__(func=self._eval)

    def _eval(self, T):
        n = self.n
        x = self.x

        # tau and derivative
        exponent = self.a + self.b_param / T + self.c_param * np.log(T) + self.d_param * T
        tau = np.exp(exponent)
        tau_prime = tau * (-self.b_param / T**2 + self.c_param / T + self.d_param)

        # modified surface fractions F'
        Fp = self.q_prime * x / np.dot(self.q_prime, x)

        # A_i = sum_j F'_j * tau_j,i and A'_i = sum_j F'_j * tau'_j,i
        A_vec = np.dot(Fp, tau)       # shape (n,)
        A_prime = np.dot(Fp, tau_prime)  # shape (n,)

        # h^E = RT^2 * sum_i q'_i * x_i * (A'_i/A_i + sum_j F'_j * (tau'_ij*A_j - tau_ij*A'_j)/A_j^2)
        hE = 0.0
        for i in range(n):
            term1 = A_prime[i] / A_vec[i]

            term2 = 0.0
            for j in range(n):
                term2 += Fp[j] * (tau_prime[i, j] * A_vec[j]
                                   - tau[i, j] * A_prime[j]) / A_vec[j]**2

            hE += self.q_prime[i] * x[i] * (term1 + term2)

        return R * T**2 * hE


class ExcessEnthalpyWilson(Function):
    r"""Excess enthalpy from the Wilson model (IK-CAPE Chapter 9.6.3).

    Computes the molar excess enthalpy :math:`h^E` of a liquid mixture
    using the Wilson activity coefficient model. The Wilson model is
    suitable for completely miscible systems:

    .. math::

        \frac{g^E}{T} = -R \sum_i x_i \ln\!\left(\sum_j x_j \Lambda_{ij}\right)

    .. math::

        h^E = R T^2 \sum_i x_i \frac{\sigma'_i}{\sigma_i}

    where :math:`\sigma_i = \sum_j x_j \Lambda_{ij}` and
    :math:`\sigma'_i = \sum_j x_j \Lambda'_{ij}`, with:

    .. math::

        \Lambda'_{ij} = \Lambda_{ij}
        \left(-\frac{b_{ij}}{T^2} + \frac{c_{ij}}{T} + d_{ij}\right)

    **Input port:** ``T`` -- temperature [K].

    **Output port:** ``hE`` -- molar excess enthalpy [J/mol].

    Parameters
    ----------
    x : array_like
        Mole fractions [N].
    a : array_like
        Constant part of :math:`\ln \Lambda_{ij}` [N x N].
    b : array_like, optional
        Coefficient of :math:`1/T` [N x N]. Default: zeros.
    c : array_like, optional
        Coefficient of :math:`\ln T` [N x N]. Default: zeros.
    d : array_like, optional
        Coefficient of :math:`T` [N x N]. Default: zeros.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"hE": 0}

    def __init__(self, x, a, b=None, c=None, d=None):
        self.x = np.asarray(x, dtype=float)
        self.n = len(self.x)
        n = self.n

        self.a = np.asarray(a, dtype=float).reshape(n, n)
        self.b = np.zeros((n, n)) if b is None else np.asarray(b, dtype=float).reshape(n, n)
        self.c = np.zeros((n, n)) if c is None else np.asarray(c, dtype=float).reshape(n, n)
        self.d = np.zeros((n, n)) if d is None else np.asarray(d, dtype=float).reshape(n, n)

        super().__init__(func=self._eval)

    def _eval(self, T):
        n = self.n
        x = self.x

        # Lambda and its derivative
        Lam = np.exp(self.a + self.b / T + self.c * np.log(T) + self.d * T)
        Lam_prime = Lam * (-self.b / T**2 + self.c / T + self.d)

        # sigma_i = sum_j x_j * Lambda_ij, sigma'_i = sum_j x_j * Lambda'_ij
        sigma = np.dot(Lam, x)          # shape (n,) -- Lam[i,:] . x
        sigma_prime = np.dot(Lam_prime, x)

        # h^E = RT^2 * sum_i x_i * sigma'_i / sigma_i
        hE = R * T**2 * np.dot(x, sigma_prime / sigma)
        return hE


class ExcessEnthalpyRedlichKister(Function):
    r"""Excess enthalpy from the Redlich-Kister expansion (IK-CAPE Chapter 9.6.5).

    Computes the molar excess enthalpy using a Redlich-Kister polynomial
    expansion. This is a flexible, empirical model for representing binary
    excess properties. For a multi-component mixture, the excess enthalpy
    is computed as a sum of binary pair contributions:

    .. math::

        h^E = \frac{1}{2} \sum_i \sum_j h^E_{ij}

    Each binary pair contribution:

    .. math::

        h^E_{ij} = \frac{x_i x_j}{x_i + x_j}
        \left(A(T) x_d + B(T) x_d^2 + C(T) x_d^3 + \ldots\right)

    where :math:`x_d = x_i - x_j` and the coefficients :math:`A(T)`,
    :math:`B(T)`, ... are temperature-dependent polynomials:
    :math:`A(T) = a_0 + a_1 T + a_2 T^2 + \ldots`

    **Input port:** ``T`` -- temperature [K].

    **Output port:** ``hE`` -- molar excess enthalpy [J/mol].

    Parameters
    ----------
    x : array_like
        Mole fractions [N].
    coeffs : dict
        Redlich-Kister coefficients keyed by binary pair ``(i, j)`` as tuples.
        Each value is a list of polynomial coefficient arrays, one per
        Redlich-Kister term. Example for a single pair (0,1) with 3 terms::

            {(0, 1): [[a0, a1], [b0, b1], [c0]]}

        This gives A(T)=a0+a1*T, B(T)=b0+b1*T, C(T)=c0.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"hE": 0}

    def __init__(self, x, coeffs):
        self.x = np.asarray(x, dtype=float)
        self.n = len(self.x)
        self.coeffs = coeffs

        super().__init__(func=self._eval)

    def _eval(self, T):
        x = self.x
        hE = 0.0

        for (i, j), terms in self.coeffs.items():
            xi, xj = x[i], x[j]
            x_sum = xi + xj
            if x_sum < 1e-30:
                continue

            xd = xi - xj

            # evaluate each Redlich-Kister term
            pair_hE = 0.0
            xd_power = xd
            for poly_coeffs in terms:
                # temperature polynomial: c0 + c1*T + c2*T^2 + ...
                coeff_val = 0.0
                T_power = 1.0
                for c in poly_coeffs:
                    coeff_val += c * T_power
                    T_power *= T
                pair_hE += coeff_val * xd_power
                xd_power *= xd

            hE += xi * xj / x_sum * pair_hE

        return 0.5 * hE


# ENTHALPY DEPARTURE FROM EOS (9.7) =====================================================

class EnthalpyDepartureRKS(Function):
    r"""Isothermal enthalpy departure from the SRK EoS (IK-CAPE Chapter 9.7.1).

    Computes the departure of the real-gas enthalpy from the ideal-gas
    enthalpy at the same temperature using the Soave-Redlich-Kwong equation
    of state. This quantity is needed to convert between ideal-gas and
    real-gas thermodynamic properties:

    .. math::

        \Delta h = h(T,P) - h^{ig}(T)

    For a pure component:

    .. math::

        \Delta h_i = RT(Z_i - 1) + \frac{T\frac{da_i}{dT} - a_i}{b_i}
        \ln\!\left(1 + \frac{b_i P_i^s}{Z_i R T}\right)

    For a mixture:

    .. math::

        \Delta h_m = RT(Z_m - 1) + \frac{T\frac{da_m}{dT} - a_m}{b_m}
        \ln\!\left(1 + \frac{b_m P}{Z_m R T}\right)

    **Input ports:** ``T`` -- temperature [K], ``P`` -- pressure [Pa].

    **Output port:** ``dh`` -- enthalpy departure [J/mol].

    Parameters
    ----------
    Tc : float or array_like
        Critical temperature(s) [K].
    Pc : float or array_like
        Critical pressure(s) [Pa].
    omega : float or array_like
        Acentric factor(s) [-].
    x : array_like, optional
        Mole fractions [N]. Required for mixtures.
    k : array_like, optional
        Binary interaction parameters [N x N]. Default: zeros.
    phase : str, optional
        ``"vapor"`` (default) or ``"liquid"``.
    """

    input_port_labels = {"T": 0, "P": 1}
    output_port_labels = {"dh": 0}

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

        # SRK constants
        self.m = 0.48 + 1.574 * self.omega - 0.176 * self.omega**2
        self.a_c = 0.42748 * R**2 * self.Tc**2 / self.Pc
        self.b_i = 0.08664 * R * self.Tc / self.Pc

        super().__init__(func=self._eval)

    def _eval(self, T, P):
        nc = self.nc
        x = self.x

        # a_i(T) and da_i/dT
        sqrt_Tr = np.sqrt(T / self.Tc)
        alpha = (1 + self.m * (1 - sqrt_Tr))**2
        a_i = self.a_c * alpha

        # da_i/dT = a_c * 2*(1+m*(1-sqrt(T/Tc))) * m * (-1/(2*sqrt(Tc*T)))
        #         = -a_c * m * (1+m*(1-sqrt_Tr)) / sqrt(Tc * T)
        da_i_dT = -self.a_c * self.m * (1 + self.m * (1 - sqrt_Tr)) / np.sqrt(self.Tc * T)

        # mixing rules
        if nc == 1:
            a_m = a_i[0]
            b_m = self.b_i[0]
            da_m_dT = da_i_dT[0]
        else:
            a_m = 0.0
            da_m_dT = 0.0
            for i in range(nc):
                for j in range(nc):
                    a_ij = np.sqrt(a_i[i] * a_i[j]) * (1 - self.k[i, j])
                    a_m += x[i] * x[j] * a_ij
                    # d(a_ij)/dT using product rule on sqrt(a_i * a_j)
                    if a_i[i] * a_i[j] > 0:
                        da_ij_dT = (1 - self.k[i, j]) / (2 * np.sqrt(a_i[i] * a_i[j])) * (
                            a_i[i] * da_i_dT[j] + a_i[j] * da_i_dT[i])
                    else:
                        da_ij_dT = 0.0
                    da_m_dT += x[i] * x[j] * da_ij_dT
            b_m = np.dot(x, self.b_i)

        # solve SRK cubic
        A_m = a_m * P / (R**2 * T**2)
        B_m = b_m * P / (R * T)
        coeffs = [1, -1, A_m - B_m - B_m**2, -A_m * B_m]
        Z = _solve_cubic_eos(coeffs, self.phase)

        # enthalpy departure
        dh = R * T * (Z - 1) + (T * da_m_dT - a_m) / b_m * np.log(1 + b_m * P / (Z * R * T))
        return dh


class EnthalpyDeparturePR(Function):
    r"""Isothermal enthalpy departure from the Peng-Robinson EoS (IK-CAPE Chapter 9.7.2).

    Computes the departure of the real-gas enthalpy from the ideal-gas
    enthalpy using the Peng-Robinson equation of state:

    .. math::

        \Delta h_i = RT(Z_i - 1) - \frac{1}{2\sqrt{2}\,b_i}
        \left(a_i(T) - T\frac{\partial a_i}{\partial T}\right)
        \ln\frac{v_i + (1+\sqrt{2})b_i}{v_i + (1-\sqrt{2})b_i}

    For a mixture, subscript *i* is replaced by mixture quantities
    :math:`a_m, b_m, Z_m, v_m`.

    **Input ports:** ``T`` -- temperature [K], ``P`` -- pressure [Pa].

    **Output port:** ``dh`` -- enthalpy departure [J/mol].

    Parameters
    ----------
    Tc : float or array_like
        Critical temperature(s) [K].
    Pc : float or array_like
        Critical pressure(s) [Pa].
    omega : float or array_like
        Acentric factor(s) [-].
    x : array_like, optional
        Mole fractions [N]. Required for mixtures.
    k : array_like, optional
        Binary interaction parameters [N x N]. Default: zeros.
    phase : str, optional
        ``"vapor"`` (default) or ``"liquid"``.
    """

    input_port_labels = {"T": 0, "P": 1}
    output_port_labels = {"dh": 0}

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

        # PR constants
        self.m = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        self.a_c = 0.45724 * R**2 * self.Tc**2 / self.Pc
        self.b_i = 0.07780 * R * self.Tc / self.Pc

        super().__init__(func=self._eval)

    def _eval(self, T, P):
        nc = self.nc
        x = self.x
        sqrt2 = np.sqrt(2.0)

        # a_i(T) and da_i/dT
        sqrt_Tr = np.sqrt(T / self.Tc)
        alpha = (1 + self.m * (1 - sqrt_Tr))**2
        a_i = self.a_c * alpha
        da_i_dT = -self.a_c * self.m * (1 + self.m * (1 - sqrt_Tr)) / np.sqrt(self.Tc * T)

        # mixing rules
        if nc == 1:
            a_m = a_i[0]
            b_m = self.b_i[0]
            da_m_dT = da_i_dT[0]
        else:
            a_m = 0.0
            da_m_dT = 0.0
            for i in range(nc):
                for j in range(nc):
                    a_ij = np.sqrt(a_i[i] * a_i[j]) * (1 - self.k[i, j])
                    a_m += x[i] * x[j] * a_ij
                    if a_i[i] * a_i[j] > 0:
                        da_ij_dT = (1 - self.k[i, j]) / (2 * np.sqrt(a_i[i] * a_i[j])) * (
                            a_i[i] * da_i_dT[j] + a_i[j] * da_i_dT[i])
                    else:
                        da_ij_dT = 0.0
                    da_m_dT += x[i] * x[j] * da_ij_dT
            b_m = np.dot(x, self.b_i)

        # solve PR cubic
        A_m = a_m * P / (R**2 * T**2)
        B_m = b_m * P / (R * T)
        coeffs = [1,
                  -(1 - B_m),
                  A_m - 3 * B_m**2 - 2 * B_m,
                  -(A_m * B_m - B_m**2 - B_m**3)]
        Z = _solve_cubic_eos(coeffs, self.phase)
        v = Z * R * T / P

        # enthalpy departure (PR)
        log_term = np.log((v + (1 + sqrt2) * b_m) / (v + (1 - sqrt2) * b_m))
        dh = R * T * (Z - 1) - 1.0 / (2 * sqrt2 * b_m) * (a_m - T * da_m_dT) * log_term
        return dh
