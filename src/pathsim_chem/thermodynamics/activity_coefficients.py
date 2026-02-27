#########################################################################################
##
##                  IK-CAPE Activity Coefficient Models (Chapter 4)
##
##    Models for computing liquid-phase activity coefficients from
##    composition and temperature.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function


# CONSTANTS =============================================================================

R = 8.314462  # gas constant [J/(mol*K)]


# BLOCKS ================================================================================

class NRTL(Function):
    r"""Non-Random Two-Liquid activity coefficient model (IK-CAPE Chapter 4.1).

    Computes liquid-phase activity coefficients for a multi-component mixture
    at a given temperature. The NRTL model is widely used for strongly
    non-ideal liquid mixtures, including partially miscible systems.
    It is especially suitable for polar and associating systems such as
    alcohol-water, amine-water, and organic acid mixtures.

    The model uses binary interaction parameters :math:`\tau_{ij}` and
    non-randomness parameters :math:`\alpha_{ij}` that are typically fitted
    to experimental VLE data.

    **Input port:** ``T`` -- temperature [K].

    **Output ports:** ``out_0``, ``out_1``, ... -- activity coefficients
    :math:`\gamma_i` for each component (one per mole fraction entry).

    .. math::

        \ln \gamma_i = \frac{\sum_j \tau_{ji} G_{ji} x_j}{\sum_k G_{ki} x_k}
        + \sum_j \frac{x_j G_{ij}}{\sum_k G_{kj} x_k}
        \left( \tau_{ij} - \frac{\sum_l \tau_{lj} G_{lj} x_l}{\sum_k G_{kj} x_k} \right)

    where:

    .. math::

        G_{ji} = \exp(-\alpha_{ji} \, \tau_{ji})

        \tau_{ji} = a_{ji} + b_{ji}/T + e_{ji} \ln T + f_{ji} T

        \alpha_{ji} = c_{ji} + d_{ji} (T - 273.15)

    Parameters
    ----------
    x : array_like
        Mole fractions [N]. Fixed mixture composition.
    a : array_like
        Constant part of the energy interaction parameter :math:`\tau` [N x N].
        Diagonal elements should be zero. For constant (temperature-independent)
        tau values, set ``a=tau`` and leave ``b``, ``e``, ``f`` as zero.
    b : array_like, optional
        Coefficient of :math:`1/T` in :math:`\tau_{ji}` [N x N]. Default: zeros.
    c : array_like, optional
        Constant part of the non-randomness parameter :math:`\alpha` [N x N].
        Default: 0.3 for all off-diagonal pairs (a common starting value).
    d : array_like, optional
        Temperature-dependent part of :math:`\alpha` [N x N]. Default: zeros.
    e : array_like, optional
        Coefficient of :math:`\ln T` in :math:`\tau_{ji}` [N x N]. Default: zeros.
    f : array_like, optional
        Coefficient of :math:`T` in :math:`\tau_{ji}` [N x N]. Default: zeros.
    """

    input_port_labels = {"T": 0}

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

        # temperature-dependent parameters
        tau = self.a + self.b / T + self.e * np.log(T) + self.f * T
        alpha = self.c + self.d * (T - 273.15)
        G = np.exp(-alpha * tau)

        ln_gamma = np.zeros(n)
        for i in range(n):
            # denominators: sum_k G_ki * x_k for each column
            den_i = np.dot(G[:, i], x)

            # first term
            num1 = np.dot(tau[:, i] * G[:, i], x)
            term1 = num1 / den_i

            # second term
            term2 = 0.0
            for j in range(n):
                den_j = np.dot(G[:, j], x)
                num_inner = np.dot(tau[:, j] * G[:, j], x)
                term2 += (x[j] * G[i, j] / den_j) * (tau[i, j] - num_inner / den_j)

            ln_gamma[i] = term1 + term2

        return tuple(np.exp(ln_gamma))


class Wilson(Function):
    r"""Wilson activity coefficient model (IK-CAPE Chapter 4.3).

    Computes liquid-phase activity coefficients for a multi-component mixture.
    The Wilson model is well suited for completely miscible systems with
    moderate non-ideality, such as hydrocarbon mixtures and many
    organic-organic systems. It cannot represent liquid-liquid phase splitting.

    The model uses binary interaction parameters :math:`\Lambda_{ij}` related
    to molar volume ratios and energy differences between molecular pairs.

    **Input port:** ``T`` -- temperature [K].

    **Output ports:** ``out_0``, ``out_1``, ... -- activity coefficients
    :math:`\gamma_i` for each component.

    .. math::

        \ln \gamma_i = 1 - \ln\!\left(\sum_j x_j \Lambda_{ij}\right)
        - \sum_k \frac{x_k \Lambda_{ki}}{\sum_j x_j \Lambda_{kj}}

    where:

    .. math::

        \Lambda_{ij} = \exp(a_{ij} + b_{ij}/T + c_{ij} \ln T + d_{ij} T)

    Parameters
    ----------
    x : array_like
        Mole fractions [N]. Fixed mixture composition.
    a : array_like
        Constant part of :math:`\ln \Lambda_{ij}` [N x N]. Diagonal should
        be zero so that :math:`\Lambda_{ii} = 1`.
    b : array_like, optional
        Coefficient of :math:`1/T` in :math:`\ln \Lambda_{ij}` [N x N].
        Default: zeros.
    c : array_like, optional
        Coefficient of :math:`\ln T` in :math:`\ln \Lambda_{ij}` [N x N].
        Default: zeros.
    d : array_like, optional
        Coefficient of :math:`T` in :math:`\ln \Lambda_{ij}` [N x N].
        Default: zeros.
    """

    input_port_labels = {"T": 0}

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

        # temperature-dependent Lambda
        Lam = np.exp(self.a + self.b / T + self.c * np.log(T) + self.d * T)

        ln_gamma = np.zeros(n)
        for i in range(n):
            # sum_j x_j * Lambda_ij
            s1 = np.dot(x, Lam[i, :])

            # sum_k x_k * Lambda_ki / sum_j x_j * Lambda_kj
            s2 = 0.0
            for k in range(n):
                den_k = np.dot(x, Lam[k, :])
                s2 += x[k] * Lam[k, i] / den_k

            ln_gamma[i] = 1.0 - np.log(s1) - s2

        return tuple(np.exp(ln_gamma))


class UNIQUAC(Function):
    r"""Universal Quasi-Chemical activity coefficient model (IK-CAPE Chapter 4.2).

    Computes liquid-phase activity coefficients by combining a combinatorial
    contribution (based on molecular size and shape via van der Waals volume
    ``r`` and surface area ``q`` parameters) with a residual contribution
    (based on intermolecular interactions via temperature-dependent
    :math:`\tau_{ij}` parameters). UNIQUAC can handle strongly non-ideal
    systems including partially miscible liquids and forms the theoretical
    basis of the UNIFAC group-contribution method.

    **Input port:** ``T`` -- temperature [K].

    **Output ports:** ``out_0``, ``out_1``, ... -- activity coefficients
    :math:`\gamma_i` for each component.

    The activity coefficient is split into combinatorial and residual parts:

    .. math::

        \ln \gamma_i = \ln \gamma_i^C + \ln \gamma_i^R

    Combinatorial part (molecular size/shape effects):

    .. math::

        \ln \gamma_i^C = \ln\frac{V_i}{x_i} + \frac{z}{2} q_i \ln\frac{F_i}{V_i}
        + l_i - \frac{V_i}{x_i} \sum_j x_j l_j

    Residual part (intermolecular energy interactions):

    .. math::

        \ln \gamma_i^R = q'_i \left[1 - \ln\!\left(\sum_j F'_j \tau_{ji}\right)
        - \sum_j \frac{F'_j \tau_{ij}}{\sum_k F'_k \tau_{kj}} \right]

    Parameters
    ----------
    x : array_like
        Mole fractions [N]. Fixed mixture composition.
    r : array_like
        Van der Waals volume (size) parameters [N], from UNIFAC tables
        or fitted to data. Example: ethanol=2.1055, water=0.92.
    q : array_like
        Van der Waals surface area parameters [N]. Example: ethanol=1.972,
        water=1.4.
    a : array_like
        Constant part of the interaction parameter exponent [N x N].
        Diagonal should be zero. The interaction parameter is computed as
        :math:`\tau_{ji} = \exp(a_{ji} + b_{ji}/T + c_{ji} \ln T + d_{ji} T)`.
    q_prime : array_like, optional
        Modified surface area for the residual part [N]. Used in some
        formulations (e.g., for water/alcohol systems). Defaults to ``q``.
    b : array_like, optional
        Coefficient of :math:`1/T` in the :math:`\tau` exponent [N x N].
        Default: zeros.
    c : array_like, optional
        Coefficient of :math:`\ln T` in the :math:`\tau` exponent [N x N].
        Default: zeros.
    d : array_like, optional
        Coefficient of :math:`T` in the :math:`\tau` exponent [N x N].
        Default: zeros.
    z : float, optional
        Lattice coordination number (default 10, the standard value).
    """

    input_port_labels = {"T": 0}

    def __init__(self, x, r, q, a, q_prime=None, b=None, c=None, d=None, z=10):
        self.x = np.asarray(x, dtype=float)
        self.n = len(self.x)
        n = self.n

        self.r = np.asarray(r, dtype=float)
        self.q = np.asarray(q, dtype=float)
        self.q_prime = self.q.copy() if q_prime is None else np.asarray(q_prime, dtype=float)
        self.z = z

        self.a = np.asarray(a, dtype=float).reshape(n, n)
        self.b_param = np.zeros((n, n)) if b is None else np.asarray(b, dtype=float).reshape(n, n)
        self.c_param = np.zeros((n, n)) if c is None else np.asarray(c, dtype=float).reshape(n, n)
        self.d_param = np.zeros((n, n)) if d is None else np.asarray(d, dtype=float).reshape(n, n)

        # l_i = z/2 * (r_i - q_i) - (r_i - 1)
        self.l = self.z / 2 * (self.r - self.q) - (self.r - 1)

        super().__init__(func=self._eval)

    def _eval(self, T):
        n = self.n
        x = self.x

        # temperature-dependent interaction parameters
        tau = np.exp(self.a + self.b_param / T + self.c_param * np.log(T) + self.d_param * T)

        # volume and surface fractions
        V = self.r * x / np.dot(self.r, x)
        F = self.q * x / np.dot(self.q, x)
        Fp = self.q_prime * x / np.dot(self.q_prime, x)

        # combinatorial part
        ln_gamma_C = np.zeros(n)
        sum_xl = np.dot(x, self.l)
        for i in range(n):
            ln_gamma_C[i] = (np.log(V[i] / x[i])
                             + self.z / 2 * self.q[i] * np.log(F[i] / V[i])
                             + self.l[i] - V[i] / x[i] * sum_xl)

        # residual part
        ln_gamma_R = np.zeros(n)
        for i in range(n):
            s1 = np.dot(Fp, tau[:, i])  # sum_j F'_j * tau_ji

            s2 = 0.0
            for j in range(n):
                den = np.dot(Fp, tau[:, j])  # sum_k F'_k * tau_kj
                s2 += Fp[j] * tau[i, j] / den

            ln_gamma_R[i] = self.q_prime[i] * (1 - np.log(s1) - s2)

        ln_gamma = ln_gamma_C + ln_gamma_R
        return tuple(np.exp(ln_gamma))


class FloryHuggins(Function):
    r"""Flory-Huggins activity coefficient model (IK-CAPE Chapter 4.4).

    Computes liquid-phase activity coefficients for polymer-solvent systems.
    The Flory-Huggins model accounts for the large size difference between
    polymer and solvent molecules by using volume fractions instead of mole
    fractions. It is the standard model for polymer solutions and is suitable
    for systems with one solvent and one or more dissolved polymer species.

    Component 1 is always the **solvent** (with segment number :math:`r_1 = 1`).
    The remaining components are polymer species with segment numbers
    :math:`r_{2i}` that represent the ratio of polymer to solvent molar volume.

    **Input port:** ``T`` -- temperature [K].

    **Output ports:** ``out_0``, ``out_1``, ... -- activity coefficients
    :math:`\gamma_i` for each component (solvent first, then polymers).

    The model uses volume fractions and the Flory-Huggins interaction
    parameter :math:`\chi`:

    .. math::

        \phi_1 = \frac{x_1}{\bar{r}}, \quad
        \phi_{2i} = \frac{r_{2i} x_{2i}}{\bar{r}}, \quad
        \bar{r} = x_1 + \sum_i r_{2i} x_{2i}

    Solvent activity coefficient:

    .. math::

        \ln \gamma_1 = 1 + \ln\frac{1}{\bar{r}} - \frac{1}{\bar{r}}
        + (1 - \phi_1) \sum_i \phi_{2i} \chi_{1,2i}

    Polymer activity coefficient for species k:

    .. math::

        \ln \gamma_{2k} = 1 + \ln\frac{r_{2k}}{\bar{r}} - \frac{r_{2k}}{\bar{r}}
        + \phi_1 r_{2k} \left( \chi_{1,2k}
        - \sum_i \phi_{2i} \chi_{1,2i} \right)

    Temperature-dependent interaction parameter:

    .. math::

        \chi_{1,2i} = \chi^0_{1,2i} + \frac{\chi^1_{1,2i}}{T}

    Parameters
    ----------
    x : array_like
        Mole fractions [N] where ``x[0]`` is the solvent and ``x[1:]``
        are the polymer species.
    r : array_like
        Segment numbers [N-1] for each polymer species. These represent
        the ratio of the polymer molar volume to the solvent molar volume.
    chi0 : array_like
        Constant part of the Flory-Huggins interaction parameter [N-1].
    chi1 : array_like, optional
        Temperature-dependent part of :math:`\chi` [N-1]. Default: zeros.
    """

    input_port_labels = {"T": 0}

    def __init__(self, x, r, chi0, chi1=None):
        self.x = np.asarray(x, dtype=float)
        self.n = len(self.x)
        n_poly = self.n - 1

        self.r_poly = np.asarray(r, dtype=float)
        self.chi0 = np.asarray(chi0, dtype=float)
        self.chi1 = np.zeros(n_poly) if chi1 is None else np.asarray(chi1, dtype=float)

        super().__init__(func=self._eval)

    def _eval(self, T):
        x = self.x
        x1 = x[0]
        x2 = x[1:]

        # interaction parameters
        chi = self.chi0 + self.chi1 / T

        # average segment number (r_solvent = 1)
        r_bar = x1 + np.dot(self.r_poly, x2)

        # volume fractions
        phi1 = x1 / r_bar
        phi2 = self.r_poly * x2 / r_bar

        # weighted interaction sum
        chi_sum = np.dot(phi2, chi)

        # solvent activity coefficient
        ln_gamma1 = 1.0 + np.log(1.0 / r_bar) - 1.0 / r_bar + (1.0 - phi1) * chi_sum

        # polymer activity coefficients
        ln_gamma2 = np.zeros(len(x2))
        for k in range(len(x2)):
            ln_gamma2[k] = (1.0 + np.log(self.r_poly[k] / r_bar)
                            - self.r_poly[k] / r_bar
                            + phi1 * self.r_poly[k] * (chi[k] - chi_sum))

        ln_gamma = np.concatenate(([ln_gamma1], ln_gamma2))
        return tuple(np.exp(ln_gamma))
