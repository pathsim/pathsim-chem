#########################################################################################
##
##                    IK-CAPE Equations of State (Chapter 7)
##
##    Cubic equations of state for computing molar volumes and
##    compressibility factors of gas and liquid phases.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function


# CONSTANTS =============================================================================

R = 8.314462  # gas constant [J/(mol*K)]


# HELPERS ===============================================================================

def _solve_cubic_eos(coeffs, phase="vapor"):
    """Solve cubic equation in Z and return the appropriate root.

    Parameters
    ----------
    coeffs : array_like
        Polynomial coefficients [1, c2, c1, c0] for Z^3 + c2*Z^2 + c1*Z + c0 = 0.
    phase : str
        "vapor" selects the largest real root, "liquid" the smallest positive real root.

    Returns
    -------
    float
        The selected compressibility factor Z.
    """
    roots = np.roots(coeffs)

    # keep only real roots (small imaginary part)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real

    # filter positive roots
    positive_roots = real_roots[real_roots > 0]

    if len(positive_roots) == 0:
        # fallback: take root with smallest imaginary part
        idx = np.argmin(np.abs(roots.imag))
        return roots[idx].real

    if phase == "vapor":
        return float(np.max(positive_roots))
    else:
        return float(np.min(positive_roots))


# BLOCKS ================================================================================

class PengRobinson(Function):
    r"""Peng-Robinson cubic equation of state (IK-CAPE Chapter 7.2).

    Computes the molar volume and compressibility factor of a pure component
    or mixture at a given temperature and pressure. The Peng-Robinson EoS
    (1976) is one of the most widely used cubic equations of state in
    chemical engineering, offering improved liquid density predictions
    over the original Redlich-Kwong equation. It is suitable for
    hydrocarbons, gases, and many industrial fluids.

    Supports both pure components (single Tc/Pc/omega) and mixtures
    (arrays of Tc/Pc/omega plus mole fractions and optional binary
    interaction parameters). The cubic equation is solved analytically
    and the ``phase`` parameter controls whether the vapor (largest Z)
    or liquid (smallest Z) root is selected.

    **Input ports:** ``T`` -- temperature [K], ``P`` -- pressure [Pa].

    **Output ports:** ``v`` -- molar volume [m^3/mol],
    ``z`` -- compressibility factor [-].

    Pressure-explicit form:

    .. math::

        P = \frac{RT}{v - b} - \frac{a(T)}{v(v+b) + b(v-b)}

    Solved as a cubic in compressibility factor :math:`Z = Pv/(RT)`:

    .. math::

        Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0

    where :math:`A = a_m P / (R^2 T^2)` and :math:`B = b_m P / (RT)`.

    Pure component parameters are computed from critical properties:

    .. math::

        a_i = 0.45724 \frac{R^2 T_{c,i}^2}{P_{c,i}} \alpha_i(T), \quad
        b_i = 0.07780 \frac{R T_{c,i}}{P_{c,i}}

        \alpha_i = \left[1 + m_i(1 - \sqrt{T/T_{c,i}})\right]^2, \quad
        m_i = 0.37464 + 1.54226\omega_i - 0.26992\omega_i^2

    Standard van der Waals one-fluid mixing rules:

    .. math::

        a_m = \sum_i \sum_j x_i x_j \sqrt{a_i a_j}(1 - k_{ij}), \quad
        b_m = \sum_i x_i b_i

    Parameters
    ----------
    Tc : float or array_like
        Critical temperature(s) [K]. Scalar for pure component, array for mixture.
    Pc : float or array_like
        Critical pressure(s) [Pa].
    omega : float or array_like
        Acentric factor(s) [-]. Characterizes deviation from spherical
        symmetry. Available in standard reference tables (e.g., DIPPR).
    x : array_like, optional
        Mole fractions [N]. Required for mixtures, omit for pure components.
    k : array_like, optional
        Binary interaction parameters [N x N]. Symmetric, with
        :math:`k_{ii} = 0`. Default: zero for all pairs.
    phase : str, optional
        Phase root selection: ``"vapor"`` (default) picks the largest Z root,
        ``"liquid"`` picks the smallest positive Z root.
    """

    input_port_labels = {"T": 0, "P": 1}
    output_port_labels = {"v": 0, "z": 1}

    def __init__(self, Tc, Pc, omega, x=None, k=None, phase="vapor"):
        self.Tc = np.atleast_1d(np.asarray(Tc, dtype=float))
        self.Pc = np.atleast_1d(np.asarray(Pc, dtype=float))
        self.omega = np.atleast_1d(np.asarray(omega, dtype=float))
        self.n = len(self.Tc)
        self.phase = phase

        if x is None:
            self.x = np.ones(1) if self.n == 1 else None
        else:
            self.x = np.asarray(x, dtype=float)

        if k is None:
            self.k = np.zeros((self.n, self.n))
        else:
            self.k = np.asarray(k, dtype=float).reshape(self.n, self.n)

        # constant pure component parameters
        self.m = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        self.a_c = 0.45724 * R**2 * self.Tc**2 / self.Pc
        self.b_i = 0.07780 * R * self.Tc / self.Pc

        super().__init__(func=self._eval)

    def _eval(self, T, P):
        n = self.n
        x = self.x

        # temperature-dependent alpha and a_i
        alpha = (1 + self.m * (1 - np.sqrt(T / self.Tc)))**2
        a_i = self.a_c * alpha

        # mixing rules
        if n == 1:
            a_m = a_i[0]
            b_m = self.b_i[0]
        else:
            a_m = 0.0
            for i in range(n):
                for j in range(n):
                    a_ij = np.sqrt(a_i[i] * a_i[j]) * (1 - self.k[i, j])
                    a_m += x[i] * x[j] * a_ij
            b_m = np.dot(x, self.b_i)

        # dimensionless parameters
        A = a_m * P / (R**2 * T**2)
        B = b_m * P / (R * T)

        # cubic in Z: Z^3 - (1-B)*Z^2 + (A-3B^2-2B)*Z - (AB-B^2-B^3) = 0
        coeffs = [1,
                  -(1 - B),
                  A - 3 * B**2 - 2 * B,
                  -(A * B - B**2 - B**3)]

        Z = _solve_cubic_eos(coeffs, self.phase)
        v = Z * R * T / P

        return v, Z


class RedlichKwongSoave(Function):
    r"""Soave-Redlich-Kwong cubic equation of state (IK-CAPE Chapter 7.1).

    Computes the molar volume and compressibility factor of a pure component
    or mixture at a given temperature and pressure. The SRK EoS (Soave, 1972)
    introduced a temperature-dependent attractive term to the original
    Redlich-Kwong equation, making it suitable for VLE calculations of
    hydrocarbons and light gases. It is widely used in natural gas
    and refinery process simulation.

    Has the same interface and mixing rules as :class:`PengRobinson`, but
    uses different constants and a slightly different pressure equation
    (the attractive term denominator is :math:`v(v+b)` instead of
    :math:`v(v+b)+b(v-b)`).

    **Input ports:** ``T`` -- temperature [K], ``P`` -- pressure [Pa].

    **Output ports:** ``v`` -- molar volume [m^3/mol],
    ``z`` -- compressibility factor [-].

    Pressure-explicit form:

    .. math::

        P = \frac{RT}{v - b} - \frac{a(T)}{v(v + b)}

    Solved as a cubic in :math:`Z`:

    .. math::

        Z^3 - Z^2 + (A - B - B^2)Z - AB = 0

    Pure component parameters:

    .. math::

        a_i = 0.42748 \frac{R^2 T_{c,i}^2}{P_{c,i}} \alpha_i(T), \quad
        b_i = 0.08664 \frac{R T_{c,i}}{P_{c,i}}

        \alpha_i = \left[1 + m_i(1 - \sqrt{T/T_{c,i}})\right]^2, \quad
        m_i = 0.48 + 1.574\omega_i - 0.176\omega_i^2

    Parameters
    ----------
    Tc : float or array_like
        Critical temperature(s) [K]. Scalar for pure component, array for mixture.
    Pc : float or array_like
        Critical pressure(s) [Pa].
    omega : float or array_like
        Acentric factor(s) [-].
    x : array_like, optional
        Mole fractions [N]. Required for mixtures, omit for pure components.
    k : array_like, optional
        Binary interaction parameters [N x N]. Default: zero for all pairs.
    phase : str, optional
        Phase root selection: ``"vapor"`` (default) or ``"liquid"``.
    """

    input_port_labels = {"T": 0, "P": 1}
    output_port_labels = {"v": 0, "z": 1}

    def __init__(self, Tc, Pc, omega, x=None, k=None, phase="vapor"):
        self.Tc = np.atleast_1d(np.asarray(Tc, dtype=float))
        self.Pc = np.atleast_1d(np.asarray(Pc, dtype=float))
        self.omega = np.atleast_1d(np.asarray(omega, dtype=float))
        self.n = len(self.Tc)
        self.phase = phase

        if x is None:
            self.x = np.ones(1) if self.n == 1 else None
        else:
            self.x = np.asarray(x, dtype=float)

        if k is None:
            self.k = np.zeros((self.n, self.n))
        else:
            self.k = np.asarray(k, dtype=float).reshape(self.n, self.n)

        # constant pure component parameters
        self.m = 0.48 + 1.574 * self.omega - 0.176 * self.omega**2
        self.a_c = 0.42748 * R**2 * self.Tc**2 / self.Pc
        self.b_i = 0.08664 * R * self.Tc / self.Pc

        super().__init__(func=self._eval)

    def _eval(self, T, P):
        n = self.n
        x = self.x

        # temperature-dependent alpha and a_i
        alpha = (1 + self.m * (1 - np.sqrt(T / self.Tc)))**2
        a_i = self.a_c * alpha

        # mixing rules
        if n == 1:
            a_m = a_i[0]
            b_m = self.b_i[0]
        else:
            a_m = 0.0
            for i in range(n):
                for j in range(n):
                    a_ij = np.sqrt(a_i[i] * a_i[j]) * (1 - self.k[i, j])
                    a_m += x[i] * x[j] * a_ij
            b_m = np.dot(x, self.b_i)

        # dimensionless parameters
        A = a_m * P / (R**2 * T**2)
        B = b_m * P / (R * T)

        # cubic in Z: Z^3 - Z^2 + (A-B-B^2)*Z - AB = 0
        coeffs = [1, -1, A - B - B**2, -A * B]

        Z = _solve_cubic_eos(coeffs, self.phase)
        v = Z * R * T / P

        return v, Z
