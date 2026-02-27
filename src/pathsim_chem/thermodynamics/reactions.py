#########################################################################################
##
##                    IK-CAPE Chemical Reactions (Chapter 10)
##
##    Temperature-dependent equilibrium constants, kinetic rate constants,
##    and reaction rate expressions for chemical process simulation.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.function import Function


# CONSTANTS =============================================================================

R = 8.314462  # gas constant [J/(mol*K)]


# BLOCKS ================================================================================

class EquilibriumConstant(Function):
    r"""Temperature-dependent chemical equilibrium constant (IK-CAPE Chapter 10.1).

    Computes the equilibrium constant :math:`K_{eq}(T)` for a chemical
    reaction as a function of temperature using a six-parameter correlation.
    The equilibrium constant is used in equilibrium reactor models to
    determine the extent of reaction at thermodynamic equilibrium.

    It is related to the standard Gibbs energy of reaction:

    .. math::

        \Delta G^\circ = -RT \ln K_{eq}

    Temperature dependence:

    .. math::

        \ln K_{eq}(T) = a_0 + \frac{a_1}{T} + a_2 \ln T
        + a_3 T + a_4 T^2 + a_5 T^3

    The simplest form (van't Hoff) uses only :math:`a_0` and :math:`a_1`,
    where :math:`a_1 = -\Delta H^\circ / R`. Additional terms capture
    the temperature dependence of the heat of reaction via heat capacity.

    **Input port:** ``T`` -- temperature [K].

    **Output port:** ``Keq`` -- equilibrium constant [-] (dimensionless
    or in appropriate pressure/concentration units depending on convention).

    Parameters
    ----------
    a0 : float
        Constant term.
    a1 : float, optional
        Coefficient of :math:`1/T`. Related to the standard enthalpy of
        reaction. Default: 0.
    a2 : float, optional
        Coefficient of :math:`\ln T`. Related to heat capacity change.
        Default: 0.
    a3 : float, optional
        Coefficient of :math:`T`. Default: 0.
    a4 : float, optional
        Coefficient of :math:`T^2`. Default: 0.
    a5 : float, optional
        Coefficient of :math:`T^3`. Default: 0.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"Keq": 0}

    def __init__(self, a0, a1=0.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0):
        self.coeffs = (a0, a1, a2, a3, a4, a5)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3, a4, a5 = self.coeffs
        ln_K = a0 + a1 / T + a2 * np.log(T) + a3 * T + a4 * T**2 + a5 * T**3
        return np.exp(ln_K)


class KineticRateConstant(Function):
    r"""Temperature-dependent kinetic rate constant (IK-CAPE Chapter 10.2).

    Computes the reaction rate constant :math:`k(T)` for a kinetically
    controlled reaction using an extended Arrhenius-type correlation.
    This is the fundamental expression for reaction kinetics, describing
    how fast a chemical reaction proceeds at a given temperature.

    Two forms are supported:

    **Standard form** (4 parameters):

    .. math::

        \ln k(T) = a_0 + \frac{a_1}{T} + a_2 \ln T + a_3 T

    where :math:`a_0 = \ln A` (pre-exponential factor) and
    :math:`a_1 = -E_a/R` (activation energy).

    **Extended form** with separate forward and reverse constants,
    where the reverse constant can be derived from:

    .. math::

        k_{reverse} = k_{forward} / K_{eq}

    **Input port:** ``T`` -- temperature [K].

    **Output port:** ``k`` -- rate constant (units depend on reaction order).

    Parameters
    ----------
    a0 : float
        Constant term (:math:`\ln A` for Arrhenius).
    a1 : float, optional
        Coefficient of :math:`1/T` (:math:`-E_a/R`). Default: 0.
    a2 : float, optional
        Coefficient of :math:`\ln T` (power-law temperature exponent).
        Default: 0.
    a3 : float, optional
        Coefficient of :math:`T`. Default: 0.
    """

    input_port_labels = {"T": 0}
    output_port_labels = {"k": 0}

    def __init__(self, a0, a1=0.0, a2=0.0, a3=0.0):
        self.coeffs = (a0, a1, a2, a3)
        super().__init__(func=self._eval)

    def _eval(self, T):
        a0, a1, a2, a3 = self.coeffs
        ln_k = a0 + a1 / T + a2 * np.log(T) + a3 * T
        return np.exp(ln_k)


class PowerLawRate(Function):
    r"""Power-law reaction rate expression (IK-CAPE Chapter 10.2 KILM/KIVM).

    Computes the volumetric rate of reaction for a system of :math:`N_r`
    reactions involving :math:`N_c` species using power-law kinetics.
    Each reaction rate is the product of the rate constant and the
    concentrations (or mole fractions) raised to their stoichiometric powers:

    .. math::

        r_k = f_k \prod_i c_i^{\nu_{i,k}}

    For equilibrium-limited reactions, the rate is modified by an
    equilibrium driving-force term:

    .. math::

        r_k = f_k \prod_i c_i^{\nu_{i,k}^{fwd}}
        \left(1 - \frac{\prod_i c_i^{\nu_{i,k}}}{K_{eq,k}}\right)

    This block computes the **net production rate** of each species as a
    linear combination of individual reaction rates:

    .. math::

        R_i = \sum_k \nu_{i,k} \, r_k

    **Input ports:**

    - ``f_0``, ``f_1``, ... -- rate constants for each reaction (from
      :class:`KineticRateConstant` blocks upstream)
    - ``c_0``, ``c_1``, ... -- concentrations or mole fractions of species

    **Output ports:** ``R_0``, ``R_1``, ... -- net production rate for each
    species [mol/(m^3 s)] or [1/s] depending on the units of *f* and *c*.

    Parameters
    ----------
    nu : array_like
        Stoichiometric coefficient matrix [N_c x N_r]. Negative for
        reactants, positive for products.
    nu_fwd : array_like, optional
        Forward reaction orders [N_c x N_r]. If not given, the absolute
        values of the negative stoichiometric coefficients are used (i.e.,
        elementary reaction kinetics).
    Keq : array_like, optional
        Equilibrium constants [N_r]. If provided, an equilibrium driving
        force term is included. Default: None (irreversible reactions).
    """

    def __init__(self, nu, nu_fwd=None, Keq=None):
        self.nu = np.asarray(nu, dtype=float)
        self.n_species, self.n_rxn = self.nu.shape

        if nu_fwd is None:
            # elementary kinetics: forward orders = |negative stoich coeffs|
            self.nu_fwd = np.where(self.nu < 0, -self.nu, 0.0)
        else:
            self.nu_fwd = np.asarray(nu_fwd, dtype=float)

        self.Keq = None if Keq is None else np.asarray(Keq, dtype=float)

        super().__init__(func=self._eval)

    def _eval(self, *inputs):
        nr = self.n_rxn
        nc = self.n_species

        # first nr inputs are rate constants, next nc are concentrations
        f = np.array(inputs[:nr])
        c = np.array(inputs[nr:nr + nc])

        # compute reaction rates
        rates = np.zeros(nr)
        for k in range(nr):
            # forward rate
            rate_fwd = f[k] * np.prod(c ** self.nu_fwd[:, k])

            if self.Keq is not None:
                # equilibrium driving force: (1 - prod(c^nu) / Keq)
                Q = np.prod(c ** self.nu[:, k])
                rate_fwd *= (1 - Q / self.Keq[k])

            rates[k] = rate_fwd

        # net production rates for each species
        R = np.dot(self.nu, rates)
        return tuple(R)
