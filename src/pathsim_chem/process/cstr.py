#########################################################################################
##
##                    Continuous Stirred-Tank Reactor (CSTR) Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

# CONSTANTS =============================================================================

R_GAS = 8.314  # universal gas constant [J/(mol·K)]

# BLOCKS ================================================================================

class CSTR(DynamicalSystem):
    """Continuous stirred-tank reactor with Arrhenius kinetics and energy balance.

    Models a well-mixed tank where a single reaction A -> products occurs with
    nth-order kinetics. The reaction rate follows the Arrhenius temperature
    dependence. An external coolant jacket provides or removes heat.

    Mathematical Formulation
    -------------------------
    The state vector is :math:`[C_A, T]` where :math:`C_A` is the concentration
    of species A and :math:`T` is the reactor temperature.

    .. math::

        \\frac{dC_A}{dt} = \\frac{C_{A,in} - C_A}{\\tau} - k(T) \\, C_A^n

    .. math::

        \\frac{dT}{dt} = \\frac{T_{in} - T}{\\tau}
            + \\frac{(-\\Delta H_{rxn})}{\\rho \\, C_p} \\, k(T) \\, C_A^n
            + \\frac{UA}{\\rho \\, C_p \\, V} \\, (T_c - T)

    where the Arrhenius rate constant is:

    .. math::

        k(T) = k_0 \\, \\exp\\!\\left(-\\frac{E_a}{R \\, T}\\right)

    and the residence time is :math:`\\tau = V / F`.

    Parameters
    ----------
    V : float
        Reactor volume [m³].
    F : float
        Volumetric flow rate [m³/s].
    k0 : float
        Pre-exponential Arrhenius factor [1/s for n=1, (m³/mol)^(n-1)/s].
    Ea : float
        Activation energy [J/mol].
    n : float
        Reaction order with respect to species A [-].
    dH_rxn : float
        Heat of reaction [J/mol]. Negative for exothermic reactions.
    rho : float
        Fluid density [kg/m³].
    Cp : float
        Fluid heat capacity [J/(kg·K)].
    UA : float
        Overall heat transfer coefficient times area [W/K].
    C_A0 : float
        Initial concentration of A [mol/m³].
    T0 : float
        Initial reactor temperature [K].
    """

    input_port_labels = {
        "C_in": 0,
        "T_in": 1,
        "T_c":  2,
    }

    output_port_labels = {
        "C_out": 0,
        "T_out": 1,
    }

    def __init__(self, V=1.0, F=0.1, k0=1e6, Ea=50000.0, n=1.0,
                 dH_rxn=-50000.0, rho=1000.0, Cp=4184.0, UA=500.0,
                 C_A0=0.0, T0=300.0):

        # input validation
        if V <= 0:
            raise ValueError(f"'V' must be positive but is {V}")
        if F <= 0:
            raise ValueError(f"'F' must be positive but is {F}")
        if rho <= 0:
            raise ValueError(f"'rho' must be positive but is {rho}")
        if Cp <= 0:
            raise ValueError(f"'Cp' must be positive but is {Cp}")

        # store parameters
        self.V = V
        self.F = F
        self.k0 = k0
        self.Ea = Ea
        self.n = n
        self.dH_rxn = dH_rxn
        self.rho = rho
        self.Cp = Cp
        self.UA = UA

        # derived
        self.tau = V / F

        # rhs of CSTR ode system
        def _fn_d(x, u, t):
            C_A, T = x
            C_A_in, T_in, T_c = u

            tau = self.V / self.F
            k = self.k0 * np.exp(-self.Ea / (R_GAS * T))
            r = k * C_A**self.n

            dC_A = (C_A_in - C_A) / tau - r
            dT = ((T_in - T) / tau
                  + (-self.dH_rxn) / (self.rho * self.Cp) * r
                  + self.UA / (self.rho * self.Cp * self.V) * (T_c - T))

            return np.array([dC_A, dT])

        # jacobian of rhs wrt state [C_A, T]
        def _jc_d(x, u, t):
            C_A, T = x
            tau = self.V / self.F
            k = self.k0 * np.exp(-self.Ea / (R_GAS * T))

            # partial derivatives
            dk_dT = k * self.Ea / (R_GAS * T**2)

            # dr/dC_A and dr/dT
            dr_dCA = k * self.n * C_A**(self.n - 1) if C_A > 0 else 0.0
            dr_dT = dk_dT * C_A**self.n

            rcp = (-self.dH_rxn) / (self.rho * self.Cp)
            ua_term = self.UA / (self.rho * self.Cp * self.V)

            J = np.array([
                [-1.0/tau - dr_dCA,         -dr_dT],
                [rcp * dr_dCA,  -1.0/tau + rcp * dr_dT - ua_term]
            ])
            return J

        # output function: well-mixed => outlet = state
        def _fn_a(x, u, t):
            return x.copy()

        super().__init__(
            func_dyn=_fn_d,
            jac_dyn=_jc_d,
            func_alg=_fn_a,
            initial_value=np.array([C_A0, T0]),
        )
