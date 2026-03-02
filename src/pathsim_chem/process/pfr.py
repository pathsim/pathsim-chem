#########################################################################################
##
##                    Plug Flow Reactor (PFR) Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

# CONSTANTS =============================================================================

R_GAS = 8.314  # universal gas constant [J/(mol·K)]

# BLOCKS ================================================================================

class PFR(DynamicalSystem):
    """Plug flow reactor with Arrhenius kinetics and energy balance.

    Discretized tubular reactor divided into N cells along its length.
    Each cell has concentration and temperature states with nth-order
    kinetics and an energy balance including heat of reaction.

    Mathematical Formulation
    -------------------------
    For each cell :math:`i = 1, \\ldots, N`:

    .. math::

        \\frac{dC_i}{dt} = \\frac{F}{V_{cell}} (C_{i-1} - C_i) - k(T_i) \\, C_i^n

    .. math::

        \\frac{dT_i}{dt} = \\frac{F}{V_{cell}} (T_{i-1} - T_i)
            + \\frac{(-\\Delta H_{rxn})}{\\rho \\, C_p} \\, k(T_i) \\, C_i^n

    where the Arrhenius rate constant is:

    .. math::

        k(T) = k_0 \\, \\exp\\!\\left(-\\frac{E_a}{R \\, T}\\right)

    The state vector is ordered as
    :math:`[C_1, T_1, C_2, T_2, \\ldots, C_N, T_N]`.

    Parameters
    ----------
    N_cells : int
        Number of discretization cells [-].
    V : float
        Total reactor volume [m³].
    F : float
        Volumetric flow rate [m³/s].
    k0 : float
        Pre-exponential Arrhenius factor [1/s for n=1].
    Ea : float
        Activation energy [J/mol].
    n : float
        Reaction order [-].
    dH_rxn : float
        Heat of reaction [J/mol]. Negative for exothermic.
    rho : float
        Fluid density [kg/m³].
    Cp : float
        Fluid heat capacity [J/(kg·K)].
    C0 : float
        Initial concentration [mol/m³].
    T0 : float
        Initial temperature [K].
    """

    input_port_labels = {
        "C_in": 0,
        "T_in": 1,
    }

    def __init__(self, N_cells=5, V=1.0, F=0.1, k0=1e6, Ea=50000.0, n=1.0,
                 dH_rxn=-50000.0, rho=1000.0, Cp=4184.0,
                 C0=0.0, T0=300.0):

        # input validation
        if N_cells < 1:
            raise ValueError(f"'N_cells' must be >= 1 but is {N_cells}")
        if V <= 0:
            raise ValueError(f"'V' must be positive but is {V}")
        if F <= 0:
            raise ValueError(f"'F' must be positive but is {F}")
        if rho <= 0:
            raise ValueError(f"'rho' must be positive but is {rho}")
        if Cp <= 0:
            raise ValueError(f"'Cp' must be positive but is {Cp}")

        # store parameters
        self.N_cells = int(N_cells)
        self.V = V
        self.F = F
        self.k0 = k0
        self.Ea = Ea
        self.n = n
        self.dH_rxn = dH_rxn
        self.rho = rho
        self.Cp = Cp

        N = self.N_cells

        # dynamic output port labels: outlets + per-cell profiles
        self.output_port_labels = {"C_out": 0, "T_out": 1}
        for i in range(N):
            self.output_port_labels[f"C_{i+1}"] = 2 + i
            self.output_port_labels[f"T_{i+1}"] = 2 + N + i

        # initial state: interleaved [C_1, T_1, C_2, T_2, ...]
        x0 = np.empty(2 * N)
        x0[0::2] = C0
        x0[1::2] = T0

        # ensure u has expected 2 elements (handles framework probing)
        def _pad_u(u):
            u = np.atleast_1d(u)
            if len(u) < 2:
                padded = np.zeros(2)
                padded[:len(u)] = u
                return padded
            return u

        # rhs of PFR ode (vectorized)
        def _fn_d(x, u, t):
            u = _pad_u(u)
            C_in, T_in = u
            N = self.N_cells

            V_cell = self.V / N
            f_flow = self.F / V_cell
            rcp = (-self.dH_rxn) / (self.rho * self.Cp)

            C = x[0::2]
            T = x[1::2]

            # upstream values with boundary conditions
            C_prev = np.empty(N)
            C_prev[0] = C_in
            C_prev[1:] = C[:-1]

            T_prev = np.empty(N)
            T_prev[0] = T_in
            T_prev[1:] = T[:-1]

            # Arrhenius rate per cell
            k = self.k0 * np.exp(-self.Ea / (R_GAS * T))
            r = k * np.abs(C)**self.n  # abs for numerical safety

            dx = np.empty(2 * N)
            dx[0::2] = f_flow * (C_prev - C) - r
            dx[1::2] = f_flow * (T_prev - T) + rcp * r

            return dx

        # analytical jacobian (block-tridiagonal structure)
        def _jc_d(x, u, t):
            N = self.N_cells

            V_cell = self.V / N
            f_flow = self.F / V_cell
            rcp = (-self.dH_rxn) / (self.rho * self.Cp)

            C = x[0::2]
            T = x[1::2]

            k = self.k0 * np.exp(-self.Ea / (R_GAS * T))
            dk_dT = k * self.Ea / (R_GAS * T**2)

            dim = 2 * N
            J = np.zeros((dim, dim))

            for i in range(N):
                ci = 2 * i      # concentration index
                ti = 2 * i + 1  # temperature index

                C_i = max(abs(C[i]), 1e-30)
                dr_dC = k[i] * self.n * C_i**(self.n - 1) if C_i > 0 else 0.0
                dr_dT = dk_dT[i] * C_i**self.n

                # dC_i/dC_i, dC_i/dT_i
                J[ci, ci] = -f_flow - dr_dC
                J[ci, ti] = -dr_dT

                # dT_i/dC_i, dT_i/dT_i
                J[ti, ci] = rcp * dr_dC
                J[ti, ti] = -f_flow + rcp * dr_dT

                # upstream coupling: dC_i/dC_{i-1}, dT_i/dT_{i-1}
                if i > 0:
                    J[ci, 2*(i-1)] = f_flow
                    J[ti, 2*(i-1) + 1] = f_flow

            return J

        # output: outlets + per-cell profiles
        # [C_out, T_out, C_1..C_N, T_1..T_N]
        def _fn_a(x, u, t):
            N = self.N_cells
            out = np.empty(2 + 2 * N)
            out[0] = x[2 * (N - 1)]       # C_out
            out[1] = x[2 * (N - 1) + 1]   # T_out
            out[2:N + 2] = x[0::2]        # concentration profile
            out[N + 2:] = x[1::2]         # temperature profile
            return out

        super().__init__(
            func_dyn=_fn_d,
            jac_dyn=_jc_d,
            func_alg=_fn_a,
            initial_value=x0,
        )
