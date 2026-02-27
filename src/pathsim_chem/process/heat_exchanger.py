#########################################################################################
##
##                Counter-Current Shell & Tube Heat Exchanger Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

# BLOCKS ================================================================================

class HeatExchanger(DynamicalSystem):
    """Counter-current shell and tube heat exchanger with discretized cells.

    The exchanger is divided into N cells along its length. The hot stream
    flows from cell 1 to N, the cold stream flows from cell N to 1
    (counter-current). Each cell exchanges heat proportional to the local
    temperature difference.

    Mathematical Formulation
    -------------------------
    For each cell :math:`i = 1, \\ldots, N`:

    .. math::

        \\frac{dT_{h,i}}{dt} = \\frac{F_h}{V_{cell,h}} (T_{h,i-1} - T_{h,i})
            - \\frac{UA_{cell}}{\\rho_h C_{p,h} V_{cell,h}} (T_{h,i} - T_{c,i})

    .. math::

        \\frac{dT_{c,i}}{dt} = \\frac{F_c}{V_{cell,c}} (T_{c,i+1} - T_{c,i})
            + \\frac{UA_{cell}}{\\rho_c C_{p,c} V_{cell,c}} (T_{h,i} - T_{c,i})

    where :math:`T_{h,0} = T_{h,in}` and :math:`T_{c,N+1} = T_{c,in}`.

    The state vector is ordered as
    :math:`[T_{h,1}, T_{c,1}, T_{h,2}, T_{c,2}, \\ldots, T_{h,N}, T_{c,N}]`.

    Parameters
    ----------
    N_cells : int
        Number of discretization cells along the exchanger [-].
    F_h : float
        Hot stream volumetric flow rate [m³/s].
    F_c : float
        Cold stream volumetric flow rate [m³/s].
    V_h : float
        Total hot-side volume [m³].
    V_c : float
        Total cold-side volume [m³].
    UA : float
        Total overall heat transfer coefficient times area [W/K].
    rho_h : float
        Hot stream density [kg/m³].
    Cp_h : float
        Hot stream heat capacity [J/(kg·K)].
    rho_c : float
        Cold stream density [kg/m³].
    Cp_c : float
        Cold stream heat capacity [J/(kg·K)].
    T_h0 : float
        Initial hot-side temperature [K].
    T_c0 : float
        Initial cold-side temperature [K].
    """

    input_port_labels = {
        "T_h_in": 0,
        "T_c_in": 1,
    }

    output_port_labels = {
        "T_h_out": 0,
        "T_c_out": 1,
    }

    def __init__(self, N_cells=5, F_h=0.1, F_c=0.1, V_h=0.5, V_c=0.5,
                 UA=500.0, rho_h=1000.0, Cp_h=4184.0, rho_c=1000.0, Cp_c=4184.0,
                 T_h0=370.0, T_c0=300.0):

        # input validation
        if N_cells < 1:
            raise ValueError(f"'N_cells' must be >= 1 but is {N_cells}")
        if F_h <= 0:
            raise ValueError(f"'F_h' must be positive but is {F_h}")
        if F_c <= 0:
            raise ValueError(f"'F_c' must be positive but is {F_c}")
        if V_h <= 0:
            raise ValueError(f"'V_h' must be positive but is {V_h}")
        if V_c <= 0:
            raise ValueError(f"'V_c' must be positive but is {V_c}")

        # store parameters
        self.N_cells = int(N_cells)
        self.F_h = F_h
        self.F_c = F_c
        self.V_h = V_h
        self.V_c = V_c
        self.UA = UA
        self.rho_h = rho_h
        self.Cp_h = Cp_h
        self.rho_c = rho_c
        self.Cp_c = Cp_c

        # per-cell quantities
        N = self.N_cells
        V_cell_h = V_h / N
        V_cell_c = V_c / N
        UA_cell = UA / N

        # initial state: interleaved [T_h1, T_c1, T_h2, T_c2, ...]
        x0 = np.empty(2 * N)
        x0[0::2] = T_h0  # hot side
        x0[1::2] = T_c0  # cold side

        # rhs of heat exchanger ode
        def _fn_d(x, u, t):
            T_h_in, T_c_in = u
            dx = np.zeros(2 * N)

            _V_cell_h = self.V_h / self.N_cells
            _V_cell_c = self.V_c / self.N_cells
            _UA_cell = self.UA / self.N_cells

            alpha_h = _UA_cell / (self.rho_h * self.Cp_h * _V_cell_h)
            alpha_c = _UA_cell / (self.rho_c * self.Cp_c * _V_cell_c)
            flow_h = self.F_h / _V_cell_h
            flow_c = self.F_c / _V_cell_c

            for i in range(self.N_cells):
                T_h_i = x[2*i]
                T_c_i = x[2*i + 1]

                # hot side: upstream is i-1, or inlet
                T_h_prev = x[2*(i-1)] if i > 0 else T_h_in
                # cold side: upstream (counter-current) is i+1, or inlet
                T_c_next = x[2*(i+1) + 1] if i < self.N_cells - 1 else T_c_in

                dx[2*i]     = flow_h * (T_h_prev - T_h_i) - alpha_h * (T_h_i - T_c_i)
                dx[2*i + 1] = flow_c * (T_c_next - T_c_i) + alpha_c * (T_h_i - T_c_i)

            return dx

        # output: hot outlet = last cell hot, cold outlet = first cell cold
        def _fn_a(x, u, t):
            T_h_out = x[2*(self.N_cells - 1)]    # last hot cell
            T_c_out = x[1]                         # first cold cell
            return np.array([T_h_out, T_c_out])

        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=x0,
        )
