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

        N = self.N_cells

        # dynamic output port labels: outlets + per-cell profiles
        self.output_port_labels = {"T_h_out": 0, "T_c_out": 1}
        for i in range(N):
            self.output_port_labels[f"T_h_{i+1}"] = 2 + i
            self.output_port_labels[f"T_c_{i+1}"] = 2 + N + i

        # initial state: interleaved [T_h1, T_c1, T_h2, T_c2, ...]
        x0 = np.empty(2 * N)
        x0[0::2] = T_h0
        x0[1::2] = T_c0

        # ensure u has expected 2 elements (handles framework probing)
        def _pad_u(u):
            u = np.atleast_1d(u)
            if len(u) < 2:
                padded = np.zeros(2)
                padded[:len(u)] = u
                return padded
            return u

        # rhs of heat exchanger ode (vectorized)
        def _fn_d(x, u, t):
            u = _pad_u(u)
            T_h_in, T_c_in = u
            N = self.N_cells

            V_cell_h = self.V_h / N
            V_cell_c = self.V_c / N
            UA_cell = self.UA / N

            fh = self.F_h / V_cell_h
            fc = self.F_c / V_cell_c
            ah = UA_cell / (self.rho_h * self.Cp_h * V_cell_h)
            ac = UA_cell / (self.rho_c * self.Cp_c * V_cell_c)

            T_h = x[0::2]
            T_c = x[1::2]

            # upstream temperatures with boundary conditions
            T_h_prev = np.empty(N)
            T_h_prev[0] = T_h_in
            T_h_prev[1:] = T_h[:-1]

            T_c_next = np.empty(N)
            T_c_next[-1] = T_c_in
            T_c_next[:-1] = T_c[1:]

            dx = np.empty(2 * N)
            dx[0::2] = fh * (T_h_prev - T_h) - ah * (T_h - T_c)
            dx[1::2] = fc * (T_c_next - T_c) + ac * (T_h - T_c)

            return dx

        # analytical jacobian (block-tridiagonal structure)
        def _jc_d(x, u, t):
            N = self.N_cells

            V_cell_h = self.V_h / N
            V_cell_c = self.V_c / N
            UA_cell = self.UA / N

            fh = self.F_h / V_cell_h
            fc = self.F_c / V_cell_c
            ah = UA_cell / (self.rho_h * self.Cp_h * V_cell_h)
            ac = UA_cell / (self.rho_c * self.Cp_c * V_cell_c)

            dim = 2 * N
            J = np.zeros((dim, dim))

            for i in range(N):
                hi = 2 * i      # hot index
                ci = 2 * i + 1  # cold index

                # dT_h_i/dT_h_i, dT_h_i/dT_c_i
                J[hi, hi] = -fh - ah
                J[hi, ci] = ah

                # dT_c_i/dT_h_i, dT_c_i/dT_c_i
                J[ci, hi] = ac
                J[ci, ci] = -fc - ac

                # dT_h_i/dT_h_{i-1} (hot upstream coupling)
                if i > 0:
                    J[hi, 2*(i-1)] = fh

                # dT_c_i/dT_c_{i+1} (cold upstream coupling, counter-current)
                if i < N - 1:
                    J[ci, 2*(i+1) + 1] = fc

            return J

        # output: outlets + per-cell profiles
        # [T_h_out, T_c_out, T_h_1..T_h_N, T_c_1..T_c_N]
        def _fn_a(x, u, t):
            N = self.N_cells
            out = np.empty(2 + 2 * N)
            out[0] = x[2 * (N - 1)]   # T_h_out
            out[1] = x[1]              # T_c_out
            out[2:N + 2] = x[0::2]    # hot profile
            out[N + 2:] = x[1::2]     # cold profile
            return out

        super().__init__(
            func_dyn=_fn_d,
            jac_dyn=_jc_d,
            func_alg=_fn_a,
            initial_value=x0,
        )
