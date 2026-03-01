#########################################################################################
##
##                    Multi-Component Isothermal Flash Drum Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
from scipy.optimize import brentq

from pathsim.blocks.dynsys import DynamicalSystem

# BLOCKS ================================================================================

class MultiComponentFlash(DynamicalSystem):
    """Generalized isothermal flash drum for N components with Raoult's law VLE.

    Models a flash drum with liquid holdup for an N-component mixture. Feed enters
    as liquid, vapor and liquid streams exit. Temperature and pressure are specified
    as inputs. VLE is computed using K-values from Raoult's law with Antoine
    equation vapor pressures.

    The Rachford-Rice equation is solved iteratively using Brent's method:

    .. math::

        f(\\beta) = \\sum_i \\frac{z_i (K_i - 1)}{1 + \\beta (K_i - 1)} = 0

    K-values from Raoult's law:

    .. math::

        K_i = \\frac{\\exp(A_i - B_i / (T + C_i))}{P}

    Dynamic States
    ---------------
    The holdup moles of each component in the liquid phase:

    .. math::

        \\frac{dN_i}{dt} = F z_i - V y_i - L x_i

    Parameters
    ----------
    N_comp : int
        Number of components (must be >= 2).
    holdup : float
        Total liquid holdup [mol]. Assumed approximately constant.
    antoine_A : array_like
        Antoine A parameters for each component [ln(Pa)].
    antoine_B : array_like
        Antoine B parameters for each component [K].
    antoine_C : array_like
        Antoine C parameters for each component [K].
    N0 : array_like or None
        Initial component holdup moles [mol]. If None, equal split assumed.
    """

    def __init__(self, N_comp=3, holdup=100.0,
                 antoine_A=None, antoine_B=None, antoine_C=None,
                 N0=None):

        # input validation
        if N_comp < 2:
            raise ValueError(f"'N_comp' must be >= 2 but is {N_comp}")
        if holdup <= 0:
            raise ValueError(f"'holdup' must be positive but is {holdup}")

        self.N_comp = int(N_comp)
        self.holdup = holdup
        nc = self.N_comp

        # default Antoine parameters: benzene / toluene / p-xylene (ln(Pa), K)
        if antoine_A is not None:
            self.antoine_A = np.array(antoine_A, dtype=float)
        else:
            self.antoine_A = np.array([20.7936, 20.9064, 20.9891])[:nc]

        if antoine_B is not None:
            self.antoine_B = np.array(antoine_B, dtype=float)
        else:
            self.antoine_B = np.array([2788.51, 3096.52, 3346.65])[:nc]

        if antoine_C is not None:
            self.antoine_C = np.array(antoine_C, dtype=float)
        else:
            self.antoine_C = np.array([-52.36, -53.67, -57.84])[:nc]

        if len(self.antoine_A) != nc:
            raise ValueError(f"Antoine parameters must have length {nc}")
        if len(self.antoine_B) != nc:
            raise ValueError(f"Antoine parameters must have length {nc}")
        if len(self.antoine_C) != nc:
            raise ValueError(f"Antoine parameters must have length {nc}")

        # initial holdup (equal moles by default)
        if N0 is not None:
            x0 = np.array(N0, dtype=float)
        else:
            x0 = np.full(nc, holdup / nc)

        if len(x0) != nc:
            raise ValueError(f"'N0' must have length {nc}")

        # dynamic port labels: set before super().__init__()
        # inputs: F, z_1, ..., z_{nc-1}, T, P
        inp = {"F": 0}
        for i in range(1, nc):
            inp[f"z_{i}"] = i
        inp["T"] = nc
        inp["P"] = nc + 1
        self.input_port_labels = inp

        n_inputs = nc + 2

        # outputs: V_rate, L_rate, y_1, ..., y_{nc-1}, x_1, ..., x_{nc-1}
        out = {"V_rate": 0, "L_rate": 1}
        for i in range(1, nc):
            out[f"y_{i}"] = 1 + i
        for i in range(1, nc):
            out[f"x_{i}"] = nc + i
        self.output_port_labels = out

        # shared VLE computation
        def _solve_vle(z, T, P):
            """Solve N-component Rachford-Rice, return (beta, y, x)."""
            P_sat = np.exp(self.antoine_A - self.antoine_B / (T + self.antoine_C))
            K = P_sat / P

            # bubble/dew point checks
            bubble = np.sum(z * K)
            K_safe = np.where(K > 1e-12, K, 1e-12)
            dew = np.sum(z / K_safe)

            if bubble <= 1.0:
                # subcooled: all liquid
                beta = 0.0
                y = z * K
                y_s = y.sum()
                if y_s > 0:
                    y = y / y_s
                return beta, y, z.copy()

            if dew <= 1.0:
                # superheated: all vapor
                beta = 1.0
                x_eq = z / K_safe
                x_s = x_eq.sum()
                if x_s > 0:
                    x_eq = x_eq / x_s
                return beta, z.copy(), x_eq

            # two-phase: solve Rachford-Rice via Brent's method
            Km1 = K - 1.0

            def rr_func(beta):
                return np.sum(z * Km1 / (1.0 + beta * Km1))

            # Whitson & Michelsen bounds
            K_max = K.max()
            K_min = K.min()
            beta_min = max(0.0, 1.0 / (1.0 - K_max)) if K_max != 1.0 else 0.0
            beta_max = min(1.0, 1.0 / (1.0 - K_min)) if K_min != 1.0 else 1.0

            # ensure valid bracket
            beta_min = max(beta_min, 0.0)
            beta_max = min(beta_max, 1.0)
            if beta_min >= beta_max:
                beta_min, beta_max = 0.0, 1.0

            try:
                beta = brentq(rr_func, beta_min, beta_max, xtol=1e-12)
            except ValueError:
                # fallback: try full [0, 1] bracket
                try:
                    beta = brentq(rr_func, 0.0, 1.0, xtol=1e-12)
                except ValueError:
                    beta = 0.5

            beta = np.clip(beta, 0.0, 1.0)

            y = z * K / (1.0 + beta * Km1)
            x_eq = z / (1.0 + beta * Km1)

            # normalize for numerical safety
            y_s = y.sum()
            x_s = x_eq.sum()
            if y_s > 0:
                y = y / y_s
            if x_s > 0:
                x_eq = x_eq / x_s

            return beta, y, x_eq

        def _pad_u(u):
            u = np.atleast_1d(u)
            if len(u) < n_inputs:
                padded = np.zeros(n_inputs)
                padded[:len(u)] = u
                return padded
            return u

        def _extract_z(u):
            """Extract full composition vector from inputs (last component inferred)."""
            z_partial = u[1:nc]  # z_1 ... z_{nc-1}
            z_last = 1.0 - np.sum(z_partial)
            return np.append(z_partial, z_last)

        # rhs of flash drum ode
        def _fn_d(x, u, t):
            u = _pad_u(u)
            F_in = u[0]
            z = _extract_z(u)
            T = u[nc]
            P = u[nc + 1]

            beta, y, x_eq = _solve_vle(z, T, P)

            V_rate = beta * F_in
            L_rate = (1.0 - beta) * F_in

            return F_in * z - V_rate * y - L_rate * x_eq

        # algebraic output
        def _fn_a(x, u, t):
            u = _pad_u(u)
            F_in = u[0]
            z = _extract_z(u)
            T = u[nc]
            P = u[nc + 1]

            beta, y, x_eq = _solve_vle(z, T, P)

            V_rate = beta * F_in
            L_rate = (1.0 - beta) * F_in

            # output: V_rate, L_rate, y_1..y_{nc-1}, x_1..x_{nc-1}
            result = np.empty(2 + 2 * (nc - 1))
            result[0] = V_rate
            result[1] = L_rate
            result[2:2 + nc - 1] = y[:nc - 1]
            result[2 + nc - 1:] = x_eq[:nc - 1]

            return result

        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=x0,
        )
