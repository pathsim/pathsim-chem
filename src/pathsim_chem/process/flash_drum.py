#########################################################################################
##
##                    Binary Isothermal Flash Drum Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

# BLOCKS ================================================================================

class FlashDrum(DynamicalSystem):
    """Binary isothermal flash drum with Raoult's law vapor-liquid equilibrium.

    Models a flash drum with liquid holdup for a binary mixture. Feed enters as
    liquid, vapor and liquid streams exit. Temperature and pressure are specified
    as inputs. VLE is computed algebraically using K-values from Raoult's law:

    .. math::

        K_i = \\frac{P_{sat,i}(T)}{P}

    where the Antoine equation gives the saturation pressure:

    .. math::

        \\ln P_{sat,i} = A_i - \\frac{B_i}{T + C_i}

    The Rachford-Rice equation determines the vapor fraction :math:`\\beta`:

    .. math::

        \\sum_i \\frac{z_i (K_i - 1)}{1 + \\beta (K_i - 1)} = 0

    For a binary system this has the analytical solution:

    .. math::

        \\beta = -\\frac{z_1(K_1 - 1) + z_2(K_2 - 1)}{(K_1 - 1)(K_2 - 1)}

    Dynamic States
    ---------------
    The holdup moles of each component in the liquid phase:

    .. math::

        \\frac{dN_i}{dt} = F z_i - V y_i - L x_i

    Parameters
    ----------
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

    input_port_labels = {
        "F":   0,
        "z_1": 1,
        "T":   2,
        "P":   3,
    }

    output_port_labels = {
        "V_rate": 0,
        "L_rate": 1,
        "y_1":    2,
        "x_1":    3,
    }

    def __init__(self, holdup=100.0,
                 antoine_A=None, antoine_B=None, antoine_C=None,
                 N0=None):

        # input validation
        if holdup <= 0:
            raise ValueError(f"'holdup' must be positive but is {holdup}")

        self.holdup = holdup

        # default Antoine parameters: benzene / toluene (ln(Pa), K)
        # Converted from log10/mmHg: A_ln = ln(10)*A_log10 + ln(133.322)
        self.antoine_A = np.array(antoine_A) if antoine_A is not None else np.array([20.7936, 20.9064])
        self.antoine_B = np.array(antoine_B) if antoine_B is not None else np.array([2788.51, 3096.52])
        self.antoine_C = np.array(antoine_C) if antoine_C is not None else np.array([-52.36, -53.67])

        if len(self.antoine_A) != 2:
            raise ValueError("Binary flash: Antoine parameters must have length 2")

        # initial holdup (equal moles by default)
        if N0 is not None:
            x0 = np.array(N0, dtype=float)
        else:
            x0 = np.array([holdup / 2.0, holdup / 2.0])

        # shared VLE computation
        def _solve_vle(z, T, P):
            """Solve binary Rachford-Rice analytically, return (beta, y, x)."""
            P_sat = np.exp(self.antoine_A - self.antoine_B / (T + self.antoine_C))
            K = P_sat / P

            # check bubble/dew point conditions
            bubble = np.sum(z * K)   # if < 1, subcooled (all liquid)
            dew = np.sum(z / K) if np.all(K > 1e-12) else np.inf  # if < 1, superheated (all vapor)

            if bubble <= 1.0:
                # subcooled: all liquid
                beta = 0.0
            elif dew <= 1.0:
                # superheated: all vapor
                beta = 1.0
            else:
                # two-phase: solve Rachford-Rice
                d1 = K[0] - 1.0
                d2 = K[1] - 1.0
                den = d1 * d2
                beta = -(z[0] * d1 + z[1] * d2) / den if abs(den) > 1e-12 else 0.0
                beta = np.clip(beta, 0.0, 1.0)

            y = z * K / (1.0 + beta * (K - 1.0))
            x_eq = z / (1.0 + beta * (K - 1.0))

            # normalize for numerical safety
            y_s, x_s = y.sum(), x_eq.sum()
            if y_s > 0:
                y = y / y_s
            if x_s > 0:
                x_eq = x_eq / x_s

            return beta, y, x_eq

        # ensure u has expected 4 elements (handles framework probing)
        def _pad_u(u):
            u = np.atleast_1d(u)
            if len(u) < 4:
                padded = np.zeros(4)
                padded[:len(u)] = u
                return padded
            return u

        # rhs of flash drum ode
        def _fn_d(x, u, t):
            u = _pad_u(u)
            F_in, z_1, T, P = u
            z = np.array([z_1, 1.0 - z_1])

            beta, y, x_eq = _solve_vle(z, T, P)

            V_rate = beta * F_in
            L_rate = (1.0 - beta) * F_in

            return F_in * z - V_rate * y - L_rate * x_eq

        # algebraic output
        def _fn_a(x, u, t):
            u = _pad_u(u)
            F_in, z_1, T, P = u
            z = np.array([z_1, 1.0 - z_1])

            beta, y, x_eq = _solve_vle(z, T, P)

            V_rate = beta * F_in
            L_rate = (1.0 - beta) * F_in

            return np.array([V_rate, L_rate, y[0], x_eq[0]])

        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=x0,
        )
