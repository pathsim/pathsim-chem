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

    For a binary system this has an analytical solution.

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

        # default Antoine parameters: benzene / toluene (Pa, K)
        self.antoine_A = np.array(antoine_A) if antoine_A is not None else np.array([15.9008, 16.0137])
        self.antoine_B = np.array(antoine_B) if antoine_B is not None else np.array([2788.51, 3096.52])
        self.antoine_C = np.array(antoine_C) if antoine_C is not None else np.array([-52.36, -53.67])

        if len(self.antoine_A) != 2:
            raise ValueError("Binary flash: Antoine parameters must have length 2")

        # initial holdup (equal moles by default)
        if N0 is not None:
            x0 = np.array(N0, dtype=float)
        else:
            x0 = np.array([holdup / 2.0, holdup / 2.0])

        # rhs of flash drum ode
        def _fn_d(x, u, t):
            F_in, z_1, T, P = u

            z = np.array([z_1, 1.0 - z_1])
            N_total = x[0] + x[1]

            # liquid composition from holdup
            if N_total > 0:
                x_liq = x / N_total
            else:
                x_liq = np.array([0.5, 0.5])

            # K-values from Antoine + Raoult
            P_sat = np.exp(self.antoine_A - self.antoine_B / (T + self.antoine_C))
            K = P_sat / P

            # Rachford-Rice for binary: analytical solution
            # beta = (z1*(K1-1) + z2*(K2-1)) ... but simpler:
            # beta = (1 - sum(z/K)) / (sum(z*K) - 1) ... use standard form
            # For binary: beta = (z1 - 1/K1_eff) / (1 - 1/K1_eff) ...
            # Actually use: beta from f(beta) = sum zi(Ki-1)/(1+beta(Ki-1)) = 0
            # For binary with z = x_liq (liquid feed to flash):
            denom1 = K[0] - 1.0
            denom2 = K[1] - 1.0

            if abs(denom1) < 1e-12 and abs(denom2) < 1e-12:
                # no separation
                beta = 0.0
            else:
                # Solve Rachford-Rice for feed composition z
                # f(beta) = z[0]*(K[0]-1)/(1+beta*(K[0]-1)) + z[1]*(K[1]-1)/(1+beta*(K[1]-1)) = 0
                # For binary: beta = -z[0]*(K[0]-1)/((K[0]-1)*(K[1]-1)) ...
                # Direct: beta = (z[0]*(K[0]-1) + z[1]*(K[1]-1)) isn't right
                # Newton or analytical: for 2 components
                # z1(K1-1)/(1+b(K1-1)) + z2(K2-1)/(1+b(K2-1)) = 0
                # z1(K1-1)(1+b(K2-1)) + z2(K2-1)(1+b(K1-1)) = 0
                # z1(K1-1) + z1(K1-1)*b*(K2-1) + z2(K2-1) + z2(K2-1)*b*(K1-1) = 0
                # [z1(K1-1) + z2(K2-1)] + b*(K1-1)*(K2-1)*[z1+z2] = 0
                # b = -[z1(K1-1) + z2(K2-1)] / [(K1-1)*(K2-1)]
                num = z[0] * denom1 + z[1] * denom2
                den = denom1 * denom2
                if abs(den) < 1e-12:
                    beta = 0.0
                else:
                    beta = -num / den

            # clamp beta to [0, 1]
            beta = np.clip(beta, 0.0, 1.0)

            # vapor and liquid compositions
            y = z * K / (1.0 + beta * (K - 1.0))
            x_eq = z / (1.0 + beta * (K - 1.0))

            # normalize for safety
            y_sum = y.sum()
            x_sum = x_eq.sum()
            if y_sum > 0:
                y = y / y_sum
            if x_sum > 0:
                x_eq = x_eq / x_sum

            # flow rates
            V_rate = beta * F_in
            L_rate = (1.0 - beta) * F_in

            # holdup dynamics
            dN = F_in * z - V_rate * y - L_rate * x_eq

            return dN

        # algebraic output: compute VLE and return rates + compositions
        def _fn_a(x, u, t):
            F_in, z_1, T, P = u

            z = np.array([z_1, 1.0 - z_1])

            # K-values
            P_sat = np.exp(self.antoine_A - self.antoine_B / (T + self.antoine_C))
            K = P_sat / P

            denom1 = K[0] - 1.0
            denom2 = K[1] - 1.0

            if abs(denom1) < 1e-12 and abs(denom2) < 1e-12:
                beta = 0.0
            else:
                num = z[0] * denom1 + z[1] * denom2
                den = denom1 * denom2
                beta = -num / den if abs(den) > 1e-12 else 0.0

            beta = np.clip(beta, 0.0, 1.0)

            y = z * K / (1.0 + beta * (K - 1.0))
            x_eq = z / (1.0 + beta * (K - 1.0))

            y_sum = y.sum()
            x_sum = x_eq.sum()
            if y_sum > 0:
                y = y / y_sum
            if x_sum > 0:
                x_eq = x_eq / x_sum

            V_rate = beta * F_in
            L_rate = (1.0 - beta) * F_in

            return np.array([V_rate, L_rate, y[0], x_eq[0]])

        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=x0,
        )
