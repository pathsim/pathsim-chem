#########################################################################################
##
##                    Single Equilibrium Distillation Tray Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

# BLOCKS ================================================================================

class DistillationTray(DynamicalSystem):
    """Single equilibrium distillation tray with constant molar overflow.

    Models one tray of a distillation column under the constant molar
    overflow (CMO) assumption. Liquid flows down from the tray above,
    vapor rises from the tray below. VLE is computed using constant
    relative volatility.

    Multiple trays can be wired in series via ``Connection`` objects to
    build a full distillation column.

    Mathematical Formulation
    -------------------------
    For a binary system with mole fraction :math:`x` of the light component
    on the tray:

    .. math::

        M \\frac{dx}{dt} = L_{in} \\, x_{in} + V_{in} \\, y_{in}
                          - L_{out} \\, x - V_{out} \\, y

    where VLE with constant relative volatility :math:`\\alpha` gives:

    .. math::

        y = \\frac{\\alpha \\, x}{1 + (\\alpha - 1) \\, x}

    Under CMO: :math:`L_{out} = L_{in}` and :math:`V_{out} = V_{in}`.

    Parameters
    ----------
    M : float
        Liquid holdup on the tray [mol].
    alpha : float
        Relative volatility of light to heavy component [-].
    x0 : float
        Initial liquid mole fraction of light component [-].
    """

    input_port_labels = {
        "L_in": 0,
        "x_in": 1,
        "V_in": 2,
        "y_in": 3,
    }

    output_port_labels = {
        "L_out": 0,
        "x_out": 1,
        "V_out": 2,
        "y_out": 3,
    }

    def __init__(self, M=1.0, alpha=2.5, x0=0.5):

        # input validation
        if M <= 0:
            raise ValueError(f"'M' must be positive but is {M}")
        if alpha <= 0:
            raise ValueError(f"'alpha' must be positive but is {alpha}")
        if not 0.0 <= x0 <= 1.0:
            raise ValueError(f"'x0' must be in [0, 1] but is {x0}")

        self.M = M
        self.alpha = alpha

        # VLE: y = alpha*x / (1 + (alpha-1)*x)
        def _vle(x_val):
            return self.alpha * x_val / (1.0 + (self.alpha - 1.0) * x_val)

        # ensure u has expected 4 elements (handles framework probing)
        def _pad_u(u):
            u = np.atleast_1d(u)
            if len(u) < 4:
                padded = np.zeros(4)
                padded[:len(u)] = u
                return padded
            return u

        # rhs of tray ode
        def _fn_d(x, u, t):
            u = _pad_u(u)
            L_in, x_in, V_in, y_in = u

            x_tray = x[0]
            y_tray = _vle(x_tray)

            # CMO: L_out = L_in, V_out = V_in
            dx = (L_in * x_in + V_in * y_in - L_in * x_tray - V_in * y_tray) / self.M
            return np.array([dx])

        # jacobian wrt x
        def _jc_d(x, u, t):
            u = _pad_u(u)
            L_in, x_in, V_in, y_in = u
            x_tray = x[0]

            # dy/dx = alpha / (1 + (alpha-1)*x)^2
            denom = (1.0 + (self.alpha - 1.0) * x_tray)**2
            dy_dx = self.alpha / denom

            return np.array([[-L_in / self.M - V_in * dy_dx / self.M]])

        # output: L_out, x, V_out, y (CMO: pass-through flows)
        def _fn_a(x, u, t):
            u = _pad_u(u)
            L_in, x_in, V_in, y_in = u
            x_tray = x[0]
            y_tray = _vle(x_tray)
            return np.array([L_in, x_tray, V_in, y_tray])

        super().__init__(
            func_dyn=_fn_d,
            jac_dyn=_jc_d,
            func_alg=_fn_a,
            initial_value=np.array([x0]),
        )
