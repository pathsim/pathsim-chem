#########################################################################################
##
##                         Reactor Point Kinetics Block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

# CONSTANTS =============================================================================

# Keepin U-235 thermal 6-group delayed neutron data
BETA_U235 = [0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273]
LAMBDA_U235 = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]

# BLOCKS ================================================================================

class PointKinetics(DynamicalSystem):
    """Reactor point kinetics equations with delayed neutron precursors.

    Models the time-dependent neutron population in a nuclear reactor using
    the point kinetics equations (PKE). The neutron density responds to
    reactivity changes through prompt fission and delayed neutron emission
    from G precursor groups.

    Mathematical Formulation
    -------------------------
    The state vector is :math:`[n, C_1, C_2, \\ldots, C_G]` where :math:`n`
    is the neutron density (or power) and :math:`C_i` are the delayed neutron
    precursor concentrations.

    .. math::

        \\frac{dn}{dt} = \\frac{\\rho - \\beta}{\\Lambda} \\, n
            + \\sum_{i=1}^{G} \\lambda_i \\, C_i + S

    .. math::

        \\frac{dC_i}{dt} = \\frac{\\beta_i}{\\Lambda} \\, n
            - \\lambda_i \\, C_i \\qquad i = 1, \\ldots, G

    where :math:`\\beta = \\sum_i \\beta_i` is the total delayed neutron
    fraction and :math:`\\Lambda` is the prompt neutron generation time.

    Parameters
    ----------
    n0 : float
        Initial neutron density [-]. Default 1.0 (normalized).
    Lambda : float
        Prompt neutron generation time [s].
    beta : array_like
        Delayed neutron fractions per group [-].
    lam : array_like
        Precursor decay constants per group [1/s].
    """

    input_port_labels = {
        "rho": 0,
        "S":   1,
    }

    output_port_labels = {
        "n": 0,
    }

    def __init__(self, n0=1.0, Lambda=1e-5,
                 beta=None, lam=None):

        # defaults
        if beta is None:
            beta = BETA_U235
        if lam is None:
            lam = LAMBDA_U235

        beta = np.asarray(beta, dtype=float)
        lam = np.asarray(lam, dtype=float)

        # input validation
        if Lambda <= 0:
            raise ValueError(f"'Lambda' must be positive but is {Lambda}")
        if len(beta) != len(lam):
            raise ValueError(
                f"'beta' and 'lam' must have the same length "
                f"but got {len(beta)} and {len(lam)}"
            )
        if len(beta) == 0:
            raise ValueError("'beta' and 'lam' must have at least one group")

        # store parameters
        G = len(beta)
        beta_total = float(np.sum(beta))

        self.n0 = n0
        self.Lambda = Lambda
        self.beta = beta
        self.lam = lam
        self.G = G
        self.beta_total = beta_total

        # initial conditions: steady state at rho = 0
        x0 = np.empty(G + 1)
        x0[0] = n0
        for i in range(G):
            x0[1 + i] = beta[i] / (Lambda * lam[i]) * n0

        # rhs of point kinetics ode system
        def _fn_d(x, u, t):
            n = x[0]
            C = x[1:]

            rho = u[0]
            S = u[1] if len(u) > 1 else 0.0

            dn = (rho - beta_total) / Lambda * n + np.dot(lam, C) + S

            dC = beta / Lambda * n - lam * C

            return np.concatenate(([dn], dC))

        # jacobian of rhs wrt state [n, C_1, ..., C_G]
        def _jc_d(x, u, t):
            rho = u[0]

            J = np.zeros((G + 1, G + 1))

            J[0, 0] = (rho - beta_total) / Lambda
            for i in range(G):
                J[0, 1 + i] = lam[i]
                J[1 + i, 0] = beta[i] / Lambda
                J[1 + i, 1 + i] = -lam[i]

            return J

        # output function: neutron density only
        def _fn_a(x, u, t):
            return x[:1].copy()

        super().__init__(
            func_dyn=_fn_d,
            jac_dyn=_jc_d,
            func_alg=_fn_a,
            initial_value=x0,
        )
