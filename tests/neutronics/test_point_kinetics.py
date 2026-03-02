########################################################################################
##
##                                  TESTS FOR
##                       'neutronics.point_kinetics.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.neutronics import PointKinetics
from pathsim_chem.neutronics.point_kinetics import BETA_U235, LAMBDA_U235

from pathsim.solvers import EUF


# TESTS ================================================================================

class TestPointKinetics(unittest.TestCase):
    """Test the PointKinetics (reactor point kinetics equations) block."""

    def test_init_default(self):
        """Test default initialization with Keepin U-235 data."""
        pk = PointKinetics()
        self.assertEqual(pk.n0, 1.0)
        self.assertEqual(pk.Lambda, 1e-5)
        self.assertEqual(pk.G, 6)
        np.testing.assert_array_equal(pk.beta, BETA_U235)
        np.testing.assert_array_equal(pk.lam, LAMBDA_U235)
        self.assertAlmostEqual(pk.beta_total, sum(BETA_U235))

    def test_init_custom(self):
        """Test custom initialization with user-specified groups."""
        beta = [0.003, 0.004]
        lam = [0.1, 1.0]
        pk = PointKinetics(n0=2.0, Lambda=1e-4, beta=beta, lam=lam)

        self.assertEqual(pk.n0, 2.0)
        self.assertEqual(pk.Lambda, 1e-4)
        self.assertEqual(pk.G, 2)
        np.testing.assert_array_almost_equal(pk.beta, beta)
        np.testing.assert_array_almost_equal(pk.lam, lam)

        pk.set_solver(EUF, parent=None)
        x0 = pk.engine.initial_value
        self.assertEqual(len(x0), 3)
        self.assertAlmostEqual(x0[0], 2.0)
        # C_i(0) = beta_i / (Lambda * lam_i) * n0
        self.assertAlmostEqual(x0[1], 0.003 / (1e-4 * 0.1) * 2.0)
        self.assertAlmostEqual(x0[2], 0.004 / (1e-4 * 1.0) * 2.0)

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            PointKinetics(Lambda=0)
        with self.assertRaises(ValueError):
            PointKinetics(Lambda=-1e-5)
        with self.assertRaises(ValueError):
            PointKinetics(beta=[0.001, 0.002], lam=[0.1])
        with self.assertRaises(ValueError):
            PointKinetics(beta=[], lam=[])

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(PointKinetics.input_port_labels["rho"], 0)
        self.assertEqual(PointKinetics.input_port_labels["S"], 1)
        self.assertEqual(PointKinetics.output_port_labels["n"], 0)

    def test_state_size(self):
        """State vector has G+1 entries (neutron density + G precursor groups)."""
        for G in [1, 2, 6]:
            beta = [0.001] * G
            lam = [0.1] * G
            pk = PointKinetics(beta=beta, lam=lam)
            pk.set_solver(EUF, parent=None)
            self.assertEqual(len(pk.engine.initial_value), G + 1)

    def test_steady_state(self):
        """At rho=0 and S=0, the initial conditions are steady state (dn/dt ~ 0)."""
        pk = PointKinetics()
        pk.set_solver(EUF, parent=None)

        x0 = pk.engine.initial_value.copy()
        u = np.array([0.0, 0.0])  # rho=0, S=0
        dx = pk.op_dyn(x0, u, 0)

        np.testing.assert_allclose(dx, 0.0, atol=1e-10)

    def test_precursor_equilibrium(self):
        """Verify initial precursor concentrations satisfy equilibrium."""
        n0 = 5.0
        Lambda = 2e-5
        beta = [0.0003, 0.001, 0.002]
        lam = [0.05, 0.5, 2.0]

        pk = PointKinetics(n0=n0, Lambda=Lambda, beta=beta, lam=lam)
        pk.set_solver(EUF, parent=None)

        x0 = pk.engine.initial_value
        for i in range(3):
            expected = beta[i] / (Lambda * lam[i]) * n0
            self.assertAlmostEqual(x0[1 + i], expected, places=8)

    def test_subcritical_source(self):
        """With rho<0 and external source, system converges to steady state.

        At equilibrium: n_ss ≈ S * Lambda / (beta - rho)  (for |rho| >> 0)
        """
        beta = [0.003, 0.004]
        lam = [0.1, 1.0]
        Lambda = 1e-4
        rho = -0.05
        S = 1e6
        beta_total = sum(beta)

        # Start from source-free equilibrium, then apply source + negative rho
        pk = PointKinetics(n0=0.001, Lambda=Lambda, beta=beta, lam=lam)
        pk.set_solver(EUF, parent=None)

        # Expected steady state: dn/dt=0, dC_i/dt=0
        # From dC_i/dt=0: C_i = beta_i/(Lambda*lam_i) * n_ss
        # Substituting into dn/dt=0:
        #   0 = (rho-beta)/Lambda * n_ss + sum(lam_i * beta_i/(Lambda*lam_i) * n_ss) + S
        #   0 = (rho-beta)/Lambda * n_ss + beta/Lambda * n_ss + S
        #   0 = rho/Lambda * n_ss + S
        #   n_ss = -S * Lambda / rho
        n_ss = -S * Lambda / rho
        self.assertGreater(n_ss, 0)  # sanity check: should be positive for rho < 0

        # Verify by plugging n_ss into the equations directly
        x_ss = np.empty(3)
        x_ss[0] = n_ss
        for i in range(2):
            x_ss[1 + i] = beta[i] / (Lambda * lam[i]) * n_ss

        u = np.array([rho, S])
        dx = pk.op_dyn(x_ss, u, 0)
        np.testing.assert_allclose(dx, 0.0, atol=1e-6)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
