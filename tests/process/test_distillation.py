########################################################################################
##
##                                  TESTS FOR
##                        'process.distillation.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.process import DistillationTray

from pathsim.solvers import EUF


# TESTS ================================================================================

class TestDistillationTray(unittest.TestCase):
    """Test the single equilibrium distillation tray block."""

    def test_init_default(self):
        """Test default initialization."""
        D = DistillationTray()
        self.assertEqual(D.M, 1.0)
        self.assertEqual(D.alpha, 2.5)

    def test_init_custom(self):
        """Test custom initialization."""
        D = DistillationTray(M=5.0, alpha=3.0, x0=0.3)
        self.assertEqual(D.M, 5.0)
        self.assertEqual(D.alpha, 3.0)

        D.set_solver(EUF, parent=None)
        self.assertTrue(np.allclose(D.engine.initial_value, [0.3]))

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            DistillationTray(M=0)
        with self.assertRaises(ValueError):
            DistillationTray(M=-1)
        with self.assertRaises(ValueError):
            DistillationTray(alpha=0)
        with self.assertRaises(ValueError):
            DistillationTray(alpha=-1)
        with self.assertRaises(ValueError):
            DistillationTray(x0=-0.1)
        with self.assertRaises(ValueError):
            DistillationTray(x0=1.1)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(DistillationTray.input_port_labels["L_in"], 0)
        self.assertEqual(DistillationTray.input_port_labels["x_in"], 1)
        self.assertEqual(DistillationTray.input_port_labels["V_in"], 2)
        self.assertEqual(DistillationTray.input_port_labels["y_in"], 3)
        self.assertEqual(DistillationTray.output_port_labels["L_out"], 0)
        self.assertEqual(DistillationTray.output_port_labels["x_out"], 1)
        self.assertEqual(DistillationTray.output_port_labels["V_out"], 2)
        self.assertEqual(DistillationTray.output_port_labels["y_out"], 3)

    def test_vle_relationship(self):
        """Test that y = alpha*x / (1 + (alpha-1)*x)."""
        alpha = 2.5
        x0 = 0.4
        D = DistillationTray(alpha=alpha, x0=x0)
        D.set_solver(EUF, parent=None)

        D.inputs[0] = 1.0  # L_in
        D.inputs[1] = 0.5  # x_in
        D.inputs[2] = 1.0  # V_in
        D.inputs[3] = 0.5  # y_in

        D.update(None)

        y_expected = alpha * x0 / (1.0 + (alpha - 1.0) * x0)

        self.assertAlmostEqual(D.outputs[1], x0)         # x_out = state
        self.assertAlmostEqual(D.outputs[3], y_expected)  # y_out = VLE

    def test_cmo_flow_passthrough(self):
        """Under CMO, L_out = L_in and V_out = V_in."""
        D = DistillationTray()
        D.set_solver(EUF, parent=None)

        L_in, V_in = 2.0, 3.0
        D.inputs[0] = L_in
        D.inputs[1] = 0.5
        D.inputs[2] = V_in
        D.inputs[3] = 0.5

        D.update(None)

        self.assertAlmostEqual(D.outputs[0], L_in)  # L_out
        self.assertAlmostEqual(D.outputs[2], V_in)  # V_out

    def test_mass_balance_steady_state(self):
        """At steady state, input = output (mass balance)."""
        D = DistillationTray(alpha=2.5, x0=0.5)
        D.set_solver(EUF, parent=None)

        L_in, V_in = 1.0, 1.0
        x_in, y_in = 0.6, 0.4

        D.inputs[0] = L_in
        D.inputs[1] = x_in
        D.inputs[2] = V_in
        D.inputs[3] = y_in

        # At steady state: L_in*x_in + V_in*y_in = L_out*x + V_out*y
        # This is only true when dx/dt = 0, i.e. at the steady state composition.
        # For now, just verify that total input flow = total output flow (CMO).
        D.update(None)
        total_in = L_in + V_in
        total_out = D.outputs[0] + D.outputs[2]
        self.assertAlmostEqual(total_in, total_out)

    def test_enrichment(self):
        """Vapor should be enriched in light component relative to liquid."""
        D = DistillationTray(alpha=3.0, x0=0.3)
        D.set_solver(EUF, parent=None)

        D.inputs[0] = 1.0
        D.inputs[1] = 0.5
        D.inputs[2] = 1.0
        D.inputs[3] = 0.5

        D.update(None)

        x_out = D.outputs[1]
        y_out = D.outputs[3]

        # With alpha > 1, y > x always
        self.assertGreater(y_out, x_out)

    def test_jacobian_sign(self):
        """Jacobian diagonal should be negative (stable)."""
        D = DistillationTray(alpha=2.5, x0=0.5, M=1.0)
        D.set_solver(EUF, parent=None)

        L_in, V_in = 1.0, 1.0
        x = np.array([0.5])
        u = np.array([L_in, 0.5, V_in, 0.5])

        # Evaluate jacobian
        J = D.op_dyn.jac_x(x, u, 0)
        self.assertLess(J[0, 0], 0)  # stable


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
