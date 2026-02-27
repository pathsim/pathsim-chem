########################################################################################
##
##                                  TESTS FOR
##                            'process.cstr.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.process import CSTR

from pathsim.solvers import EUF


# TESTS ================================================================================

class TestCSTR(unittest.TestCase):
    """Test the CSTR (Continuous Stirred-Tank Reactor) block."""

    def test_init_default(self):
        """Test default initialization."""
        C = CSTR()
        self.assertEqual(C.V, 1.0)
        self.assertEqual(C.F, 0.1)
        self.assertEqual(C.k0, 1e6)
        self.assertEqual(C.Ea, 50000.0)
        self.assertEqual(C.n, 1.0)
        self.assertEqual(C.dH_rxn, -50000.0)
        self.assertEqual(C.rho, 1000.0)
        self.assertEqual(C.Cp, 4184.0)
        self.assertEqual(C.UA, 500.0)

    def test_init_custom(self):
        """Test custom initialization."""
        C = CSTR(V=2.0, F=0.5, k0=1e4, Ea=40000.0, n=2.0,
                 dH_rxn=-30000.0, rho=800.0, Cp=3000.0, UA=200.0,
                 C_A0=1.5, T0=350.0)
        self.assertEqual(C.V, 2.0)
        self.assertEqual(C.F, 0.5)

        C.set_solver(EUF, parent=None)
        self.assertTrue(np.allclose(C.engine.initial_value, [1.5, 350.0]))

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            CSTR(V=0)
        with self.assertRaises(ValueError):
            CSTR(V=-1)
        with self.assertRaises(ValueError):
            CSTR(F=0)
        with self.assertRaises(ValueError):
            CSTR(rho=-1)
        with self.assertRaises(ValueError):
            CSTR(Cp=0)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(CSTR.input_port_labels["C_in"], 0)
        self.assertEqual(CSTR.input_port_labels["T_in"], 1)
        self.assertEqual(CSTR.input_port_labels["T_c"], 2)
        self.assertEqual(CSTR.output_port_labels["C_out"], 0)
        self.assertEqual(CSTR.output_port_labels["T_out"], 1)

    def test_output_equals_state(self):
        """Test that outputs equal state (well-mixed assumption)."""
        C = CSTR(C_A0=2.0, T0=350.0)
        C.set_solver(EUF, parent=None)
        C.update(None)

        self.assertAlmostEqual(C.outputs[0], 2.0)
        self.assertAlmostEqual(C.outputs[1], 350.0)

    def test_steady_state_no_reaction(self):
        """At very low k0, reactor is essentially non-reactive.
        Steady state should be C_out ≈ C_in, T_out ≈ T_in (when UA=0)."""
        C = CSTR(V=1.0, F=1.0, k0=0.0, Ea=0.0, n=1.0,
                 dH_rxn=0.0, UA=0.0, C_A0=1.0, T0=350.0)
        C.set_solver(EUF, parent=None)

        # Set feed conditions
        C.inputs[0] = 1.0    # C_A_in
        C.inputs[1] = 350.0  # T_in
        C.inputs[2] = 300.0  # T_c (irrelevant when UA=0)

        # At steady state with k0=0: dC/dt = (C_in - C)/tau = 0 => C = C_in
        # With initial C_A = C_in, the derivative should be zero
        state = C.engine.get()
        self.assertTrue(np.allclose(state, [1.0, 350.0]))

    def test_arrhenius_rate(self):
        """Verify Arrhenius rate constant computation indirectly through dynamics."""
        R_GAS = 8.314
        k0, Ea, T = 1e6, 50000.0, 350.0
        k_expected = k0 * np.exp(-Ea / (R_GAS * T))

        # At C_A=1, n=1, the reaction rate should be k
        C = CSTR(V=1.0, F=1.0, k0=k0, Ea=Ea, n=1.0,
                 dH_rxn=0.0, UA=0.0, C_A0=1.0, T0=T)
        C.set_solver(EUF, parent=None)

        C.inputs[0] = 1.0  # C_A_in = C_A => flow terms cancel
        C.inputs[1] = T    # T_in = T => flow terms cancel
        C.inputs[2] = T    # T_c = T

        # dC_A/dt = 0 (flow) - k * C_A = -k
        # dT/dt = 0 since dH_rxn=0 and UA=0
        # Evaluate the dynamics function directly
        x = np.array([1.0, T])
        u = np.array([1.0, T, T])
        dx = C.op_dyn(x, u, 0)

        self.assertAlmostEqual(dx[0], -k_expected, places=5)
        self.assertAlmostEqual(dx[1], 0.0, places=5)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
