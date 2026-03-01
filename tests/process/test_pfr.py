########################################################################################
##
##                                  TESTS FOR
##                              'process.pfr.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.process import PFR

from pathsim.solvers import EUF


# TESTS ================================================================================

class TestPFR(unittest.TestCase):
    """Test the plug flow reactor block."""

    def test_init_default(self):
        """Test default initialization."""
        P = PFR()
        self.assertEqual(P.N_cells, 5)
        self.assertEqual(P.V, 1.0)
        self.assertEqual(P.F, 0.1)
        self.assertEqual(P.k0, 1e6)
        self.assertEqual(P.Ea, 50000.0)
        self.assertEqual(P.n, 1.0)

    def test_init_custom(self):
        """Test custom initialization."""
        P = PFR(N_cells=10, V=2.0, F=0.5, k0=1e4, Ea=40000.0, n=2.0,
                dH_rxn=-30000.0, rho=800.0, Cp=3000.0, C0=1.5, T0=350.0)
        self.assertEqual(P.N_cells, 10)
        self.assertEqual(P.V, 2.0)

        P.set_solver(EUF, parent=None)
        state = P.engine.initial_value
        self.assertEqual(len(state), 20)  # 2 * N_cells
        self.assertTrue(np.all(state[0::2] == 1.5))   # concentrations
        self.assertTrue(np.all(state[1::2] == 350.0))  # temperatures

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            PFR(N_cells=0)
        with self.assertRaises(ValueError):
            PFR(V=-1)
        with self.assertRaises(ValueError):
            PFR(F=0)
        with self.assertRaises(ValueError):
            PFR(rho=-1)
        with self.assertRaises(ValueError):
            PFR(Cp=0)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(PFR.input_port_labels["C_in"], 0)
        self.assertEqual(PFR.input_port_labels["T_in"], 1)
        self.assertEqual(PFR.output_port_labels["C_out"], 0)
        self.assertEqual(PFR.output_port_labels["T_out"], 1)

    def test_state_size(self):
        """Test that state vector has correct size."""
        for N in [1, 3, 10]:
            P = PFR(N_cells=N)
            P.set_solver(EUF, parent=None)
            self.assertEqual(len(P.engine.initial_value), 2 * N)

    def test_output_initial(self):
        """Test outputs at initial state."""
        P = PFR(N_cells=3, C0=2.0, T0=350.0)
        P.set_solver(EUF, parent=None)
        P.update(None)

        # Last cell values
        self.assertAlmostEqual(P.outputs[0], 2.0)   # C_out
        self.assertAlmostEqual(P.outputs[1], 350.0)  # T_out

    def test_no_reaction(self):
        """With k0=0, no reaction occurs. At steady state C_out = C_in."""
        P = PFR(N_cells=3, V=1.0, F=1.0, k0=0.0, Ea=0.0, n=1.0,
                dH_rxn=0.0, C0=1.0, T0=350.0)
        P.set_solver(EUF, parent=None)

        P.inputs[0] = 1.0    # C_in
        P.inputs[1] = 350.0  # T_in

        # When C=C_in and T=T_in everywhere, with no reaction, derivatives = 0
        x = P.engine.get()
        u = np.array([1.0, 350.0])
        dx = P.op_dyn(x, u, 0)

        self.assertTrue(np.allclose(dx, 0.0, atol=1e-10))

    def test_reaction_direction(self):
        """With reaction, concentration should decrease along the reactor."""
        R_GAS = 8.314
        T = 350.0
        k0, Ea = 1e6, 50000.0
        k = k0 * np.exp(-Ea / (R_GAS * T))

        P = PFR(N_cells=1, V=1.0, F=1.0, k0=k0, Ea=Ea, n=1.0,
                dH_rxn=0.0, C0=1.0, T0=T)
        P.set_solver(EUF, parent=None)

        # Set inlet = current state, so flow terms cancel
        P.inputs[0] = 1.0  # C_in
        P.inputs[1] = T    # T_in

        x = np.array([1.0, T])
        u = np.array([1.0, T])
        dx = P.op_dyn(x, u, 0)

        # dC/dt = 0 (flow) - k*C = -k
        self.assertAlmostEqual(dx[0], -k, places=5)
        # dT/dt = 0 since dH_rxn=0
        self.assertAlmostEqual(dx[1], 0.0, places=5)

    def test_energy_conservation_exothermic(self):
        """For exothermic reaction (dH_rxn < 0), temperature should increase."""
        P = PFR(N_cells=1, V=1.0, F=1.0, k0=1e3, Ea=20000.0, n=1.0,
                dH_rxn=-50000.0, rho=1000.0, Cp=4184.0,
                C0=2.0, T0=350.0)
        P.set_solver(EUF, parent=None)

        P.inputs[0] = 2.0
        P.inputs[1] = 350.0

        x = np.array([2.0, 350.0])
        u = np.array([2.0, 350.0])
        dx = P.op_dyn(x, u, 0)

        # Concentration should decrease (reaction consuming)
        self.assertLess(dx[0], 0)
        # Temperature should increase (exothermic)
        self.assertGreater(dx[1], 0)

    def test_jacobian_stability(self):
        """Diagonal of Jacobian should be negative (stable)."""
        P = PFR(N_cells=2, V=1.0, F=0.1, k0=1e3, Ea=30000.0, n=1.0,
                dH_rxn=0.0, C0=1.0, T0=350.0)
        P.set_solver(EUF, parent=None)

        x = P.engine.get()
        u = np.array([1.0, 350.0])
        J = P.op_dyn.jac_x(x, u, 0)

        # All diagonal elements should be negative
        for i in range(len(x)):
            self.assertLess(J[i, i], 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
