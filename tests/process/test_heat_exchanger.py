########################################################################################
##
##                                  TESTS FOR
##                       'process.heat_exchanger.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.process import HeatExchanger

from pathsim.solvers import EUF


# TESTS ================================================================================

class TestHeatExchanger(unittest.TestCase):
    """Test the counter-current heat exchanger block."""

    def test_init_default(self):
        """Test default initialization."""
        H = HeatExchanger()
        self.assertEqual(H.N_cells, 5)
        self.assertEqual(H.F_h, 0.1)
        self.assertEqual(H.F_c, 0.1)
        self.assertEqual(H.UA, 500.0)

    def test_init_custom(self):
        """Test custom initialization."""
        H = HeatExchanger(N_cells=10, F_h=0.5, F_c=0.3, V_h=1.0, V_c=0.8,
                          UA=1000.0, rho_h=900.0, Cp_h=3500.0,
                          rho_c=1000.0, Cp_c=4184.0,
                          T_h0=400.0, T_c0=290.0)
        self.assertEqual(H.N_cells, 10)
        self.assertEqual(H.F_h, 0.5)

        H.set_solver(EUF, parent=None)
        state = H.engine.initial_value
        self.assertEqual(len(state), 20)  # 2 * N_cells
        self.assertTrue(np.all(state[0::2] == 400.0))  # hot cells
        self.assertTrue(np.all(state[1::2] == 290.0))  # cold cells

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            HeatExchanger(N_cells=0)
        with self.assertRaises(ValueError):
            HeatExchanger(F_h=-1)
        with self.assertRaises(ValueError):
            HeatExchanger(F_c=0)
        with self.assertRaises(ValueError):
            HeatExchanger(V_h=-1)
        with self.assertRaises(ValueError):
            HeatExchanger(V_c=0)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(HeatExchanger.input_port_labels["T_h_in"], 0)
        self.assertEqual(HeatExchanger.input_port_labels["T_c_in"], 1)
        hx = HeatExchanger(N_cells=5)
        self.assertEqual(hx.output_port_labels["T_h_out"], 0)
        self.assertEqual(hx.output_port_labels["T_c_out"], 1)
        # per-cell ports
        self.assertEqual(hx.output_port_labels["T_h_1"], 2)
        self.assertEqual(hx.output_port_labels["T_c_1"], 7)
        self.assertEqual(hx.output_port_labels["T_h_5"], 6)
        self.assertEqual(hx.output_port_labels["T_c_5"], 11)

    def test_state_size(self):
        """Test that state vector has correct size."""
        for N in [1, 3, 10]:
            H = HeatExchanger(N_cells=N)
            H.set_solver(EUF, parent=None)
            self.assertEqual(len(H.engine.initial_value), 2 * N)

    def test_output_initial(self):
        """Test outputs at initial state."""
        H = HeatExchanger(N_cells=3, T_h0=400.0, T_c0=300.0)
        H.set_solver(EUF, parent=None)
        H.update(None)

        # Hot outlet = last hot cell = T_h0 initially
        self.assertAlmostEqual(H.outputs[0], 400.0)
        # Cold outlet = first cold cell = T_c0 initially
        self.assertAlmostEqual(H.outputs[1], 300.0)

    def test_energy_conservation_direction(self):
        """Test that heat flows from hot to cold (derivatives have correct sign)."""
        H = HeatExchanger(N_cells=1, F_h=0.1, F_c=0.1, V_h=0.5, V_c=0.5,
                          UA=500.0, T_h0=400.0, T_c0=300.0)
        H.set_solver(EUF, parent=None)

        # Set inlet temps equal to initial (no flow effects, only heat transfer)
        H.inputs[0] = 400.0  # T_h_in
        H.inputs[1] = 300.0  # T_c_in

        # Evaluate dynamics
        x = H.engine.get()
        u = np.array([400.0, 300.0])
        dx = H.op_dyn(x, u, 0)

        # Hot side should cool (negative dT_h) due to heat transfer
        # Cold side should warm (positive dT_c) due to heat transfer
        # With flow terms = 0 (since inlet = state), only UA term matters
        self.assertLess(dx[0], 0)    # dT_h < 0 (cooling)
        self.assertGreater(dx[1], 0)  # dT_c > 0 (warming)

    def test_no_transfer_equal_temps(self):
        """When hot and cold are at same temperature, no heat transfer occurs."""
        T_eq = 350.0
        H = HeatExchanger(N_cells=2, T_h0=T_eq, T_c0=T_eq)
        H.set_solver(EUF, parent=None)

        H.inputs[0] = T_eq
        H.inputs[1] = T_eq

        x = H.engine.get()
        u = np.array([T_eq, T_eq])
        dx = H.op_dyn(x, u, 0)

        # All derivatives should be zero
        self.assertTrue(np.allclose(dx, 0.0, atol=1e-10))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
