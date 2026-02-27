########################################################################################
##
##                                  TESTS FOR
##                         'process.flash_drum.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.process import FlashDrum

from pathsim.solvers import EUF


# TESTS ================================================================================

class TestFlashDrum(unittest.TestCase):
    """Test the binary isothermal flash drum block."""

    def test_init_default(self):
        """Test default initialization."""
        F = FlashDrum()
        self.assertEqual(F.holdup, 100.0)
        self.assertEqual(len(F.antoine_A), 2)
        self.assertEqual(len(F.antoine_B), 2)
        self.assertEqual(len(F.antoine_C), 2)

    def test_init_custom(self):
        """Test custom initialization."""
        F = FlashDrum(holdup=200.0,
                      antoine_A=[15.0, 16.0],
                      antoine_B=[2800.0, 3100.0],
                      antoine_C=[-50.0, -55.0],
                      N0=[120.0, 80.0])
        self.assertEqual(F.holdup, 200.0)

        F.set_solver(EUF, parent=None)
        self.assertTrue(np.allclose(F.engine.initial_value, [120.0, 80.0]))

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            FlashDrum(holdup=0)
        with self.assertRaises(ValueError):
            FlashDrum(holdup=-10)
        with self.assertRaises(ValueError):
            FlashDrum(antoine_A=[1.0, 2.0, 3.0])  # not binary

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(FlashDrum.input_port_labels["F"], 0)
        self.assertEqual(FlashDrum.input_port_labels["z_1"], 1)
        self.assertEqual(FlashDrum.input_port_labels["T"], 2)
        self.assertEqual(FlashDrum.input_port_labels["P"], 3)
        self.assertEqual(FlashDrum.output_port_labels["V_rate"], 0)
        self.assertEqual(FlashDrum.output_port_labels["L_rate"], 1)
        self.assertEqual(FlashDrum.output_port_labels["y_1"], 2)
        self.assertEqual(FlashDrum.output_port_labels["x_1"], 3)

    def test_default_holdup(self):
        """Default holdup should split equally between components."""
        F = FlashDrum(holdup=100.0)
        F.set_solver(EUF, parent=None)
        self.assertTrue(np.allclose(F.engine.initial_value, [50.0, 50.0]))

    def test_output_flow_balance(self):
        """V_rate + L_rate should equal F_in."""
        F = FlashDrum()
        F.set_solver(EUF, parent=None)

        F_in = 10.0
        F.inputs[0] = F_in    # F
        F.inputs[1] = 0.5     # z_1
        F.inputs[2] = 360.0   # T [K]
        F.inputs[3] = 101325.0  # P [Pa]

        F.update(None)

        V_rate = F.outputs[0]
        L_rate = F.outputs[1]
        self.assertAlmostEqual(V_rate + L_rate, F_in, places=8)

    def test_vle_consistency(self):
        """y_1 and x_1 should be in [0, 1]."""
        F = FlashDrum()
        F.set_solver(EUF, parent=None)

        F.inputs[0] = 10.0      # F
        F.inputs[1] = 0.5       # z_1
        F.inputs[2] = 360.0     # T [K]
        F.inputs[3] = 101325.0  # P [Pa]

        F.update(None)

        y_1 = F.outputs[2]
        x_1 = F.outputs[3]
        self.assertGreaterEqual(y_1, 0.0)
        self.assertLessEqual(y_1, 1.0)
        self.assertGreaterEqual(x_1, 0.0)
        self.assertLessEqual(x_1, 1.0)

    def test_light_component_enriched_in_vapor(self):
        """The more volatile component should be enriched in vapor phase."""
        # Default Antoine params: component 1 has lower B (higher Psat) => more volatile
        F = FlashDrum()
        F.set_solver(EUF, parent=None)

        F.inputs[0] = 10.0
        F.inputs[1] = 0.5
        F.inputs[2] = 360.0
        F.inputs[3] = 101325.0

        F.update(None)

        y_1 = F.outputs[2]
        x_1 = F.outputs[3]

        # y_1 > x_1 for the light (more volatile) component
        self.assertGreater(y_1, x_1)

    def test_no_feed(self):
        """With zero feed, rates should be zero."""
        F = FlashDrum()
        F.set_solver(EUF, parent=None)

        F.inputs[0] = 0.0      # F = 0
        F.inputs[1] = 0.5
        F.inputs[2] = 350.0
        F.inputs[3] = 101325.0

        F.update(None)

        self.assertAlmostEqual(F.outputs[0], 0.0)  # V_rate
        self.assertAlmostEqual(F.outputs[1], 0.0)  # L_rate


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
