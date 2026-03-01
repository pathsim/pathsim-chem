########################################################################################
##
##                                  TESTS FOR
##                             'process.valve.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.process import Valve


# TESTS ================================================================================

class TestValve(unittest.TestCase):
    """Test the pressure-drop valve block."""

    def test_init_default(self):
        """Test default Cv."""
        V = Valve()
        self.assertEqual(V.Cv, 1.0)

    def test_init_custom(self):
        """Test custom Cv."""
        V = Valve(Cv=5.0)
        self.assertEqual(V.Cv, 5.0)

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            Valve(Cv=0)
        with self.assertRaises(ValueError):
            Valve(Cv=-1)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(Valve.input_port_labels["P_in"], 0)
        self.assertEqual(Valve.input_port_labels["P_out"], 1)
        self.assertEqual(Valve.input_port_labels["T_in"], 2)
        self.assertEqual(Valve.output_port_labels["F"], 0)
        self.assertEqual(Valve.output_port_labels["T_out"], 1)

    def test_forward_flow(self):
        """Flow should be positive when P_in > P_out."""
        V = Valve(Cv=2.0)
        V.inputs[0] = 200000.0  # P_in
        V.inputs[1] = 100000.0  # P_out
        V.inputs[2] = 350.0     # T_in

        V.update(None)

        F = V.outputs[0]
        self.assertGreater(F, 0)
        expected = 2.0 * np.sqrt(100000.0)
        self.assertAlmostEqual(F, expected, places=4)

    def test_reverse_flow(self):
        """Flow should be negative when P_in < P_out."""
        V = Valve(Cv=2.0)
        V.inputs[0] = 100000.0  # P_in
        V.inputs[1] = 200000.0  # P_out
        V.inputs[2] = 350.0

        V.update(None)

        F = V.outputs[0]
        self.assertLess(F, 0)

    def test_zero_dp(self):
        """No flow when pressures are equal."""
        V = Valve(Cv=2.0)
        V.inputs[0] = 101325.0
        V.inputs[1] = 101325.0
        V.inputs[2] = 350.0

        V.update(None)

        self.assertAlmostEqual(V.outputs[0], 0.0)

    def test_temperature_passthrough(self):
        """Temperature should pass through unchanged (isenthalpic)."""
        V = Valve()
        V.inputs[0] = 200000.0
        V.inputs[1] = 100000.0
        V.inputs[2] = 375.0

        V.update(None)

        self.assertAlmostEqual(V.outputs[1], 375.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
