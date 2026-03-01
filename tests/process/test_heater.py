########################################################################################
##
##                                  TESTS FOR
##                            'process.heater.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

from pathsim_chem.process import Heater


# TESTS ================================================================================

class TestHeater(unittest.TestCase):
    """Test the duty-specified heater/cooler block."""

    def test_init_default(self):
        """Test default parameters."""
        H = Heater()
        self.assertEqual(H.rho, 1000.0)
        self.assertEqual(H.Cp, 4184.0)

    def test_init_custom(self):
        """Test custom parameters."""
        H = Heater(rho=800.0, Cp=3000.0)
        self.assertEqual(H.rho, 800.0)
        self.assertEqual(H.Cp, 3000.0)

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            Heater(rho=0)
        with self.assertRaises(ValueError):
            Heater(rho=-1)
        with self.assertRaises(ValueError):
            Heater(Cp=0)
        with self.assertRaises(ValueError):
            Heater(Cp=-1)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(Heater.input_port_labels["F"], 0)
        self.assertEqual(Heater.input_port_labels["T_in"], 1)
        self.assertEqual(Heater.input_port_labels["Q"], 2)
        self.assertEqual(Heater.output_port_labels["F_out"], 0)
        self.assertEqual(Heater.output_port_labels["T_out"], 1)

    def test_heating(self):
        """Positive Q should increase temperature."""
        H = Heater(rho=1000.0, Cp=4184.0)
        F = 0.1  # m³/s
        T_in = 300.0
        Q = 41840.0  # W

        H.inputs[0] = F
        H.inputs[1] = T_in
        H.inputs[2] = Q

        H.update(None)

        # dT = Q / (F * rho * Cp) = 41840 / (0.1 * 1000 * 4184) = 0.1 K
        expected_T = T_in + Q / (F * 1000.0 * 4184.0)
        self.assertAlmostEqual(H.outputs[1], expected_T, places=6)

    def test_cooling(self):
        """Negative Q should decrease temperature."""
        H = Heater(rho=1000.0, Cp=4184.0)
        H.inputs[0] = 0.1
        H.inputs[1] = 350.0
        H.inputs[2] = -41840.0

        H.update(None)

        self.assertLess(H.outputs[1], 350.0)

    def test_zero_duty(self):
        """Q=0 should pass temperature through unchanged."""
        H = Heater()
        H.inputs[0] = 0.1
        H.inputs[1] = 350.0
        H.inputs[2] = 0.0

        H.update(None)

        self.assertAlmostEqual(H.outputs[1], 350.0)

    def test_flow_passthrough(self):
        """F_out should equal F_in."""
        H = Heater()
        H.inputs[0] = 0.5
        H.inputs[1] = 300.0
        H.inputs[2] = 10000.0

        H.update(None)

        self.assertAlmostEqual(H.outputs[0], 0.5)

    def test_zero_flow(self):
        """With zero flow, temperature should remain unchanged."""
        H = Heater()
        H.inputs[0] = 0.0
        H.inputs[1] = 350.0
        H.inputs[2] = 10000.0

        H.update(None)

        self.assertAlmostEqual(H.outputs[0], 0.0)
        self.assertAlmostEqual(H.outputs[1], 350.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
