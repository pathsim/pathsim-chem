########################################################################################
##
##                                  TESTS FOR
##                             'process.mixer.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

from pathsim_chem.process import Mixer


# TESTS ================================================================================

class TestMixer(unittest.TestCase):
    """Test the 2-stream mixer block."""

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(Mixer.input_port_labels["F_1"], 0)
        self.assertEqual(Mixer.input_port_labels["T_1"], 1)
        self.assertEqual(Mixer.input_port_labels["F_2"], 2)
        self.assertEqual(Mixer.input_port_labels["T_2"], 3)
        self.assertEqual(Mixer.output_port_labels["F_out"], 0)
        self.assertEqual(Mixer.output_port_labels["T_out"], 1)

    def test_mass_balance(self):
        """F_out should equal F_1 + F_2."""
        M = Mixer()
        M.inputs[0] = 5.0    # F_1
        M.inputs[1] = 350.0  # T_1
        M.inputs[2] = 3.0    # F_2
        M.inputs[3] = 300.0  # T_2

        M.update(None)

        self.assertAlmostEqual(M.outputs[0], 8.0)  # F_out

    def test_energy_balance(self):
        """F_1*T_1 + F_2*T_2 should equal F_out*T_out."""
        M = Mixer()
        F_1, T_1 = 5.0, 350.0
        F_2, T_2 = 3.0, 300.0

        M.inputs[0] = F_1
        M.inputs[1] = T_1
        M.inputs[2] = F_2
        M.inputs[3] = T_2

        M.update(None)

        F_out = M.outputs[0]
        T_out = M.outputs[1]
        self.assertAlmostEqual(F_1 * T_1 + F_2 * T_2, F_out * T_out, places=8)

    def test_equal_temperatures(self):
        """When both streams at same T, output T should match."""
        M = Mixer()
        M.inputs[0] = 5.0
        M.inputs[1] = 350.0
        M.inputs[2] = 3.0
        M.inputs[3] = 350.0

        M.update(None)

        self.assertAlmostEqual(M.outputs[1], 350.0)

    def test_zero_flow_one_stream(self):
        """With one stream at zero flow, output should match the other."""
        M = Mixer()
        M.inputs[0] = 0.0
        M.inputs[1] = 300.0
        M.inputs[2] = 5.0
        M.inputs[3] = 400.0

        M.update(None)

        self.assertAlmostEqual(M.outputs[0], 5.0)
        self.assertAlmostEqual(M.outputs[1], 400.0)

    def test_zero_total_flow(self):
        """With zero total flow, should not crash."""
        M = Mixer()
        M.inputs[0] = 0.0
        M.inputs[1] = 300.0
        M.inputs[2] = 0.0
        M.inputs[3] = 400.0

        M.update(None)

        self.assertAlmostEqual(M.outputs[0], 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
