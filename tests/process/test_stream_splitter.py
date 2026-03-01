########################################################################################
##
##                                  TESTS FOR
##                        'process.stream_splitter.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

from pathsim_chem.process import StreamSplitter


# TESTS ================================================================================

class TestStreamSplitter(unittest.TestCase):
    """Test the stream splitter block."""

    def test_init_default(self):
        """Test default split ratio."""
        S = StreamSplitter()
        self.assertEqual(S.split, 0.5)

    def test_init_custom(self):
        """Test custom split ratio."""
        S = StreamSplitter(split=0.3)
        self.assertEqual(S.split, 0.3)

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            StreamSplitter(split=-0.1)
        with self.assertRaises(ValueError):
            StreamSplitter(split=1.5)

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(StreamSplitter.input_port_labels["F_in"], 0)
        self.assertEqual(StreamSplitter.input_port_labels["T_in"], 1)
        self.assertEqual(StreamSplitter.output_port_labels["F_1"], 0)
        self.assertEqual(StreamSplitter.output_port_labels["T_1"], 1)
        self.assertEqual(StreamSplitter.output_port_labels["F_2"], 2)
        self.assertEqual(StreamSplitter.output_port_labels["T_2"], 3)

    def test_mass_balance(self):
        """F_1 + F_2 should equal F_in."""
        S = StreamSplitter(split=0.7)
        S.inputs[0] = 10.0   # F_in
        S.inputs[1] = 350.0  # T_in

        S.update(None)

        F_1 = S.outputs[0]
        F_2 = S.outputs[2]
        self.assertAlmostEqual(F_1 + F_2, 10.0, places=8)

    def test_split_ratio(self):
        """Check split produces correct flow fractions."""
        S = StreamSplitter(split=0.3)
        S.inputs[0] = 10.0
        S.inputs[1] = 350.0

        S.update(None)

        self.assertAlmostEqual(S.outputs[0], 3.0)   # F_1 = 0.3 * 10
        self.assertAlmostEqual(S.outputs[2], 7.0)   # F_2 = 0.7 * 10

    def test_temperature_unchanged(self):
        """Both outlet temperatures should equal inlet."""
        S = StreamSplitter(split=0.4)
        S.inputs[0] = 10.0
        S.inputs[1] = 375.0

        S.update(None)

        self.assertAlmostEqual(S.outputs[1], 375.0)  # T_1
        self.assertAlmostEqual(S.outputs[3], 375.0)  # T_2

    def test_split_zero(self):
        """With split=0, all flow goes to second outlet."""
        S = StreamSplitter(split=0.0)
        S.inputs[0] = 10.0
        S.inputs[1] = 350.0

        S.update(None)

        self.assertAlmostEqual(S.outputs[0], 0.0)   # F_1
        self.assertAlmostEqual(S.outputs[2], 10.0)  # F_2

    def test_split_one(self):
        """With split=1, all flow goes to first outlet."""
        S = StreamSplitter(split=1.0)
        S.inputs[0] = 10.0
        S.inputs[1] = 350.0

        S.update(None)

        self.assertAlmostEqual(S.outputs[0], 10.0)  # F_1
        self.assertAlmostEqual(S.outputs[2], 0.0)   # F_2


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
