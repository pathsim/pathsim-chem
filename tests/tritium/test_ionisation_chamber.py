########################################################################################
##
##                                  TESTS FOR
##                     'tritium.ionisation_chamber.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

from pathsim_chem.tritium import IonisationChamber


# TESTS ================================================================================

class TestIonisationChamber(unittest.TestCase):
    """Test the IonisationChamber block."""

    def test_init_constant_efficiency(self):
        """Test initialization with constant detection efficiency."""
        ic = IonisationChamber(detection_efficiency=0.8)
        self.assertEqual(ic.detection_efficiency, 0.8)
        self.assertIsNone(ic.detection_threshold)

    def test_init_threshold(self):
        """Test initialization with detection threshold."""
        ic = IonisationChamber(detection_threshold=10.0)
        self.assertEqual(ic.detection_threshold, 10.0)
        self.assertTrue(callable(ic.detection_efficiency))

    def test_init_callable_efficiency(self):
        """Test initialization with callable detection efficiency."""
        eff = lambda c: min(c / 100.0, 1.0)
        ic = IonisationChamber(detection_efficiency=eff)
        self.assertIs(ic.detection_efficiency, eff)

    def test_init_validation_both(self):
        """Providing both parameters should raise ValueError."""
        with self.assertRaises(ValueError):
            IonisationChamber(detection_efficiency=0.5, detection_threshold=10.0)

    def test_init_validation_neither(self):
        """Providing neither parameter should raise ValueError."""
        with self.assertRaises(ValueError):
            IonisationChamber()

    def test_port_labels(self):
        """Test port label definitions."""
        self.assertEqual(IonisationChamber.input_port_labels["flux_in"], 0)
        self.assertEqual(IonisationChamber.input_port_labels["flow_rate"], 1)
        self.assertEqual(IonisationChamber.output_port_labels["flux_out"], 0)
        self.assertEqual(IonisationChamber.output_port_labels["signal"], 1)

    def test_passthrough(self):
        """Sample flux passes through unchanged."""
        ic = IonisationChamber(detection_efficiency=0.5)
        ic.inputs[0] = 100.0  # flux_in
        ic.inputs[1] = 10.0   # flow_rate
        ic.update(None)

        self.assertAlmostEqual(ic.outputs[0], 100.0)

    def test_signal_constant_efficiency(self):
        """Signal = efficiency * concentration."""
        ic = IonisationChamber(detection_efficiency=0.8)
        ic.inputs[0] = 200.0  # flux_in
        ic.inputs[1] = 10.0   # flow_rate -> concentration = 20
        ic.update(None)

        self.assertAlmostEqual(ic.outputs[1], 0.8 * 20.0)

    def test_signal_threshold_above(self):
        """Above threshold, signal = concentration."""
        ic = IonisationChamber(detection_threshold=5.0)
        ic.inputs[0] = 100.0  # flux
        ic.inputs[1] = 10.0   # flow -> concentration = 10 > 5
        ic.update(None)

        self.assertAlmostEqual(ic.outputs[1], 10.0)

    def test_signal_threshold_below(self):
        """Below threshold, signal = 0."""
        ic = IonisationChamber(detection_threshold=50.0)
        ic.inputs[0] = 100.0  # flux
        ic.inputs[1] = 10.0   # flow -> concentration = 10 < 50
        ic.update(None)

        self.assertAlmostEqual(ic.outputs[1], 0.0)

    def test_signal_callable_efficiency(self):
        """Callable efficiency applied to concentration."""
        # Linear ramp: efficiency = c / 100, capped at 1
        eff = lambda c: min(c / 100.0, 1.0)
        ic = IonisationChamber(detection_efficiency=eff)
        ic.inputs[0] = 500.0  # flux
        ic.inputs[1] = 10.0   # flow -> concentration = 50
        ic.update(None)

        # efficiency(50) = 0.5, signal = 0.5 * 50 = 25
        self.assertAlmostEqual(ic.outputs[1], 25.0)

    def test_zero_flow_rate(self):
        """Zero flow rate should not crash, signal = 0."""
        ic = IonisationChamber(detection_efficiency=1.0)
        ic.inputs[0] = 100.0
        ic.inputs[1] = 0.0
        ic.update(None)

        self.assertAlmostEqual(ic.outputs[0], 100.0)
        self.assertAlmostEqual(ic.outputs[1], 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
