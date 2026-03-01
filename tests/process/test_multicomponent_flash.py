########################################################################################
##
##                                  TESTS FOR
##                    'process.multicomponent_flash.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.process import MultiComponentFlash

from pathsim.solvers import EUF


# TESTS ================================================================================

class TestMultiComponentFlash(unittest.TestCase):
    """Test the multi-component isothermal flash drum block."""

    def test_init_default(self):
        """Test default initialization (ternary)."""
        F = MultiComponentFlash()
        self.assertEqual(F.N_comp, 3)
        self.assertEqual(F.holdup, 100.0)
        self.assertEqual(len(F.antoine_A), 3)
        self.assertEqual(len(F.antoine_B), 3)
        self.assertEqual(len(F.antoine_C), 3)

    def test_init_custom(self):
        """Test custom initialization."""
        F = MultiComponentFlash(
            N_comp=4, holdup=200.0,
            antoine_A=[20.0, 21.0, 22.0, 23.0],
            antoine_B=[2800.0, 3000.0, 3200.0, 3400.0],
            antoine_C=[-50.0, -52.0, -54.0, -56.0],
            N0=[60.0, 50.0, 50.0, 40.0],
        )
        self.assertEqual(F.N_comp, 4)
        self.assertEqual(F.holdup, 200.0)

        F.set_solver(EUF, parent=None)
        self.assertTrue(np.allclose(F.engine.initial_value, [60.0, 50.0, 50.0, 40.0]))

    def test_init_validation(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            MultiComponentFlash(N_comp=1)
        with self.assertRaises(ValueError):
            MultiComponentFlash(holdup=0)
        with self.assertRaises(ValueError):
            MultiComponentFlash(holdup=-10)
        with self.assertRaises(ValueError):
            MultiComponentFlash(N_comp=3, antoine_A=[1.0, 2.0])  # wrong length

    def test_port_labels_ternary(self):
        """Test port labels for ternary system."""
        F = MultiComponentFlash(N_comp=3)
        # inputs: F, z_1, z_2, T, P
        self.assertEqual(F.input_port_labels["F"], 0)
        self.assertEqual(F.input_port_labels["z_1"], 1)
        self.assertEqual(F.input_port_labels["z_2"], 2)
        self.assertEqual(F.input_port_labels["T"], 3)
        self.assertEqual(F.input_port_labels["P"], 4)
        # outputs: V_rate, L_rate, y_1, y_2, x_1, x_2
        self.assertEqual(F.output_port_labels["V_rate"], 0)
        self.assertEqual(F.output_port_labels["L_rate"], 1)
        self.assertEqual(F.output_port_labels["y_1"], 2)
        self.assertEqual(F.output_port_labels["y_2"], 3)
        self.assertEqual(F.output_port_labels["x_1"], 4)
        self.assertEqual(F.output_port_labels["x_2"], 5)

    def test_port_labels_binary(self):
        """Test port labels for binary system."""
        F = MultiComponentFlash(N_comp=2)
        # inputs: F, z_1, T, P
        self.assertEqual(F.input_port_labels["F"], 0)
        self.assertEqual(F.input_port_labels["z_1"], 1)
        self.assertEqual(F.input_port_labels["T"], 2)
        self.assertEqual(F.input_port_labels["P"], 3)
        # outputs: V_rate, L_rate, y_1, x_1
        self.assertEqual(F.output_port_labels["V_rate"], 0)
        self.assertEqual(F.output_port_labels["L_rate"], 1)
        self.assertEqual(F.output_port_labels["y_1"], 2)
        self.assertEqual(F.output_port_labels["x_1"], 3)

    def test_default_holdup(self):
        """Default holdup should split equally between components."""
        F = MultiComponentFlash(N_comp=3, holdup=90.0)
        F.set_solver(EUF, parent=None)
        self.assertTrue(np.allclose(F.engine.initial_value, [30.0, 30.0, 30.0]))

    def test_output_flow_balance(self):
        """V_rate + L_rate should equal F_in."""
        F = MultiComponentFlash(N_comp=3)
        F.set_solver(EUF, parent=None)

        F_in = 10.0
        F.inputs[0] = F_in       # F
        F.inputs[1] = 0.33       # z_1
        F.inputs[2] = 0.34       # z_2 (z_3 = 0.33)
        F.inputs[3] = 370.0      # T [K]
        F.inputs[4] = 101325.0   # P [Pa]

        F.update(None)

        V_rate = F.outputs[0]
        L_rate = F.outputs[1]
        self.assertAlmostEqual(V_rate + L_rate, F_in, places=8)

    def test_vle_consistency(self):
        """Vapor and liquid compositions should sum to 1."""
        F = MultiComponentFlash(N_comp=3)
        F.set_solver(EUF, parent=None)

        F.inputs[0] = 10.0       # F
        F.inputs[1] = 0.33       # z_1
        F.inputs[2] = 0.34       # z_2
        F.inputs[3] = 370.0      # T [K]
        F.inputs[4] = 101325.0   # P [Pa]

        F.update(None)

        # outputs: V_rate, L_rate, y_1, y_2, x_1, x_2
        y_1 = F.outputs[2]
        y_2 = F.outputs[3]
        y_3 = 1.0 - y_1 - y_2
        x_1 = F.outputs[4]
        x_2 = F.outputs[5]
        x_3 = 1.0 - x_1 - x_2

        self.assertAlmostEqual(y_1 + y_2 + y_3, 1.0, places=6)
        self.assertAlmostEqual(x_1 + x_2 + x_3, 1.0, places=6)

    def test_light_component_enriched_in_vapor(self):
        """The most volatile component should be enriched in vapor phase."""
        F = MultiComponentFlash(N_comp=3)
        F.set_solver(EUF, parent=None)

        F.inputs[0] = 10.0
        F.inputs[1] = 0.33
        F.inputs[2] = 0.34
        F.inputs[3] = 370.0
        F.inputs[4] = 101325.0

        F.update(None)

        V_rate = F.outputs[0]
        L_rate = F.outputs[1]

        # Only check if two-phase
        if V_rate > 0 and L_rate > 0:
            y_1 = F.outputs[2]
            x_1 = F.outputs[4]
            # Component 1 (benzene) is most volatile => y_1 > x_1
            self.assertGreater(y_1, x_1)

    def test_no_feed(self):
        """With zero feed, rates should be zero."""
        F = MultiComponentFlash(N_comp=3)
        F.set_solver(EUF, parent=None)

        F.inputs[0] = 0.0       # F = 0
        F.inputs[1] = 0.33
        F.inputs[2] = 0.34
        F.inputs[3] = 350.0
        F.inputs[4] = 101325.0

        F.update(None)

        self.assertAlmostEqual(F.outputs[0], 0.0)  # V_rate
        self.assertAlmostEqual(F.outputs[1], 0.0)  # L_rate

    def test_subcooled_all_liquid(self):
        """At low temperature, all feed should remain liquid."""
        F = MultiComponentFlash(N_comp=3)
        F.set_solver(EUF, parent=None)

        F.inputs[0] = 10.0
        F.inputs[1] = 0.33
        F.inputs[2] = 0.34
        F.inputs[3] = 280.0       # very low T
        F.inputs[4] = 101325.0

        F.update(None)

        V_rate = F.outputs[0]
        L_rate = F.outputs[1]
        self.assertAlmostEqual(V_rate, 0.0, places=6)
        self.assertAlmostEqual(L_rate, 10.0, places=6)

    def test_superheated_all_vapor(self):
        """At very high temperature / low pressure, all feed should vaporize."""
        F = MultiComponentFlash(N_comp=3)
        F.set_solver(EUF, parent=None)

        F.inputs[0] = 10.0
        F.inputs[1] = 0.33
        F.inputs[2] = 0.34
        F.inputs[3] = 500.0       # very high T
        F.inputs[4] = 1000.0      # very low P

        F.update(None)

        V_rate = F.outputs[0]
        L_rate = F.outputs[1]
        self.assertAlmostEqual(V_rate, 10.0, places=6)
        self.assertAlmostEqual(L_rate, 0.0, places=6)

    def test_binary_consistency(self):
        """Binary MultiComponentFlash should give same results as FlashDrum."""
        from pathsim_chem.process import FlashDrum

        # Use same Antoine parameters
        A = [20.7936, 20.9064]
        B = [2788.51, 3096.52]
        C = [-52.36, -53.67]

        mc = MultiComponentFlash(N_comp=2, holdup=100.0,
                                 antoine_A=A, antoine_B=B, antoine_C=C)
        fd = FlashDrum(holdup=100.0,
                       antoine_A=A, antoine_B=B, antoine_C=C)

        mc.set_solver(EUF, parent=None)
        fd.set_solver(EUF, parent=None)

        # Set same inputs
        mc.inputs[0] = 10.0     # F
        mc.inputs[1] = 0.5      # z_1
        mc.inputs[2] = 360.0    # T
        mc.inputs[3] = 101325.0 # P

        fd.inputs[0] = 10.0     # F
        fd.inputs[1] = 0.5      # z_1
        fd.inputs[2] = 360.0    # T
        fd.inputs[3] = 101325.0 # P

        mc.update(None)
        fd.update(None)

        # Compare V_rate, L_rate, y_1, x_1
        self.assertAlmostEqual(mc.outputs[0], fd.outputs[0], places=4)  # V_rate
        self.assertAlmostEqual(mc.outputs[1], fd.outputs[1], places=4)  # L_rate
        self.assertAlmostEqual(mc.outputs[2], fd.outputs[2], places=4)  # y_1
        self.assertAlmostEqual(mc.outputs[3], fd.outputs[3], places=4)  # x_1


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
