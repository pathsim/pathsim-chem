########################################################################################
##
##                                  TESTS FOR
##                  'thermodynamics.equations_of_state.py'
##
##    IK-CAPE Equations of State (Chapter 7)
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.thermodynamics import (
    PengRobinson,
    RedlichKwongSoave,
)


# CONSTANTS ============================================================================

R = 8.314462


# HELPERS ==============================================================================

def eval_block(block, T, P):
    """Set T and P inputs, update, return (v, z)."""
    block.inputs[0] = T
    block.inputs[1] = P
    block.update(None)
    return block.outputs[0], block.outputs[1]


# TESTS ================================================================================

class TestPengRobinson(unittest.TestCase):

    def test_init_pure(self):
        # Methane: Tc=190.6K, Pc=4.6MPa, omega=0.011
        PR = PengRobinson(Tc=190.6, Pc=4.6e6, omega=0.011)
        self.assertEqual(PR.n, 1)

    def test_ideal_gas_limit(self):
        # At low pressure and high T, Z -> 1 and v -> RT/P
        PR = PengRobinson(Tc=190.6, Pc=4.6e6, omega=0.011)
        T, P = 1000, 100  # very high T, very low P
        v, z = eval_block(PR, T, P)
        self.assertAlmostEqual(z, 1.0, delta=0.01)
        expected_v = R * T / P
        self.assertAlmostEqual(v, expected_v, delta=expected_v * 0.01)

    def test_methane_gas(self):
        # Methane at 300K, 1 atm - should behave as near-ideal gas
        PR = PengRobinson(Tc=190.6, Pc=4.6e6, omega=0.011)
        v, z = eval_block(PR, 300, 101325)
        self.assertAlmostEqual(z, 1.0, delta=0.05)
        self.assertGreater(v, 0)

    def test_water_vapor(self):
        # Water at 500K, 1 atm
        PR = PengRobinson(Tc=647.096, Pc=22064000, omega=0.3443)
        v, z = eval_block(PR, 500, 101325)
        self.assertAlmostEqual(z, 1.0, delta=0.05)

    def test_liquid_phase(self):
        # Methane below Tc at high pressure - liquid phase
        PR = PengRobinson(Tc=190.6, Pc=4.6e6, omega=0.011, phase="liquid")
        v, z = eval_block(PR, 120, 5e6)
        self.assertLess(z, 0.5)
        self.assertGreater(v, 0)

    def test_mixture(self):
        # Methane + ethane mixture
        PR = PengRobinson(
            Tc=[190.6, 305.3],
            Pc=[4.6e6, 4.872e6],
            omega=[0.011, 0.099],
            x=[0.7, 0.3],
        )
        v, z = eval_block(PR, 300, 101325)
        self.assertGreater(z, 0.9)
        self.assertGreater(v, 0)

    def test_at_critical_point(self):
        # At Tc, Pc => Z should be near 0.307 for PR
        PR = PengRobinson(Tc=190.6, Pc=4.6e6, omega=0.011)
        v, z = eval_block(PR, 190.6, 4.6e6)
        self.assertAlmostEqual(z, 0.307, delta=0.05)


class TestRedlichKwongSoave(unittest.TestCase):

    def test_init_pure(self):
        RKS = RedlichKwongSoave(Tc=190.6, Pc=4.6e6, omega=0.011)
        self.assertEqual(RKS.n, 1)

    def test_ideal_gas_limit(self):
        RKS = RedlichKwongSoave(Tc=190.6, Pc=4.6e6, omega=0.011)
        T, P = 1000, 100
        v, z = eval_block(RKS, T, P)
        self.assertAlmostEqual(z, 1.0, delta=0.01)

    def test_methane_gas(self):
        RKS = RedlichKwongSoave(Tc=190.6, Pc=4.6e6, omega=0.011)
        v, z = eval_block(RKS, 300, 101325)
        self.assertAlmostEqual(z, 1.0, delta=0.05)
        self.assertGreater(v, 0)

    def test_liquid_phase(self):
        RKS = RedlichKwongSoave(Tc=190.6, Pc=4.6e6, omega=0.011, phase="liquid")
        v, z = eval_block(RKS, 120, 5e6)
        self.assertLess(z, 0.5)
        self.assertGreater(v, 0)

    def test_mixture(self):
        RKS = RedlichKwongSoave(
            Tc=[190.6, 305.3],
            Pc=[4.6e6, 4.872e6],
            omega=[0.011, 0.099],
            x=[0.7, 0.3],
        )
        v, z = eval_block(RKS, 300, 101325)
        self.assertGreater(z, 0.9)
        self.assertGreater(v, 0)

    def test_pr_vs_rks_similar(self):
        # Both EoS should give similar results for ideal gas conditions
        PR = PengRobinson(Tc=190.6, Pc=4.6e6, omega=0.011)
        RKS = RedlichKwongSoave(Tc=190.6, Pc=4.6e6, omega=0.011)

        v_pr, z_pr = eval_block(PR, 300, 101325)
        v_rks, z_rks = eval_block(RKS, 300, 101325)

        self.assertAlmostEqual(z_pr, z_rks, delta=0.01)

    def test_at_critical_point(self):
        # At Tc, Pc => Z should be near 1/3 for RKS
        RKS = RedlichKwongSoave(Tc=190.6, Pc=4.6e6, omega=0.011)
        v, z = eval_block(RKS, 190.6, 4.6e6)
        self.assertAlmostEqual(z, 0.333, delta=0.05)


# SMOKE TEST ===========================================================================

class TestSmokeAllEoS(unittest.TestCase):

    def test_all_blocks(self):
        blocks = [
            PengRobinson(Tc=190.6, Pc=4.6e6, omega=0.011),
            PengRobinson(Tc=190.6, Pc=4.6e6, omega=0.011, phase="liquid"),
            RedlichKwongSoave(Tc=190.6, Pc=4.6e6, omega=0.011),
            RedlichKwongSoave(Tc=190.6, Pc=4.6e6, omega=0.011, phase="liquid"),
        ]

        for block in blocks:
            with self.subTest(block=f"{block.__class__.__name__}({block.phase})"):
                v, z = eval_block(block, 200, 1e6)
                self.assertTrue(np.isfinite(v))
                self.assertTrue(np.isfinite(z))
                self.assertGreater(v, 0)
                self.assertGreater(z, 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
