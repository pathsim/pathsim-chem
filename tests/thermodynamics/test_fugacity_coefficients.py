########################################################################################
##
##                                  TESTS FOR
##              'thermodynamics.fugacity_coefficients.py'
##
##    IK-CAPE Fugacity Coefficients (Chapter 8)
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.thermodynamics import (
    FugacityRKS,
    FugacityPR,
    FugacityVirial,
)


# CONSTANTS ============================================================================

R = 8.314462


# HELPERS ==============================================================================

def eval_block(block, T, P):
    """Set T and P inputs, update, return outputs."""
    block.inputs[0] = T
    block.inputs[1] = P
    block.update(None)
    n_out = len([k for k in block.outputs])
    if n_out == 1:
        return (block.outputs[0],)
    return tuple(block.outputs[i] for i in range(n_out))


# TESTS ================================================================================

class TestFugacityRKS(unittest.TestCase):

    def test_init_pure(self):
        F = FugacityRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        self.assertEqual(F.nc, 1)

    def test_ideal_gas_limit(self):
        # At very low P and high T, phi -> 1
        F = FugacityRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        phi = eval_block(F, 1000, 100)
        self.assertAlmostEqual(phi[0], 1.0, delta=0.01)

    def test_methane_moderate(self):
        # Methane at 300K, 1 atm - phi close to 1
        F = FugacityRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        phi = eval_block(F, 300, 101325)
        self.assertAlmostEqual(phi[0], 1.0, delta=0.05)
        self.assertGreater(phi[0], 0)

    def test_high_pressure(self):
        # At higher pressure, phi deviates from 1
        F = FugacityRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        phi_low = eval_block(F, 300, 101325)
        phi_high = eval_block(F, 300, 3e6)
        # phi should deviate more at high pressure
        self.assertGreater(abs(phi_high[0] - 1.0), abs(phi_low[0] - 1.0))

    def test_mixture(self):
        # Methane + ethane mixture
        F = FugacityRKS(
            Tc=[190.6, 305.3],
            Pc=[4.6e6, 4.872e6],
            omega=[0.011, 0.099],
            x=[0.7, 0.3],
        )
        phis = eval_block(F, 300, 101325)
        self.assertEqual(len(phis), 2)
        for phi in phis:
            self.assertGreater(phi, 0)
            self.assertTrue(np.isfinite(phi))

    def test_liquid_phase(self):
        F = FugacityRKS(Tc=190.6, Pc=4.6e6, omega=0.011, phase="liquid")
        phi = eval_block(F, 120, 5e6)
        self.assertGreater(phi[0], 0)
        self.assertTrue(np.isfinite(phi[0]))


class TestFugacityPR(unittest.TestCase):

    def test_init_pure(self):
        F = FugacityPR(Tc=190.6, Pc=4.6e6, omega=0.011)
        self.assertEqual(F.nc, 1)

    def test_ideal_gas_limit(self):
        F = FugacityPR(Tc=190.6, Pc=4.6e6, omega=0.011)
        phi = eval_block(F, 1000, 100)
        self.assertAlmostEqual(phi[0], 1.0, delta=0.01)

    def test_methane_moderate(self):
        F = FugacityPR(Tc=190.6, Pc=4.6e6, omega=0.011)
        phi = eval_block(F, 300, 101325)
        self.assertAlmostEqual(phi[0], 1.0, delta=0.05)

    def test_mixture(self):
        F = FugacityPR(
            Tc=[190.6, 305.3],
            Pc=[4.6e6, 4.872e6],
            omega=[0.011, 0.099],
            x=[0.7, 0.3],
        )
        phis = eval_block(F, 300, 101325)
        self.assertEqual(len(phis), 2)
        for phi in phis:
            self.assertGreater(phi, 0)
            self.assertTrue(np.isfinite(phi))

    def test_pr_vs_rks_similar(self):
        # At ideal gas conditions, both should give similar phi
        F_pr = FugacityPR(Tc=190.6, Pc=4.6e6, omega=0.011)
        F_rks = FugacityRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        phi_pr = eval_block(F_pr, 300, 101325)
        phi_rks = eval_block(F_rks, 300, 101325)
        self.assertAlmostEqual(phi_pr[0], phi_rks[0], delta=0.02)

    def test_liquid_phase(self):
        F = FugacityPR(Tc=190.6, Pc=4.6e6, omega=0.011, phase="liquid")
        phi = eval_block(F, 120, 5e6)
        self.assertGreater(phi[0], 0)
        self.assertTrue(np.isfinite(phi[0]))


class TestFugacityVirial(unittest.TestCase):

    def test_init_pure(self):
        F = FugacityVirial(B=[[-4.3e-4]])
        self.assertEqual(F.nc, 1)

    def test_ideal_gas_limit(self):
        # With B=0, phi should be 1
        F = FugacityVirial(B=[[0.0]], y=[1.0])
        phi = eval_block(F, 300, 101325)
        self.assertAlmostEqual(phi[0], 1.0, places=5)

    def test_negative_B(self):
        # With negative B (attractive interactions), phi < 1
        F = FugacityVirial(B=[[-5e-4]], y=[1.0])
        phi = eval_block(F, 300, 101325)
        self.assertLess(phi[0], 1.0)
        self.assertGreater(phi[0], 0)

    def test_mixture(self):
        # Binary mixture with cross virial coefficients
        F = FugacityVirial(
            B=[[-4.3e-4, -3.5e-4],
               [-3.5e-4, -6.0e-4]],
            y=[0.7, 0.3],
        )
        phis = eval_block(F, 300, 101325)
        self.assertEqual(len(phis), 2)
        for phi in phis:
            self.assertGreater(phi, 0)
            self.assertTrue(np.isfinite(phi))

    def test_low_pressure(self):
        # At very low pressure, phi -> 1 even with non-zero B
        F = FugacityVirial(B=[[-5e-4]], y=[1.0])
        phi = eval_block(F, 500, 10)  # 10 Pa
        self.assertAlmostEqual(phi[0], 1.0, delta=0.001)


# SMOKE TEST ===========================================================================

class TestSmokeAllFugacity(unittest.TestCase):

    def test_all_blocks(self):
        blocks = [
            FugacityRKS(Tc=190.6, Pc=4.6e6, omega=0.011),
            FugacityPR(Tc=190.6, Pc=4.6e6, omega=0.011),
            FugacityVirial(B=[[-4.3e-4]], y=[1.0]),
        ]

        for block in blocks:
            with self.subTest(block=block.__class__.__name__):
                phis = eval_block(block, 300, 101325)
                for phi in phis:
                    self.assertTrue(np.isfinite(phi))
                    self.assertGreater(phi, 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
