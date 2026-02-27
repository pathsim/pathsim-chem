########################################################################################
##
##                                  TESTS FOR
##                'thermodynamics.activity_coefficients.py'
##
##    IK-CAPE Activity Coefficient Models (Chapter 4)
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.thermodynamics import (
    NRTL,
    Wilson,
    UNIQUAC,
)


# HELPERS ==============================================================================

def eval_block(block, T):
    """Set input T, update, and return all outputs."""
    block.inputs[0] = T
    block.update(None)
    n_out = len([k for k in block.outputs])
    if n_out == 1:
        return block.outputs[0]
    return tuple(block.outputs[i] for i in range(n_out))


# TESTS ================================================================================

class TestNRTL(unittest.TestCase):

    def test_init(self):
        N = NRTL(
            x=[0.5, 0.5],
            a=[[0, 1.5], [2.0, 0]],
        )
        self.assertEqual(N.n, 2)
        np.testing.assert_array_equal(N.x, [0.5, 0.5])

    def test_pure_component_gamma_is_one(self):
        # For x_1=1.0, gamma_1 should be 1.0 regardless of parameters
        N = NRTL(
            x=[1.0, 0.0],
            a=[[0, 1.5], [2.0, 0]],
            c=[[0, 0.3], [0.3, 0]],
        )
        gamma1, gamma2 = eval_block(N, 350)
        self.assertAlmostEqual(gamma1, 1.0, places=10)

    def test_symmetric_mixture(self):
        # Symmetric parameters and equal x => gamma_1 = gamma_2
        N = NRTL(
            x=[0.5, 0.5],
            a=[[0, 1.0], [1.0, 0]],
            c=[[0, 0.3], [0.3, 0]],
        )
        gamma1, gamma2 = eval_block(N, 350)
        self.assertAlmostEqual(gamma1, gamma2, places=10)
        self.assertGreater(gamma1, 1.0)  # positive deviation

    def test_ethanol_water(self):
        # Ethanol(1)-Water(2) NRTL parameters (simplified, constant tau and alpha)
        # tau_12 = -0.801, tau_21 = 3.458, alpha = 0.3
        N = NRTL(
            x=[0.4, 0.6],
            a=[[0, -0.801], [3.458, 0]],
            c=[[0, 0.3], [0.3, 0]],
        )
        gamma1, gamma2 = eval_block(N, 350)
        # Both should be > 1 for this system
        self.assertGreater(gamma1, 1.0)
        self.assertGreater(gamma2, 1.0)

    def test_temperature_dependence(self):
        # With b parameter, gamma should change with T
        N = NRTL(
            x=[0.5, 0.5],
            a=[[0, 0.5], [0.5, 0]],
            b=[[0, 100], [100, 0]],
            c=[[0, 0.3], [0.3, 0]],
        )
        g300 = eval_block(N, 300)
        g400 = eval_block(N, 400)
        # gamma should change with temperature
        self.assertNotAlmostEqual(g300[0], g400[0], places=3)

    def test_three_components(self):
        N = NRTL(
            x=[0.3, 0.3, 0.4],
            a=[[0, 0.5, 0.8],
               [0.6, 0, 0.4],
               [0.9, 0.3, 0]],
            c=[[0, 0.3, 0.3],
               [0.3, 0, 0.3],
               [0.3, 0.3, 0]],
        )
        gammas = eval_block(N, 350)
        self.assertEqual(len(gammas), 3)
        for g in gammas:
            self.assertGreater(g, 0)
            self.assertTrue(np.isfinite(g))


class TestWilson(unittest.TestCase):

    def test_init(self):
        W = Wilson(
            x=[0.5, 0.5],
            a=[[0, 0.5], [-0.3, 0]],
        )
        self.assertEqual(W.n, 2)

    def test_pure_component_gamma_is_one(self):
        W = Wilson(
            x=[1.0, 0.0],
            a=[[0, 0.5], [-0.3, 0]],
        )
        gamma1, gamma2 = eval_block(W, 350)
        self.assertAlmostEqual(gamma1, 1.0, places=10)

    def test_symmetric_mixture(self):
        W = Wilson(
            x=[0.5, 0.5],
            a=[[0, 0.5], [0.5, 0]],
        )
        gamma1, gamma2 = eval_block(W, 350)
        self.assertAlmostEqual(gamma1, gamma2, places=10)

    def test_identity_lambdas(self):
        # When a=0 (all Lambdas = 1), ideal solution => gamma = 1
        W = Wilson(
            x=[0.5, 0.5],
            a=[[0, 0], [0, 0]],
        )
        gamma1, gamma2 = eval_block(W, 350)
        self.assertAlmostEqual(gamma1, 1.0, places=10)
        self.assertAlmostEqual(gamma2, 1.0, places=10)

    def test_temperature_dependence(self):
        W = Wilson(
            x=[0.5, 0.5],
            a=[[0, 0.2], [-0.3, 0]],
            b=[[0, 200], [-150, 0]],
        )
        g300 = eval_block(W, 300)
        g400 = eval_block(W, 400)
        self.assertNotAlmostEqual(g300[0], g400[0], places=3)


class TestUNIQUAC(unittest.TestCase):

    def test_init(self):
        U = UNIQUAC(
            x=[0.5, 0.5],
            r=[2.1055, 0.92],
            q=[1.972, 1.4],
            a=[[0, -1.318], [2.772, 0]],
        )
        self.assertEqual(U.n, 2)

    def test_pure_component_gamma_is_one(self):
        U = UNIQUAC(
            x=[1.0, 0.0],
            r=[2.1055, 0.92],
            q=[1.972, 1.4],
            a=[[0, -1.318], [2.772, 0]],
        )
        gamma1, gamma2 = eval_block(U, 350)
        self.assertAlmostEqual(gamma1, 1.0, places=10)

    def test_symmetric_case(self):
        # Identical r, q, symmetric a => gamma_1 = gamma_2
        U = UNIQUAC(
            x=[0.5, 0.5],
            r=[1.0, 1.0],
            q=[1.0, 1.0],
            a=[[0, -0.5], [-0.5, 0]],
        )
        gamma1, gamma2 = eval_block(U, 350)
        self.assertAlmostEqual(gamma1, gamma2, places=10)

    def test_ethanol_water(self):
        # Ethanol(1)-Water(2) UNIQUAC
        # r: ethanol=2.1055, water=0.92
        # q: ethanol=1.972, water=1.4
        U = UNIQUAC(
            x=[0.4, 0.6],
            r=[2.1055, 0.92],
            q=[1.972, 1.4],
            a=[[0, -1.318], [2.772, 0]],
        )
        gamma1, gamma2 = eval_block(U, 350)
        self.assertGreater(gamma1, 0)
        self.assertGreater(gamma2, 0)
        self.assertTrue(np.isfinite(gamma1))
        self.assertTrue(np.isfinite(gamma2))

    def test_three_components(self):
        U = UNIQUAC(
            x=[0.3, 0.3, 0.4],
            r=[2.1, 0.92, 1.5],
            q=[1.97, 1.4, 1.2],
            a=[[0, -1.0, 0.5],
               [2.0, 0, -0.3],
               [0.8, 0.4, 0]],
        )
        gammas = eval_block(U, 350)
        self.assertEqual(len(gammas), 3)
        for g in gammas:
            self.assertGreater(g, 0)
            self.assertTrue(np.isfinite(g))


# SMOKE TEST ===========================================================================

class TestSmokeAllActivityCoefficients(unittest.TestCase):

    def test_all_at_350K(self):
        blocks = [
            NRTL(x=[0.5, 0.5], a=[[0, 1.0], [1.5, 0]], c=[[0, 0.3], [0.3, 0]]),
            Wilson(x=[0.5, 0.5], a=[[0, 0.5], [-0.3, 0]]),
            UNIQUAC(x=[0.5, 0.5], r=[2.1, 0.92], q=[1.97, 1.4], a=[[0, -1.0], [2.0, 0]]),
        ]

        for block in blocks:
            with self.subTest(block=block.__class__.__name__):
                result = eval_block(block, 350)
                for g in result:
                    self.assertTrue(np.isfinite(g), f"{block.__class__.__name__} non-finite")
                    self.assertGreater(g, 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
