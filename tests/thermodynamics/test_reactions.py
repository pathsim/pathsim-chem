########################################################################################
##
##                                  TESTS FOR
##                      'thermodynamics.reactions.py'
##
##    IK-CAPE Chemical Reactions (Chapter 10)
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.thermodynamics import (
    EquilibriumConstant,
    KineticRateConstant,
    PowerLawRate,
)


# CONSTANTS ============================================================================

R = 8.314462


# HELPERS ==============================================================================

def eval_block_T(block, T):
    """Set input T, update, return output."""
    block.inputs[0] = T
    block.update(None)
    return block.outputs[0]


def eval_block_multi(block, *values):
    """Set multiple inputs, update, return all outputs."""
    for i, v in enumerate(values):
        block.inputs[i] = v
    block.update(None)
    n_out = len([k for k in block.outputs])
    if n_out == 1:
        return (block.outputs[0],)
    return tuple(block.outputs[i] for i in range(n_out))


# TESTS ================================================================================

class TestEquilibriumConstant(unittest.TestCase):

    def test_init(self):
        K = EquilibriumConstant(a0=10.0, a1=-5000.0)
        self.assertEqual(K.coeffs, (10.0, -5000.0, 0.0, 0.0, 0.0, 0.0))

    def test_constant_only(self):
        # K = exp(a0)
        K = EquilibriumConstant(a0=5.0)
        result = eval_block_T(K, 300)
        self.assertAlmostEqual(result, np.exp(5.0), places=5)

    def test_vant_hoff(self):
        # ln K = a0 + a1/T => K = exp(a0 + a1/T)
        a0, a1 = 20.0, -5000.0
        K = EquilibriumConstant(a0=a0, a1=a1)
        T = 400
        expected = np.exp(a0 + a1 / T)
        result = eval_block_T(K, T)
        self.assertAlmostEqual(result, expected, places=5)

    def test_temperature_dependence(self):
        K = EquilibriumConstant(a0=10.0, a1=-5000.0)
        K_300 = eval_block_T(K, 300)
        K_500 = eval_block_T(K, 500)
        # With negative a1, ln K increases with T (less negative 1/T term)
        self.assertGreater(K_500, K_300)

    def test_all_parameters(self):
        a0, a1, a2, a3, a4, a5 = 10.0, -2000.0, 1.5, -0.001, 1e-6, -1e-9
        T = 350
        K = EquilibriumConstant(a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5)
        expected = np.exp(a0 + a1/T + a2*np.log(T) + a3*T + a4*T**2 + a5*T**3)
        result = eval_block_T(K, T)
        self.assertAlmostEqual(result, expected, places=5)

    def test_positive(self):
        K = EquilibriumConstant(a0=10.0, a1=-5000.0)
        result = eval_block_T(K, 350)
        self.assertGreater(result, 0)


class TestKineticRateConstant(unittest.TestCase):

    def test_init(self):
        k = KineticRateConstant(a0=30.0, a1=-10000.0)
        self.assertEqual(k.coeffs, (30.0, -10000.0, 0.0, 0.0))

    def test_arrhenius(self):
        # k = A * exp(-Ea/RT) => ln k = ln(A) - Ea/(RT) = a0 + a1/T
        # where a0 = ln(A), a1 = -Ea/R
        a0 = np.log(1e13)   # pre-exponential factor A = 1e13
        a1 = -15000.0        # Ea/R = 15000 K
        k = KineticRateConstant(a0=a0, a1=a1)
        T = 500
        expected = np.exp(a0 + a1 / T)
        result = eval_block_T(k, T)
        self.assertAlmostEqual(result, expected, places=5)

    def test_temperature_dependence(self):
        k = KineticRateConstant(a0=30.0, a1=-10000.0)
        k_300 = eval_block_T(k, 300)
        k_500 = eval_block_T(k, 500)
        # Rate constant increases with temperature
        self.assertGreater(k_500, k_300)

    def test_positive(self):
        k = KineticRateConstant(a0=30.0, a1=-10000.0)
        result = eval_block_T(k, 350)
        self.assertGreater(result, 0)

    def test_all_parameters(self):
        a0, a1, a2, a3 = 25.0, -8000.0, 0.5, -0.001
        T = 400
        k = KineticRateConstant(a0=a0, a1=a1, a2=a2, a3=a3)
        expected = np.exp(a0 + a1/T + a2*np.log(T) + a3*T)
        result = eval_block_T(k, T)
        self.assertAlmostEqual(result, expected, places=5)


class TestPowerLawRate(unittest.TestCase):

    def test_simple_irreversible(self):
        # A -> B, rate = k * c_A
        # nu = [[-1], [1]], forward order: |nu_A| = 1
        rate = PowerLawRate(nu=[[-1], [1]])

        # Inputs: f_0=k=2.0, c_0=c_A=0.5, c_1=c_B=0.3
        R_vals = eval_block_multi(rate, 2.0, 0.5, 0.3)
        # r = k * c_A = 2.0 * 0.5 = 1.0
        # R_A = -1 * 1.0 = -1.0, R_B = 1 * 1.0 = 1.0
        self.assertAlmostEqual(R_vals[0], -1.0, places=10)
        self.assertAlmostEqual(R_vals[1], 1.0, places=10)

    def test_bimolecular(self):
        # A + B -> C, rate = k * c_A * c_B
        rate = PowerLawRate(nu=[[-1], [-1], [1]])

        # k=3.0, c_A=0.4, c_B=0.5, c_C=0.1
        R_vals = eval_block_multi(rate, 3.0, 0.4, 0.5, 0.1)
        # r = 3.0 * 0.4 * 0.5 = 0.6
        self.assertAlmostEqual(R_vals[0], -0.6, places=10)
        self.assertAlmostEqual(R_vals[1], -0.6, places=10)
        self.assertAlmostEqual(R_vals[2], 0.6, places=10)

    def test_equilibrium_limited(self):
        # A <-> B, Keq = 10.0
        rate = PowerLawRate(
            nu=[[-1], [1]],
            Keq=[10.0],
        )

        # At c_A=1.0, c_B=0.0: driving force = 1 - (c_B/c_A) / Keq = 1
        R_vals = eval_block_multi(rate, 2.0, 1.0, 0.0)
        # r = k * c_A * (1 - c_B/(c_A*Keq)) = 2.0 * 1.0 * (1 - 0) = 2.0
        self.assertAlmostEqual(R_vals[0], -2.0, places=10)

    def test_equilibrium_at_equilibrium(self):
        # At equilibrium, net rate should be zero
        # A <-> B, Keq = 4.0, so at equilibrium c_B/c_A = 4.0
        rate = PowerLawRate(
            nu=[[-1], [1]],
            Keq=[4.0],
        )
        # c_A=0.2, c_B=0.8 => c_B/c_A = 4.0 = Keq
        R_vals = eval_block_multi(rate, 1.0, 0.2, 0.8)
        self.assertAlmostEqual(R_vals[0], 0.0, places=10)
        self.assertAlmostEqual(R_vals[1], 0.0, places=10)

    def test_two_reactions(self):
        # A -> B (rxn 1), A -> C (rxn 2)
        # nu = [[-1, -1], [1, 0], [0, 1]]
        rate = PowerLawRate(nu=[[-1, -1], [1, 0], [0, 1]])

        # k1=1.0, k2=2.0, c_A=0.5, c_B=0.2, c_C=0.3
        R_vals = eval_block_multi(rate, 1.0, 2.0, 0.5, 0.2, 0.3)
        # r1 = 1.0 * 0.5 = 0.5, r2 = 2.0 * 0.5 = 1.0
        # R_A = -0.5 - 1.0 = -1.5, R_B = 0.5, R_C = 1.0
        self.assertAlmostEqual(R_vals[0], -1.5, places=10)
        self.assertAlmostEqual(R_vals[1], 0.5, places=10)
        self.assertAlmostEqual(R_vals[2], 1.0, places=10)

    def test_custom_orders(self):
        # A -> B, but with fractional order 0.5 for A
        rate = PowerLawRate(
            nu=[[-1], [1]],
            nu_fwd=[[0.5], [0.0]],
        )
        # k=4.0, c_A=0.25, c_B=0.5
        R_vals = eval_block_multi(rate, 4.0, 0.25, 0.5)
        # r = 4.0 * 0.25^0.5 = 4.0 * 0.5 = 2.0
        self.assertAlmostEqual(R_vals[0], -2.0, places=10)
        self.assertAlmostEqual(R_vals[1], 2.0, places=10)


# SMOKE TEST ===========================================================================

class TestSmokeAllReactions(unittest.TestCase):

    def test_all_correlation_blocks(self):
        blocks_T = [
            EquilibriumConstant(a0=10.0, a1=-5000.0),
            KineticRateConstant(a0=30.0, a1=-10000.0),
        ]

        for block in blocks_T:
            with self.subTest(block=block.__class__.__name__):
                result = eval_block_T(block, 350)
                self.assertTrue(np.isfinite(result))
                self.assertGreater(result, 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
