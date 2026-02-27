########################################################################################
##
##                                  TESTS FOR
##                      'thermodynamics.averages.py'
##
##    IK-CAPE Calculation of Averages (Chapter 3)
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.thermodynamics import (
    MolarAverage,
    MassAverage,
    LogMolarAverage,
    LogMassAverage,
    LambdaAverage,
    ViscosityAverage,
    VolumeAverage,
    WilkeViscosity,
    WassiljewaMasonSaxena,
    DIPPRSurfaceTension,
)


# HELPERS ==============================================================================

def eval_block(block, *values):
    """Set inputs and return output."""
    for i, v in enumerate(values):
        block.inputs[i] = v
    block.update(None)
    return block.outputs[0]


# TESTS ================================================================================

class TestMolarAverage(unittest.TestCase):

    def test_init(self):
        M = MolarAverage(x=[0.5, 0.5])
        np.testing.assert_array_equal(M.x, [0.5, 0.5])

    def test_equal_fractions(self):
        M = MolarAverage(x=[0.5, 0.5])
        # avg = 0.5*10 + 0.5*20 = 15
        self.assertAlmostEqual(eval_block(M, 10, 20), 15.0)

    def test_pure_component(self):
        M = MolarAverage(x=[1.0, 0.0])
        self.assertAlmostEqual(eval_block(M, 42, 99), 42.0)

    def test_three_components(self):
        M = MolarAverage(x=[0.2, 0.3, 0.5])
        # 0.2*100 + 0.3*200 + 0.5*300 = 20 + 60 + 150 = 230
        self.assertAlmostEqual(eval_block(M, 100, 200, 300), 230.0)


class TestMassAverage(unittest.TestCase):

    def test_normalized_fractions(self):
        M = MassAverage(w=[0.4, 0.6])
        # avg = (0.4*10 + 0.6*20) / (0.4+0.6) = 16
        self.assertAlmostEqual(eval_block(M, 10, 20), 16.0)

    def test_unnormalized_fractions(self):
        # w=[2, 3] should give same result as [0.4, 0.6]
        M = MassAverage(w=[2, 3])
        self.assertAlmostEqual(eval_block(M, 10, 20), 16.0)


class TestLogMolarAverage(unittest.TestCase):

    def test_equal_values(self):
        L = LogMolarAverage(x=[0.5, 0.5])
        # ln(avg) = 0.5*ln(100) + 0.5*ln(100) => avg = 100
        self.assertAlmostEqual(eval_block(L, 100, 100), 100.0)

    def test_geometric_mean(self):
        L = LogMolarAverage(x=[0.5, 0.5])
        # geometric mean of 4 and 9 = sqrt(36) = 6
        self.assertAlmostEqual(eval_block(L, 4, 9), 6.0)

    def test_pure_component(self):
        L = LogMolarAverage(x=[1.0, 0.0])
        self.assertAlmostEqual(eval_block(L, 42, 99), 42.0)


class TestLogMassAverage(unittest.TestCase):

    def test_geometric_mean(self):
        L = LogMassAverage(w=[0.5, 0.5])
        # same as LogMolar when w sums to 1
        self.assertAlmostEqual(eval_block(L, 4, 9), 6.0)

    def test_unnormalized(self):
        L = LogMassAverage(w=[1, 1])
        self.assertAlmostEqual(eval_block(L, 4, 9), 6.0)


class TestLambdaAverage(unittest.TestCase):

    def test_equal_values(self):
        L = LambdaAverage(x=[0.5, 0.5])
        # 0.5*(sum + 1/sum_inv) with equal values = value
        self.assertAlmostEqual(eval_block(L, 10, 10), 10.0)

    def test_arithmetic_harmonic_mean(self):
        L = LambdaAverage(x=[0.5, 0.5])
        v1, v2 = 2.0, 8.0
        arith = 0.5 * (0.5 * v1 + 0.5 * v2)
        harmo = 0.5 * (1.0 / (0.5 / v1 + 0.5 / v2))
        expected = arith + harmo
        self.assertAlmostEqual(eval_block(L, v1, v2), expected)

    def test_pure_component(self):
        L = LambdaAverage(x=[1.0, 0.0, 0.0])
        self.assertAlmostEqual(eval_block(L, 5.0, 10.0, 20.0), 5.0)


class TestViscosityAverage(unittest.TestCase):

    def test_equal_mol_weights(self):
        # With equal M, reduces to molar average
        V = ViscosityAverage(x=[0.5, 0.5], M=[28, 28])
        self.assertAlmostEqual(eval_block(V, 10, 20), 15.0)

    def test_weighting(self):
        V = ViscosityAverage(x=[0.5, 0.5], M=[4, 16])
        # weights: 0.5*sqrt(4)=1, 0.5*sqrt(16)=2, sum=3
        # avg = (1*10 + 2*20) / 3 = 50/3
        self.assertAlmostEqual(eval_block(V, 10, 20), 50.0 / 3.0)


class TestVolumeAverage(unittest.TestCase):

    def test_equal_values(self):
        V = VolumeAverage(x=[0.5, 0.5])
        self.assertAlmostEqual(eval_block(V, 1000, 1000), 1000.0)

    def test_harmonic_mean(self):
        V = VolumeAverage(x=[0.5, 0.5])
        # 1 / (0.5/800 + 0.5/1200) = 1 / (0.000625 + 0.000417) = 960
        expected = 1.0 / (0.5 / 800 + 0.5 / 1200)
        self.assertAlmostEqual(eval_block(V, 800, 1200), expected, places=5)

    def test_pure_component(self):
        V = VolumeAverage(x=[1.0, 0.0])
        self.assertAlmostEqual(eval_block(V, 800, 1200), 800.0)


class TestWilkeViscosity(unittest.TestCase):

    def test_pure_component(self):
        W = WilkeViscosity(y=[1.0, 0.0], M=[28, 32])
        # Pure component 1, should return its viscosity
        self.assertAlmostEqual(eval_block(W, 1.8e-5, 2.0e-5), 1.8e-5)

    def test_equal_species(self):
        # Two identical species => average should equal their shared value
        W = WilkeViscosity(y=[0.5, 0.5], M=[28, 28])
        self.assertAlmostEqual(eval_block(W, 1.5e-5, 1.5e-5), 1.5e-5)

    def test_symmetric_F(self):
        # For equal mol weights, F_ij depends only on viscosity ratio
        W = WilkeViscosity(y=[0.5, 0.5], M=[28, 28])
        result = eval_block(W, 1.0e-5, 2.0e-5)
        # Should be between the two values
        self.assertGreater(result, 1.0e-5)
        self.assertLess(result, 2.0e-5)

    def test_n2_o2_air(self):
        # N2/O2 mixture (roughly air)
        W = WilkeViscosity(y=[0.79, 0.21], M=[28.014, 31.999])
        mu_N2 = 1.78e-5  # Pa.s at ~300K
        mu_O2 = 2.06e-5
        result = eval_block(W, mu_N2, mu_O2)
        # Air viscosity ~ 1.85e-5 Pa.s
        self.assertAlmostEqual(result, 1.85e-5, delta=0.1e-5)


class TestWassiljewaMasonSaxena(unittest.TestCase):

    def test_pure_component(self):
        W = WassiljewaMasonSaxena(y=[1.0, 0.0], M=[28, 32])
        # inputs: lambda_1, lambda_2, eta_1, eta_2
        lam1, lam2 = 0.026, 0.027
        eta1, eta2 = 1.8e-5, 2.0e-5
        result = eval_block(W, lam1, lam2, eta1, eta2)
        self.assertAlmostEqual(result, lam1)

    def test_equal_species(self):
        W = WassiljewaMasonSaxena(y=[0.5, 0.5], M=[28, 28])
        lam, eta = 0.025, 1.5e-5
        result = eval_block(W, lam, lam, eta, eta)
        self.assertAlmostEqual(result, lam)

    def test_bounded(self):
        W = WassiljewaMasonSaxena(y=[0.5, 0.5], M=[28, 32])
        lam1, lam2 = 0.020, 0.030
        eta1, eta2 = 1.5e-5, 2.0e-5
        result = eval_block(W, lam1, lam2, eta1, eta2)
        self.assertGreater(result, lam1)
        self.assertLess(result, lam2)


class TestDIPPRSurfaceTension(unittest.TestCase):

    def test_equal_values(self):
        D = DIPPRSurfaceTension(x=[0.5, 0.5], V=[0.05, 0.05])
        # When all values are equal, average = value
        self.assertAlmostEqual(eval_block(D, 0.072, 0.072), 0.072, places=6)

    def test_pure_component(self):
        D = DIPPRSurfaceTension(x=[1.0, 0.0], V=[0.05, 0.04])
        self.assertAlmostEqual(eval_block(D, 0.072, 0.030), 0.072, places=6)

    def test_bounded(self):
        D = DIPPRSurfaceTension(x=[0.5, 0.5], V=[0.05, 0.05])
        result = eval_block(D, 0.030, 0.072)
        self.assertGreater(result, 0.030)
        self.assertLess(result, 0.072)


# SMOKE TEST ===========================================================================

class TestSmokeAllAverages(unittest.TestCase):
    """Verify every averaging block can be instantiated and evaluated."""

    def test_all_blocks(self):
        blocks_and_inputs = [
            (MolarAverage(x=[0.5, 0.5]), [10, 20]),
            (MassAverage(w=[0.4, 0.6]), [10, 20]),
            (LogMolarAverage(x=[0.5, 0.5]), [10, 20]),
            (LogMassAverage(w=[0.5, 0.5]), [10, 20]),
            (LambdaAverage(x=[0.5, 0.5]), [10, 20]),
            (ViscosityAverage(x=[0.5, 0.5], M=[28, 32]), [1e-5, 2e-5]),
            (VolumeAverage(x=[0.5, 0.5]), [800, 1200]),
            (WilkeViscosity(y=[0.5, 0.5], M=[28, 32]), [1.5e-5, 2e-5]),
            (WassiljewaMasonSaxena(y=[0.5, 0.5], M=[28, 32]), [0.025, 0.026, 1.5e-5, 2e-5]),
            (DIPPRSurfaceTension(x=[0.5, 0.5], V=[0.05, 0.04]), [0.03, 0.07]),
        ]

        for block, inputs in blocks_and_inputs:
            with self.subTest(block=block.__class__.__name__):
                result = eval_block(block, *inputs)
                self.assertTrue(np.isfinite(result), f"{block.__class__.__name__} returned non-finite: {result}")


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
