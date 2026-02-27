########################################################################################
##
##                                  TESTS FOR
##                   'thermodynamics.correlations.py'
##
##    IK-CAPE Pure Component Property Correlations
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.thermodynamics import (
    Polynomial,
    ExponentialPolynomial,
    Watson,
    Antoine,
    ExtendedAntoine,
    Kirchhoff,
    ExtendedKirchhoff,
    Sutherland,
    Wagner,
    LiquidHeatCapacity,
    ExtendedLiquidHeatCapacity,
    DynamicViscosity,
    Rackett,
    AlyLee,
    DIPPR4,
    DIPPR5,
)


# HELPERS ==============================================================================

def eval_block(block, T):
    """Set input T, update, and return output value."""
    block.inputs[0] = T
    block.update(None)
    return block.outputs[0]


# TESTS ================================================================================

class TestPolynomial(unittest.TestCase):

    def test_init(self):
        P = Polynomial(coeffs=[1.0, 2.0, 3.0])
        self.assertEqual(P.coeffs, (1.0, 2.0, 3.0))

    def test_constant(self):
        P = Polynomial(coeffs=[42.0])
        self.assertAlmostEqual(eval_block(P, 300), 42.0)

    def test_linear(self):
        P = Polynomial(coeffs=[10.0, 0.5])
        # f(300) = 10 + 0.5*300 = 160
        self.assertAlmostEqual(eval_block(P, 300), 160.0)

    def test_quadratic(self):
        P = Polynomial(coeffs=[1.0, 0.0, 0.01])
        # f(10) = 1 + 0 + 0.01*100 = 2
        self.assertAlmostEqual(eval_block(P, 10), 2.0)


class TestExponentialPolynomial(unittest.TestCase):

    def test_init(self):
        EP = ExponentialPolynomial(coeffs=[2.0, 0.001])
        self.assertEqual(EP.coeffs, (2.0, 0.001))

    def test_constant_exponent(self):
        EP = ExponentialPolynomial(coeffs=[3.0])
        # f(T) = 10^3 = 1000
        self.assertAlmostEqual(eval_block(EP, 300), 1000.0)

    def test_evaluation(self):
        EP = ExponentialPolynomial(coeffs=[2.0, 0.001])
        T = 300
        expected = 10**(2.0 + 0.001 * 300)
        self.assertAlmostEqual(eval_block(EP, T), expected, places=5)


class TestWatson(unittest.TestCase):

    def test_init(self):
        W = Watson(a0=5.0e7, a1=0.38, a2=647.096)
        self.assertEqual(W.coeffs, (5.0e7, 0.38, 647.096, 0.0))

    def test_at_low_temp(self):
        # Water-like: Hvap = a0 * (Tc - T)^a1
        W = Watson(a0=5.2053e7, a1=0.3199, a2=647.096)
        result = eval_block(W, 373.15)
        # Should be a positive value (heat of vaporization)
        self.assertGreater(result, 0)

    def test_with_offset(self):
        W = Watson(a0=1.0, a1=1.0, a2=500.0, a3=100.0)
        # f(300) = 1.0*(500-300)^1 + 100 = 300
        self.assertAlmostEqual(eval_block(W, 300), 300.0)


class TestAntoine(unittest.TestCase):

    def test_init(self):
        A = Antoine(a0=23.2256, a1=3835.18, a2=-45.343)
        self.assertEqual(A.coeffs, (23.2256, 3835.18, -45.343))

    def test_water_boiling_point(self):
        # Water Antoine coefficients (ln, Pa, K)
        # At 373.15 K, vapor pressure should be ~101325 Pa
        A = Antoine(a0=23.2256, a1=3835.18, a2=-45.343)
        P = eval_block(A, 373.15)
        self.assertAlmostEqual(P, 101325, delta=2000)

    def test_water_at_room_temp(self):
        A = Antoine(a0=23.2256, a1=3835.18, a2=-45.343)
        P = eval_block(A, 298.15)
        # Water vapor pressure at 25C ~ 3170 Pa
        self.assertAlmostEqual(P, 3170, delta=200)


class TestExtendedAntoine(unittest.TestCase):

    def test_init_defaults(self):
        EA = ExtendedAntoine(a0=10.0, a1=-3000.0, a2=-50.0)
        self.assertEqual(EA.coeffs, (10.0, -3000.0, -50.0, 0.0, 0.0, 0.0, 0.0))

    def test_reduces_to_antoine(self):
        # When a3=a4=a5=0, should match Antoine (with sign convention a1 -> -a1)
        A = Antoine(a0=23.2256, a1=3835.18, a2=-45.343)
        EA = ExtendedAntoine(a0=23.2256, a1=-3835.18, a2=-45.343)
        T = 373.15
        self.assertAlmostEqual(eval_block(A, T), eval_block(EA, T), places=5)


class TestKirchhoff(unittest.TestCase):

    def test_init(self):
        K = Kirchhoff(a0=73.649, a1=7258.2, a2=7.3037)
        self.assertEqual(K.coeffs, (73.649, 7258.2, 7.3037))

    def test_evaluation(self):
        # Verify numerical correctness: ln f(T) = a0 - a1/T - a2*ln(T)
        K = Kirchhoff(a0=73.649, a1=7258.2, a2=7.3037)
        T = 373.15
        expected = np.exp(73.649 - 7258.2 / T - 7.3037 * np.log(T))
        self.assertAlmostEqual(eval_block(K, T), expected, places=5)


class TestExtendedKirchhoff(unittest.TestCase):

    def test_init_defaults(self):
        EK = ExtendedKirchhoff(a0=73.0, a1=-7000.0, a2=-7.0)
        self.assertEqual(EK.coeffs, (73.0, -7000.0, -7.0, 0.0, 0.0))

    def test_reduces_to_kirchhoff(self):
        # When a3=0, should match Kirchhoff (with sign convention)
        K = Kirchhoff(a0=73.649, a1=7258.2, a2=7.3037)
        EK = ExtendedKirchhoff(a0=73.649, a1=-7258.2, a2=-7.3037)
        T = 373.15
        self.assertAlmostEqual(eval_block(K, T), eval_block(EK, T), places=5)


class TestSutherland(unittest.TestCase):

    def test_init(self):
        S = Sutherland(a0=1.458e-6, a1=110.4)
        self.assertEqual(S.coeffs, (1.458e-6, 110.4))

    def test_air_viscosity(self):
        # Sutherland formula for air viscosity
        # mu = a0*sqrt(T) / (1 + a1/T)
        # At 300K, air viscosity ~ 1.85e-5 Pa.s
        S = Sutherland(a0=1.458e-6, a1=110.4)
        mu = eval_block(S, 300)
        self.assertAlmostEqual(mu, 1.85e-5, delta=0.1e-5)

    def test_increases_with_temperature(self):
        S = Sutherland(a0=1.458e-6, a1=110.4)
        mu_300 = eval_block(S, 300)
        mu_500 = eval_block(S, 500)
        self.assertGreater(mu_500, mu_300)


class TestWagner(unittest.TestCase):

    def test_init(self):
        W = Wagner(Tc=647.096, Pc=22064000, a2=-7.76451, a3=1.45838, a4=-2.77580, a5=-1.23303)
        self.assertEqual(W.coeffs, (647.096, 22064000, -7.76451, 1.45838, -2.77580, -1.23303))

    def test_water_at_critical(self):
        # At T = Tc, tau = 0, so f(Tc) = Pc
        W = Wagner(Tc=647.096, Pc=22064000, a2=-7.76451, a3=1.45838, a4=-2.77580, a5=-1.23303)
        P = eval_block(W, 647.096)
        self.assertAlmostEqual(P, 22064000, delta=1)

    def test_water_boiling_point(self):
        # Wagner equation for water at 373.15K
        W = Wagner(Tc=647.096, Pc=22064000, a2=-7.76451, a3=1.45838, a4=-2.77580, a5=-1.23303)
        P = eval_block(W, 373.15)
        self.assertAlmostEqual(P, 101325, delta=2000)


class TestLiquidHeatCapacity(unittest.TestCase):

    def test_init_defaults(self):
        C = LiquidHeatCapacity(a0=75.0)
        self.assertEqual(C.coeffs, (75.0, 0.0, 0.0, 0.0, 0.0))

    def test_constant_cp(self):
        C = LiquidHeatCapacity(a0=75.4)
        # f(T) = 75.4 when all other coeffs are zero
        self.assertAlmostEqual(eval_block(C, 300), 75.4)

    def test_water_cp(self):
        # Liquid water Cp ~ 75.3 J/(mol*K) at 298K, roughly constant
        C = LiquidHeatCapacity(a0=75.3)
        self.assertAlmostEqual(eval_block(C, 298.15), 75.3, places=1)


class TestExtendedLiquidHeatCapacity(unittest.TestCase):

    def test_init_defaults(self):
        C = ExtendedLiquidHeatCapacity(a0=75.0)
        self.assertEqual(C.coeffs, (75.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_evaluation(self):
        C = ExtendedLiquidHeatCapacity(a0=10.0, a1=0.1, a5=1000.0)
        T = 200
        expected = 10.0 + 0.1 * 200 + 1000.0 / 200
        self.assertAlmostEqual(eval_block(C, T), expected)


class TestDynamicViscosity(unittest.TestCase):

    def test_init_default(self):
        V = DynamicViscosity(a0=0.001, a1=1000)
        self.assertEqual(V.coeffs, (0.001, 1000, 0.0))

    def test_evaluation(self):
        V = DynamicViscosity(a0=0.001, a1=1000)
        T = 300
        expected = 0.001 * np.exp(1000 / 300)
        self.assertAlmostEqual(eval_block(V, T), expected, places=8)

    def test_decreases_with_temperature(self):
        # Liquid viscosity decreases with temperature
        V = DynamicViscosity(a0=0.001, a1=1000)
        mu_300 = eval_block(V, 300)
        mu_400 = eval_block(V, 400)
        self.assertGreater(mu_300, mu_400)


class TestRackett(unittest.TestCase):

    def test_init(self):
        R = Rackett(a0=0.01805, a1=0.2882, a2=647.096, a3=0.2857)
        self.assertEqual(R.coeffs, (0.01805, 0.2882, 647.096, 0.2857))

    def test_density_positive(self):
        R = Rackett(a0=0.01805, a1=0.2882, a2=647.096, a3=0.2857)
        result = eval_block(R, 373.15)
        self.assertGreater(result, 0)

    def test_density_decreases_with_temp(self):
        # Liquid density decreases as T increases toward Tc
        R = Rackett(a0=0.01805, a1=0.2882, a2=647.096, a3=0.2857)
        rho_300 = eval_block(R, 300)
        rho_500 = eval_block(R, 500)
        self.assertGreater(rho_300, rho_500)


class TestAlyLee(unittest.TestCase):

    def test_init(self):
        AL = AlyLee(a0=33363, a1=26790, a2=2610.5, a3=8896, a4=1169)
        self.assertEqual(AL.coeffs, (33363, 26790, 2610.5, 8896, 1169))

    def test_water_ideal_gas_cp(self):
        # DIPPR Aly-Lee coefficients for water ideal gas Cp (J/kmol/K)
        AL = AlyLee(a0=33363, a1=26790, a2=2610.5, a3=8896, a4=1169)
        Cp = eval_block(AL, 300)
        # Water ideal gas Cp at 300K ~ 33580 J/kmol/K
        self.assertAlmostEqual(Cp, 33580, delta=500)

    def test_increases_with_temperature(self):
        AL = AlyLee(a0=33363, a1=26790, a2=2610.5, a3=8896, a4=1169)
        Cp_300 = eval_block(AL, 300)
        Cp_500 = eval_block(AL, 500)
        self.assertGreater(Cp_500, Cp_300)


class TestDIPPR4(unittest.TestCase):

    def test_init_defaults(self):
        D = DIPPR4(Tc=647.096, a1=5.2053e7, a2=0.3199)
        self.assertEqual(D.coeffs, (647.096, 5.2053e7, 0.3199, 0.0, 0.0, 0.0))

    def test_zero_at_tc(self):
        # At T = Tc, (1-Tr)=0, so f(Tc)=0
        D = DIPPR4(Tc=647.096, a1=5.2053e7, a2=0.3199)
        self.assertAlmostEqual(eval_block(D, 647.096), 0.0)

    def test_water_hvap(self):
        # Water heat of vaporization via DIPPR-4
        # At 373.15K, Hvap ~ 2.257e6 J/kg = ~40.65 kJ/mol
        D = DIPPR4(Tc=647.096, a1=5.2053e7, a2=0.3199)
        Hvap = eval_block(D, 373.15)
        # Should be roughly 40.65e6 mJ/kmol range - ~4e7 J/kmol
        self.assertGreater(Hvap, 0)
        self.assertAlmostEqual(Hvap, 4.07e7, delta=0.5e7)


class TestDIPPR5(unittest.TestCase):

    def test_init_defaults(self):
        D = DIPPR5(a0=1.0e-6, a1=0.5)
        self.assertEqual(D.coeffs, (1.0e-6, 0.5, 0.0, 0.0))

    def test_evaluation(self):
        D = DIPPR5(a0=2.0, a1=1.5, a2=100.0, a3=5000.0)
        T = 400
        expected = 2.0 * 400**1.5 / (1 + 100 / 400 + 5000 / 400**2)
        self.assertAlmostEqual(eval_block(D, T), expected, places=5)

    def test_simple_power_law(self):
        # When a2=a3=0, reduces to a0*T^a1
        D = DIPPR5(a0=3.0, a1=2.0)
        T = 10
        self.assertAlmostEqual(eval_block(D, T), 300.0)


# SMOKE TEST ===========================================================================

class TestSmokeAllBlocks(unittest.TestCase):
    """Verify every block can be instantiated and evaluated at T=300K."""

    def test_all_blocks_at_300K(self):
        blocks = [
            Polynomial(coeffs=[1.0, 0.01]),
            ExponentialPolynomial(coeffs=[2.0, 0.001]),
            Watson(a0=5e7, a1=0.38, a2=647.0),
            Antoine(a0=23.2256, a1=3835.18, a2=-45.343),
            ExtendedAntoine(a0=23.2256, a1=-3835.18, a2=-45.343),
            Kirchhoff(a0=73.649, a1=7258.2, a2=7.3037),
            ExtendedKirchhoff(a0=73.649, a1=-7258.2, a2=-7.3037),
            Sutherland(a0=1.458e-6, a1=110.4),
            Wagner(Tc=647.096, Pc=22064000, a2=-7.76, a3=1.46, a4=-2.78, a5=-1.23),
            LiquidHeatCapacity(a0=75.3),
            ExtendedLiquidHeatCapacity(a0=75.3),
            DynamicViscosity(a0=0.001, a1=1000),
            Rackett(a0=0.01805, a1=0.2882, a2=647.096, a3=0.2857),
            AlyLee(a0=33363, a1=26790, a2=2610.5, a3=8896, a4=1169),
            DIPPR4(Tc=647.096, a1=5.2053e7, a2=0.3199),
            DIPPR5(a0=1e-6, a1=0.5),
        ]

        for block in blocks:
            with self.subTest(block=block.__class__.__name__):
                result = eval_block(block, 300)
                self.assertTrue(np.isfinite(result), f"{block.__class__.__name__} returned non-finite: {result}")


# PORT LABELS ==========================================================================

class TestPortLabels(unittest.TestCase):
    """Verify all blocks have correct port labels."""

    def test_input_port_label(self):
        block = Antoine(a0=23.0, a1=3800.0, a2=-45.0)
        self.assertEqual(block.input_port_labels, {"T": 0})

    def test_output_port_label(self):
        block = Antoine(a0=23.0, a1=3800.0, a2=-45.0)
        self.assertEqual(block.output_port_labels, {"value": 0})


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
