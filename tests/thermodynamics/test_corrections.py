########################################################################################
##
##                                  TESTS FOR
##                     'thermodynamics.corrections.py'
##
##    IK-CAPE Poynting Correction (Chapter 5) and Henry Constant (Chapter 6)
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.thermodynamics import (
    PoyntingCorrection,
    HenryConstant,
)


# CONSTANTS ============================================================================

R = 8.314462


# HELPERS ==============================================================================

def eval_block(block, *values):
    """Set inputs and return output."""
    for i, v in enumerate(values):
        block.inputs[i] = v
    block.update(None)
    return block.outputs[0]


# TESTS ================================================================================

class TestPoyntingCorrection(unittest.TestCase):

    def test_no_pressure_difference(self):
        # When P == Psat, correction should be 1.0
        PC = PoyntingCorrection()
        Fp = eval_block(PC, 373.15, 101325, 101325, 1.8e-5)
        self.assertAlmostEqual(Fp, 1.0, places=10)

    def test_high_pressure(self):
        # At P > Psat, Fp > 1
        PC = PoyntingCorrection()
        Fp = eval_block(PC, 373.15, 1e7, 101325, 1.8e-5)
        self.assertGreater(Fp, 1.0)

    def test_known_value(self):
        # Manual calculation: Fp = exp(vL*(P-Psat)/(RT))
        T = 300
        P = 1e7
        Psat = 3500
        vL = 1.8e-5  # m^3/mol, typical liquid water
        expected = np.exp(vL * (P - Psat) / (R * T))

        PC = PoyntingCorrection()
        Fp = eval_block(PC, T, P, Psat, vL)
        self.assertAlmostEqual(Fp, expected, places=10)

    def test_close_to_one_at_moderate_pressure(self):
        # For typical liquid, Poynting correction is small
        PC = PoyntingCorrection()
        Fp = eval_block(PC, 300, 200000, 100000, 1.8e-5)
        # vL*(P-Psat)/(RT) ~ 1.8e-5 * 1e5 / (8.314*300) ~ 0.0007
        self.assertAlmostEqual(Fp, 1.0, delta=0.001)


class TestHenryConstant(unittest.TestCase):

    def test_init(self):
        H = HenryConstant(a=15.0, b=-1500.0)
        self.assertEqual(H.coeffs, (15.0, -1500.0, 0.0, 0.0))

    def test_constant(self):
        # With only a, H = exp(a)
        H = HenryConstant(a=10.0)
        result = eval_block(H, 300)
        self.assertAlmostEqual(result, np.exp(10.0), places=5)

    def test_temperature_dependence(self):
        H = HenryConstant(a=15.0, b=-1500.0)
        H_300 = eval_block(H, 300)
        H_350 = eval_block(H, 350)
        # Different temperatures should give different values
        self.assertNotAlmostEqual(H_300, H_350, places=0)
        # Both should be positive
        self.assertGreater(H_300, 0)
        self.assertGreater(H_350, 0)

    def test_known_value(self):
        # ln(H) = a + b/T + c*ln(T) + d*T
        a, b, c, d = 10.0, -2000.0, 1.5, -0.001
        T = 350
        expected = np.exp(a + b / T + c * np.log(T) + d * T)

        H = HenryConstant(a=a, b=b, c=c, d=d)
        result = eval_block(H, T)
        self.assertAlmostEqual(result, expected, places=5)

    def test_o2_in_water(self):
        # O2 Henry constant in water: ln(H/Pa) ~ 21 - 1700/T
        H = HenryConstant(a=21.0, b=-1700.0)
        H_val = eval_block(H, 298.15)
        # Should be in the range of ~4e4 to ~5e4 kPa for O2 in water
        self.assertGreater(H_val, 1e6)  # positive and large


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
