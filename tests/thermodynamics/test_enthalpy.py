########################################################################################
##
##                                  TESTS FOR
##                       'thermodynamics.enthalpy.py'
##
##    IK-CAPE Enthalpy (Chapter 9)
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim_chem.thermodynamics import (
    ExcessEnthalpyNRTL,
    ExcessEnthalpyUNIQUAC,
    ExcessEnthalpyWilson,
    ExcessEnthalpyRedlichKister,
    EnthalpyDepartureRKS,
    EnthalpyDeparturePR,
)


# CONSTANTS ============================================================================

R = 8.314462


# HELPERS ==============================================================================

def eval_block_T(block, T):
    """Set input T, update, return output."""
    block.inputs[0] = T
    block.update(None)
    return block.outputs[0]


def eval_block_TP(block, T, P):
    """Set inputs T and P, update, return output."""
    block.inputs[0] = T
    block.inputs[1] = P
    block.update(None)
    return block.outputs[0]


# TESTS: EXCESS ENTHALPY ===============================================================

class TestExcessEnthalpyNRTL(unittest.TestCase):

    def test_init(self):
        hE = ExcessEnthalpyNRTL(
            x=[0.5, 0.5],
            a=[[0, 1.5], [2.0, 0]],
        )
        self.assertEqual(hE.n, 2)

    def test_zero_for_pure_component(self):
        # Pure component should have hE = 0
        hE = ExcessEnthalpyNRTL(
            x=[1.0, 0.0],
            a=[[0, 1.5], [2.0, 0]],
            b=[[0, 100], [-50, 0]],
            c=[[0, 0.3], [0.3, 0]],
        )
        result = eval_block_T(hE, 350)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_finite_for_mixture(self):
        hE = ExcessEnthalpyNRTL(
            x=[0.4, 0.6],
            a=[[0, -0.801], [3.458, 0]],
            b=[[0, 300], [-200, 0]],
            c=[[0, 0.3], [0.3, 0]],
        )
        result = eval_block_T(hE, 350)
        self.assertTrue(np.isfinite(result))

    def test_temperature_dependence(self):
        hE = ExcessEnthalpyNRTL(
            x=[0.5, 0.5],
            a=[[0, 1.0], [1.0, 0]],
            b=[[0, 500], [500, 0]],
            c=[[0, 0.3], [0.3, 0]],
        )
        h300 = eval_block_T(hE, 300)
        h400 = eval_block_T(hE, 400)
        self.assertNotAlmostEqual(h300, h400, places=0)

    def test_zero_when_no_temperature_terms(self):
        # With only constant tau (b=0, e=0, f=0), tau'=0
        # and if d=0 (S'=0), then G'=0, so hE should be 0
        hE = ExcessEnthalpyNRTL(
            x=[0.5, 0.5],
            a=[[0, 1.0], [1.5, 0]],
            c=[[0, 0.3], [0.3, 0]],
        )
        result = eval_block_T(hE, 350)
        self.assertAlmostEqual(result, 0.0, places=8)


class TestExcessEnthalpyUNIQUAC(unittest.TestCase):

    def test_init(self):
        hE = ExcessEnthalpyUNIQUAC(
            x=[0.5, 0.5],
            r=[2.1055, 0.92],
            q=[1.972, 1.4],
            a=[[0, -1.318], [2.772, 0]],
        )
        self.assertEqual(hE.n, 2)

    def test_finite_for_mixture(self):
        hE = ExcessEnthalpyUNIQUAC(
            x=[0.4, 0.6],
            r=[2.1055, 0.92],
            q=[1.972, 1.4],
            a=[[0, -1.318], [2.772, 0]],
            b=[[0, 200], [-100, 0]],
        )
        result = eval_block_T(hE, 350)
        self.assertTrue(np.isfinite(result))

    def test_zero_when_no_temperature_terms(self):
        # With only constant tau exponent (b=0, c=0, d=0), tau'=0, hE=0
        hE = ExcessEnthalpyUNIQUAC(
            x=[0.5, 0.5],
            r=[2.1, 0.92],
            q=[1.97, 1.4],
            a=[[0, -1.0], [2.0, 0]],
        )
        result = eval_block_T(hE, 350)
        self.assertAlmostEqual(result, 0.0, places=8)

    def test_temperature_dependence(self):
        hE = ExcessEnthalpyUNIQUAC(
            x=[0.5, 0.5],
            r=[2.1, 0.92],
            q=[1.97, 1.4],
            a=[[0, -1.0], [2.0, 0]],
            b=[[0, 500], [-300, 0]],
        )
        h300 = eval_block_T(hE, 300)
        h400 = eval_block_T(hE, 400)
        self.assertNotAlmostEqual(h300, h400, places=0)


class TestExcessEnthalpyWilson(unittest.TestCase):

    def test_init(self):
        hE = ExcessEnthalpyWilson(
            x=[0.5, 0.5],
            a=[[0, 0.5], [-0.3, 0]],
        )
        self.assertEqual(hE.n, 2)

    def test_zero_when_no_temperature_terms(self):
        # With only constant Lambda (b=0, c=0, d=0), Lambda'=0, hE=0
        hE = ExcessEnthalpyWilson(
            x=[0.5, 0.5],
            a=[[0, 0.5], [-0.3, 0]],
        )
        result = eval_block_T(hE, 350)
        self.assertAlmostEqual(result, 0.0, places=8)

    def test_finite_for_mixture(self):
        hE = ExcessEnthalpyWilson(
            x=[0.4, 0.6],
            a=[[0, 0.5], [-0.3, 0]],
            b=[[0, 200], [-150, 0]],
        )
        result = eval_block_T(hE, 350)
        self.assertTrue(np.isfinite(result))

    def test_temperature_dependence(self):
        hE = ExcessEnthalpyWilson(
            x=[0.5, 0.5],
            a=[[0, 0.5], [-0.3, 0]],
            b=[[0, 500], [-300, 0]],
        )
        h300 = eval_block_T(hE, 300)
        h400 = eval_block_T(hE, 400)
        self.assertNotAlmostEqual(h300, h400, places=0)

    def test_ideal_solution(self):
        # With a=0, Lambda=1, Lambda'=0 => hE=0
        hE = ExcessEnthalpyWilson(
            x=[0.5, 0.5],
            a=[[0, 0], [0, 0]],
        )
        result = eval_block_T(hE, 350)
        self.assertAlmostEqual(result, 0.0, places=10)


class TestExcessEnthalpyRedlichKister(unittest.TestCase):

    def test_init(self):
        hE = ExcessEnthalpyRedlichKister(
            x=[0.5, 0.5],
            coeffs={(0, 1): [[1000.0]]},
        )
        self.assertEqual(hE.n, 2)

    def test_symmetric_binary(self):
        # Simplest case: one-term Redlich-Kister with constant A
        # h^E_ij = x_i*x_j/(x_i+x_j) * A * (x_i-x_j)
        # For x=[0.5, 0.5]: h^E = 0.5 * 0.5 / 1.0 * A * 0 = 0
        hE = ExcessEnthalpyRedlichKister(
            x=[0.5, 0.5],
            coeffs={(0, 1): [[1000.0]]},
        )
        result = eval_block_T(hE, 300)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_asymmetric_composition(self):
        # x=[0.3, 0.7], one-term A=2000
        # h^E = 0.5 * 0.3*0.7/1.0 * 2000 * (0.3-0.7) = 0.5 * 0.21 * 2000 * (-0.4) = -84
        hE = ExcessEnthalpyRedlichKister(
            x=[0.3, 0.7],
            coeffs={(0, 1): [[2000.0]]},
        )
        result = eval_block_T(hE, 300)
        expected = 0.5 * 0.3 * 0.7 / 1.0 * 2000.0 * (0.3 - 0.7)
        self.assertAlmostEqual(result, expected, places=5)

    def test_temperature_dependent_coeffs(self):
        # A(T) = 1000 + 2*T
        hE = ExcessEnthalpyRedlichKister(
            x=[0.4, 0.6],
            coeffs={(0, 1): [[1000.0, 2.0]]},
        )
        h300 = eval_block_T(hE, 300)
        h400 = eval_block_T(hE, 400)
        self.assertNotAlmostEqual(h300, h400, places=0)

    def test_finite(self):
        hE = ExcessEnthalpyRedlichKister(
            x=[0.4, 0.6],
            coeffs={(0, 1): [[1000.0], [500.0], [200.0]]},
        )
        result = eval_block_T(hE, 350)
        self.assertTrue(np.isfinite(result))


# TESTS: ENTHALPY DEPARTURE ============================================================

class TestEnthalpyDepartureRKS(unittest.TestCase):

    def test_init(self):
        dH = EnthalpyDepartureRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        self.assertEqual(dH.nc, 1)

    def test_ideal_gas_limit(self):
        # At very low P, departure should be near 0
        dH = EnthalpyDepartureRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        result = eval_block_TP(dH, 1000, 100)
        self.assertAlmostEqual(result, 0.0, delta=10)

    def test_finite(self):
        dH = EnthalpyDepartureRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        result = eval_block_TP(dH, 300, 101325)
        self.assertTrue(np.isfinite(result))

    def test_negative_departure_vapor(self):
        # For real gas at moderate P, departure is typically negative
        dH = EnthalpyDepartureRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        result = eval_block_TP(dH, 300, 3e6)
        self.assertLess(result, 0)

    def test_mixture(self):
        dH = EnthalpyDepartureRKS(
            Tc=[190.6, 305.3],
            Pc=[4.6e6, 4.872e6],
            omega=[0.011, 0.099],
            x=[0.7, 0.3],
        )
        result = eval_block_TP(dH, 300, 101325)
        self.assertTrue(np.isfinite(result))


class TestEnthalpyDeparturePR(unittest.TestCase):

    def test_init(self):
        dH = EnthalpyDeparturePR(Tc=190.6, Pc=4.6e6, omega=0.011)
        self.assertEqual(dH.nc, 1)

    def test_ideal_gas_limit(self):
        dH = EnthalpyDeparturePR(Tc=190.6, Pc=4.6e6, omega=0.011)
        result = eval_block_TP(dH, 1000, 100)
        self.assertAlmostEqual(result, 0.0, delta=10)

    def test_finite(self):
        dH = EnthalpyDeparturePR(Tc=190.6, Pc=4.6e6, omega=0.011)
        result = eval_block_TP(dH, 300, 101325)
        self.assertTrue(np.isfinite(result))

    def test_negative_departure_vapor(self):
        dH = EnthalpyDeparturePR(Tc=190.6, Pc=4.6e6, omega=0.011)
        result = eval_block_TP(dH, 300, 3e6)
        self.assertLess(result, 0)

    def test_pr_vs_rks_similar(self):
        # At moderate conditions, RKS and PR should give similar departure
        dH_pr = EnthalpyDeparturePR(Tc=190.6, Pc=4.6e6, omega=0.011)
        dH_rks = EnthalpyDepartureRKS(Tc=190.6, Pc=4.6e6, omega=0.011)
        h_pr = eval_block_TP(dH_pr, 300, 101325)
        h_rks = eval_block_TP(dH_rks, 300, 101325)
        # Within 20% of each other
        self.assertAlmostEqual(h_pr, h_rks, delta=abs(h_rks) * 0.2 + 1)

    def test_mixture(self):
        dH = EnthalpyDeparturePR(
            Tc=[190.6, 305.3],
            Pc=[4.6e6, 4.872e6],
            omega=[0.011, 0.099],
            x=[0.7, 0.3],
        )
        result = eval_block_TP(dH, 300, 101325)
        self.assertTrue(np.isfinite(result))


# SMOKE TEST ===========================================================================

class TestSmokeAllEnthalpy(unittest.TestCase):

    def test_excess_enthalpy_blocks(self):
        blocks = [
            ExcessEnthalpyNRTL(
                x=[0.5, 0.5], a=[[0, 1.0], [1.5, 0]],
                b=[[0, 100], [100, 0]], c=[[0, 0.3], [0.3, 0]]),
            ExcessEnthalpyUNIQUAC(
                x=[0.5, 0.5], r=[2.1, 0.92], q=[1.97, 1.4],
                a=[[0, -1.0], [2.0, 0]], b=[[0, 200], [-100, 0]]),
            ExcessEnthalpyWilson(
                x=[0.5, 0.5], a=[[0, 0.5], [-0.3, 0]],
                b=[[0, 200], [-150, 0]]),
            ExcessEnthalpyRedlichKister(
                x=[0.4, 0.6], coeffs={(0, 1): [[1000.0, 2.0]]}),
        ]

        for block in blocks:
            with self.subTest(block=block.__class__.__name__):
                result = eval_block_T(block, 350)
                self.assertTrue(np.isfinite(result),
                                f"{block.__class__.__name__} non-finite")

    def test_departure_blocks(self):
        blocks = [
            EnthalpyDepartureRKS(Tc=190.6, Pc=4.6e6, omega=0.011),
            EnthalpyDeparturePR(Tc=190.6, Pc=4.6e6, omega=0.011),
        ]

        for block in blocks:
            with self.subTest(block=block.__class__.__name__):
                result = eval_block_TP(block, 300, 101325)
                self.assertTrue(np.isfinite(result))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
