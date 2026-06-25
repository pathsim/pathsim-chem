########################################################################################
##
##                                  TESTS FOR
##                                'tritium.glc.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from scipy.integrate import solve_bvp

from pathsim_chem.tritium import GLC
from pathsim_chem.tritium import glc

from pathsim import Simulation, Connection
from pathsim.blocks import BVP1D, Constant, Scope


# HELPERS =============================================================================

#fixed operating point and geometry used across the tests
_BASE = dict(P_in=2e5, L=1.0, D=0.1, T=623.0, g=9.81)

#per-evaluation boundary data (c_T_in, flow_l, y_T2_inlet, flow_g)
_INPUT = (1e-3, 1.0, 0.0, 1e-4)


def _reference(BCs):
    """Independent gold-standard solve of the Malara model with scipy.solve_bvp.

    Returns the dimensionless outlet liquid concentration ``x_T(0)`` and the
    dimensionless outlet gas fraction ``y_T2(1)``.
    """
    c_T_in, flow_l, y_T2_inlet, flow_g = _INPUT
    p = dict(_BASE)
    p.update(elements=20, BCs=BCs, c_T_in=c_T_in, flow_l=flow_l,
             flow_g=flow_g, y_T2_in=y_T2_inlet)

    phys = glc._calculate_properties(p)
    dim = glc._calculate_dimensionless_groups(p, phys)
    y2in = max(p["y_T2_in"], 1e-20)

    Bo_l, phi_l, Bo_g, phi_g, psi, nu = (
        dim["Bo_l"], dim["phi_l"], dim["Bo_g"], dim["phi_g"], dim["psi"], dim["nu"]
    )

    def ode(xi, S):
        x_T, dx, y_T2, dy = S
        th = x_T - np.sqrt(np.maximum(0, (1 - psi * xi) * y_T2 / nu))
        return np.vstack((
            dx,
            Bo_l * (phi_l * th - dx),
            dy,
            (Bo_g / (1 - psi * xi)) * ((1 + 2 * psi / Bo_g) * dy - phi_g * th),
        ))

    def bc(Sa, Sb):
        if BCs == "C-C":
            return np.array([Sa[1], Sb[0] - (1 - (1 / Bo_l) * Sb[1]),
                             Sa[2] - y2in - (1 / Bo_g) * Sa[3], Sb[3]])
        return np.array([Sa[1], Sb[0] - 1.0, Sa[2] - y2in, Sb[3]])

    xi = np.linspace(0, 1, 21)
    sol = solve_bvp(ode, bc, xi, np.zeros((4, 21)), tol=1e-5, max_nodes=10000)
    return sol.y[0, 0], sol.y[2, -1]


# TESTS ===============================================================================

class TestGLC(unittest.TestCase):
    """
    Test the implementation of the 'GLC' bubble-column gas-liquid contactor
    block from the fusion toolbox. The block inherits from the native `BVP1D`
    block and seeds it with the Malara (1995) tritium-extraction model.
    """

    def test_init(self):
        """The block is a BVP1D subclass seeded with the GLC problem dimensions."""
        blk = GLC(BCs="C-C", **_BASE)
        self.assertIsInstance(blk, BVP1D)
        self.assertEqual(blk.n, 4)
        self.assertEqual(blk.domain, (0.0, 1.0))
        #four named input ports supply the boundary data
        self.assertEqual(len(blk.inputs), 4)
        for label in ("c_T_in", "flow_l", "y_T2_inlet", "flow_g"):
            self.assertIn(label, blk.input_port_labels)
        #eight named dimensional output ports
        for label in ("c_T_out", "y_T2_out", "eff", "P_out", "Q_l", "Q_g_out",
                      "n_T_out_liquid", "n_T_out_gas"):
            self.assertIn(label, blk.output_port_labels)
        #no successful solve yet, so no dimensional results
        self.assertIsNone(blk.results())


    def test_solve_matches_reference_cc(self):
        """The C-C solve matches an independent scipy.solve_bvp reference."""
        blk = GLC(BCs="C-C", **_BASE)
        blk.inputs.update_from_array(np.array(_INPUT))
        blk.update(0.0)

        self.assertTrue(blk.success)
        x_T_ref, y_T2_ref = _reference("C-C")
        res = blk.results()
        self.assertAlmostEqual(res["c_T_outlet [mol/m^3]"],
                               x_T_ref * _INPUT[0], places=12)
        self.assertAlmostEqual(res["y_T2_outlet_gas"], y_T2_ref, places=12)


    def test_solve_matches_reference_oc(self):
        """The O-C solve matches an independent scipy.solve_bvp reference."""
        blk = GLC(BCs="O-C", **_BASE)
        blk.inputs.update_from_array(np.array(_INPUT))
        blk.update(0.0)

        self.assertTrue(blk.success)
        x_T_ref, y_T2_ref = _reference("O-C")
        res = blk.results()
        self.assertAlmostEqual(res["c_T_outlet [mol/m^3]"],
                               x_T_ref * _INPUT[0], places=12)
        self.assertAlmostEqual(res["y_T2_outlet_gas"], y_T2_ref, places=12)

        #the open inlet condition neglects the dispersive flux, so O-C does not
        #close the mass balance exactly; the residual is small but non-trivial
        #(a few per mille) and is reported honestly rather than hidden
        rel = abs(res["mass_balance_residual [mol/s]"]) / res["Total tritium in [mol/s]"]
        self.assertGreater(rel, 1e-6)
        self.assertLess(rel, 1e-2)


    def test_results_physical(self):
        """The dimensional results are physically sensible."""
        blk = GLC(BCs="C-C", **_BASE)
        blk.inputs.update_from_array(np.array(_INPUT))
        blk.update(0.0)
        res = blk.results()

        #extraction efficiency in [0, 1] and outlet below inlet concentration
        self.assertGreaterEqual(res["extraction_efficiency [fraction]"], 0.0)
        self.assertLessEqual(res["extraction_efficiency [fraction]"], 1.0)
        self.assertLess(res["c_T_outlet [mol/m^3]"], _INPUT[0])

        #hydrostatic head drops the gas pressure below the inlet pressure
        self.assertLess(res["total_gas_P_outlet [Pa]"], _BASE["P_in"])

        #closed-closed boundary conditions conserve tritium to numerical noise;
        #the residual is reported, not redistributed back into the outputs
        rel = abs(res["mass_balance_residual [mol/s]"]) / res["Total tritium in [mol/s]"]
        self.assertLess(rel, 1e-6)


    def test_output_ports(self):
        """The eight dimensional output ports match the results dictionary."""
        blk = GLC(BCs="C-C", **_BASE)
        blk.inputs.update_from_array(np.array(_INPUT))
        blk.update(0.0)

        res = blk.results()
        for label, key in (
            ("c_T_out", "c_T_outlet [mol/m^3]"),
            ("y_T2_out", "y_T2_outlet_gas"),
            ("eff", "extraction_efficiency [fraction]"),
            ("P_out", "total_gas_P_outlet [Pa]"),
            ("Q_l", "liquid_vol_flow [m^3/s]"),
            ("Q_g_out", "gas_vol_flow_outlet [m^3/s]"),
            ("n_T_out_liquid", "tritium_out_liquid [mol/s]"),
            ("n_T_out_gas", "tritium_out_gas [mol/s]"),
        ):
            idx = blk.output_port_labels[label]
            self.assertAlmostEqual(blk.outputs[idx], res[key], places=12)


    def test_unknown_bc_raises(self):
        """An unknown boundary-condition type is rejected at solve time."""
        blk = GLC(BCs="bogus", **_BASE)
        blk.inputs.update_from_array(np.array(_INPUT))
        with self.assertRaises(ValueError):
            blk.update(0.0)


    def test_input_unchanged_skips_resolve(self):
        """A re-evaluation with unchanged input does not re-solve the BVP."""
        import pathsim.blocks.bvp as bvpmod

        blk = GLC(BCs="C-C", **_BASE)
        blk.inputs.update_from_array(np.array(_INPUT))

        calls = [0]
        orig = bvpmod.solve_bvp
        def _counting(*a, **k):
            calls[0] += 1
            return orig(*a, **k)
        bvpmod.solve_bvp = _counting
        try:
            blk.update(0.0)            #first solve
            blk.update(0.0)            #unchanged input -> skipped
            self.assertEqual(calls[0], 1)

            blk.inputs[3] = 2e-4       #flow_g
            blk.update(0.0)            #changed input -> solve again
            self.assertEqual(calls[0], 2)
        finally:
            bvpmod.solve_bvp = orig


    def test_non_physical_pressure_raises(self):
        """A column tall enough to drive the outlet pressure non-positive fails."""
        #very tall column at low inlet pressure -> hydrostatic head exceeds P_in
        blk = GLC(P_in=1e4, L=10.0, D=0.1, T=623.0, BCs="C-C")
        blk.inputs.update_from_array(np.array(_INPUT))
        with self.assertRaises(ValueError):
            blk.update(0.0)


    def test_simulation(self):
        """The block runs inside a Simulation driven by constant sources."""
        srcs = [Constant(v) for v in _INPUT]
        block = GLC(BCs="C-C", **_BASE)
        sco = Scope()

        sim = Simulation(
            blocks=[*srcs, block, sco],
            connections=(
                [Connection(srcs[i], block[i]) for i in range(4)]
                #connect the named efficiency output port, as a downstream
                #consumer would
                + [Connection(block["eff"], sco[0])]
            ),
            log=False,
        )
        sim.run(0.05)

        self.assertTrue(block.success)
        res = block.results()
        np.testing.assert_allclose(
            block.outputs[block.output_port_labels["eff"]],
            res["extraction_efficiency [fraction]"], atol=1e-12
        )


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
