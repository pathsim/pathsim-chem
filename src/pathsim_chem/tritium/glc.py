# GLC block
import pathsim
import numpy as np
from scipy.integrate import solve_bvp
from scipy import constants as const
from scipy.optimize import root_scalar


def solve(params):
    def solve_tritium_extraction(dimensionless_params, y_T2_in, elements):
        """
        Solves the BVP for tritium extraction in a bubble column.

        Args:
            params (dict): A dictionary containing the dimensionless parameters:
                        Bo_l, phi_l, Bo_g, phi_g, psi, nu.
            y_T2_in (float): Inlet tritium molar fraction in the gas phase, y_T2(0-).

        Returns:
            sol: The solution object from scipy.integrate.solve_bvp.
        """

        Bo_l = dimensionless_params["Bo_l"]
        phi_l = dimensionless_params["phi_l"]
        Bo_g = dimensionless_params["Bo_g"]
        phi_g = dimensionless_params["phi_g"]
        psi = dimensionless_params["psi"]
        nu = dimensionless_params["nu"]

        def ode_system(xi, S):
            """
            Defines the system of 4 first-order ODEs.
            S[0] = x_T  (dimensionless liquid concentration)
            S[1] = dx_T/d(xi)
            S[2] = y_T2 (dimensionless gas concentration)
            S[3] = dy_T2/d(xi)

            x_T = c_T / c_T(L+)
            xi = z / L (dimensionless position)
            """
            x_T, dx_T_dxi, y_T2, dy_T2_dxi = S

            # Dimensionless driving force theta. Eq. 12
            theta = x_T - np.sqrt(((1 - (psi * xi)) / nu) * y_T2)  # MATCHES PAPER

            # Equation for d(S[0])/d(xi) = d(x_T)/d(xi)
            dS0_dxi = dx_T_dxi

            # Equation for d(S[1])/d(xi) = d^2(x_T)/d(xi)^2
            dS1_dxi = Bo_l * (phi_l * theta - dx_T_dxi)

            # Equation for d(S[2])/d(xi) = d(y_T2)/d(xi)
            dS2_dxi = dy_T2_dxi

            # Equation for d(S[3])/d(xi) = d^2(y_T2)/d(xi)^2 from eq (11)
            # Avoid division by zero if (1 - psi * xi) is close to zero at xi=1

            term1 = (1 + 2 * psi / Bo_g) * dy_T2_dxi  # Part of Eq. 9.3.3 (fourth line)
            term2 = phi_g * theta  # Part of Eq. 9.3.3 (fourth line)
            dS3_dxi = (Bo_g / (1 - psi * xi)) * (
                term1 - term2
            )  # Eq. 9.3.3 (fourth line)

            return np.vstack((dS0_dxi, dS1_dxi, dS2_dxi, dS3_dxi))

        def boundary_conditions(Sa, Sb):
            """
            Defines the boundary conditions for the BVP.
            Sa: solution at xi = 0 (liquid outlet)
            Sb: solution at xi = 1 (liquid inlet)
            """
            # Residuals that should be zero for a valid solution.
            # Based on equations (16) and (17) in the paper.

            # At xi = 0: dx_T/d(xi) = 0
            res1 = Sa[1]  # Eq. 10.1

            # At xi = 1: x_T(1) = 1 - (1/Bo_l) * dx_T/d(xi)|_1
            res2 = Sb[0] - (1 - (1 / Bo_l) * Sb[1])  # Eq. 10.2

            # At xi = 0: y_T2(0) = y_T2(0-) + (1/Bo_g) * dy_T2/d(xi)|_0
            res3 = Sa[2] - y_T2_in - (1 / Bo_g) * Sa[3]  # Eq. 10.3

            # At xi = 1: dy_T2/d(xi) = 0
            res4 = Sb[3]  # Eq. 10.4

            return np.array([res1, res2, res3, res4])

        # Set up the mesh and an initial guess for the solver.
        xi = np.linspace(0, 1, elements + 1)

        y_guess = np.zeros((4, xi.size))

        # Run the BVP solver
        sol = solve_bvp(
            ode_system, boundary_conditions, xi, y_guess, tol=1e-6, max_nodes=10000
        )

        return sol

    # Unpack parameters
    c_T_inlet = params["c_T_inlet"]
    y_T2_in = params["y_T2_in"]
    P_0 = params["P_0"]

    L = params["L"]
    D = params["D"]

    flow_l = params["flow_l"]
    u_g0 = params["u_g0"]

    g = params["g"]
    T = params["T"]

    elements = params["elements"]  # Number of mesh elements for solver

    # --- Constants ---
    g = const.g  # m/s^2, Gravitational acceleration
    R = const.R  # J/(mol·K), Universal gas constant
    N_A = const.N_A  # 1/mol, Avogadro's number
    M_LiPb = 2.875e-25  # Kg/molecule, Lipb molecular mass

    # Calculate empirical correlations
    ρ_l = 10.45e3 * (1 - 1.61e-4 * T)  # kg/m^3, Liquid (LiPb) density
    σ_l = 0.52 - 0.11e-3 * T  # N/m, Surface tension, liquid (LiPb) - gas (He) interface
    μ_l = 1.87e-4 * np.exp(11640 / (R * T))  # Pa.s, Dynamic viscosity of liquid LiPb
    ν_l = μ_l / ρ_l  # m^2/s, Kinematic viscosity of liquid LiPb
    D_T = 2.5e-7 * np.exp(
        -27000 / (R * T)
    )  # m^2/s, Tritium diffusion coefficient in liquid LiPb

    K_s = 2.32e-8 * np.exp(
        -1350 / (R * T)
    )  # atfrac*Pa^0.5, Sievert's constant for tritium in liquid LiPb
    K_s = K_s * (ρ_l / (M_LiPb * N_A))  # mol/(m^3·Pa^0.5)

    A = np.pi * (D / 2) ** 2  # m^2, Cross-sectional area of the column

    # Calculate the volumetric flow rates
    Q_l = flow_l / ρ_l  # m^3/s, Volumetric flow rate of liquid phase
    Q_g = u_g0 * A  # m^3/s, Volumetric flow rate of gas phase

    # Calculate the superficial flow velocities
    # u_g0 =  Q_g / A  # m/s, superficial gas inlet velocity
    u_l = Q_l / A  # m/s, superficial liquid inlet velocity

    # Calculate Bond, Galilei, Schmidt and Froude numbers
    Bn = (g * D**2 * ρ_l) / σ_l  # Bond number
    Ga = (g * D**3) / ν_l**2  # Galilei number
    Sc = ν_l / D_T  # Schmidt number
    Fr = u_g0 / (g * D) ** 0.5  # Froude number

    # Calculate dispersion coefficients
    E_l = (D * u_g0) / (
        (13 * Fr) / (1 + 6.5 * (Fr**0.8))
    )  # m^2/s, Effective axial dispersion coefficient, liquid phase
    E_g = (
        0.2 * D**2
    ) * u_g0  # m^2/s, Effective axial dispersion coefficient, gas phase

    # Calculate gas hold-up (phase fraction) & mass transfer coefficient
    C = 0.2 * (Bn ** (1 / 8)) * (Ga ** (1 / 12)) * Fr  # C = ε_g / (1 - ε_g)^4

    def solveEqn(ε_g, C):
        # Define the equation to solve
        eqn = ε_g / (1 - ε_g) ** 4 - C
        return eqn

    ε_g_initial_guess = 0.1
    try:
        # bracket=[0.0001, 0.9999] tells it to *only* look in this range
        sol = root_scalar(solveEqn, args=(C,), bracket=[0.00001, 0.99999])

        # print(f"--- Using root_scalar (robust method) ---")
        # print(f"C value was: {C}")
        # if sol.converged:
        #     print(f"Solved gas hold-up (εg): {sol.root:.6f}")
        #     # Verify it
        #     verification = sol.root / (1 - sol.root)**4
        #     print(f"Verification (should equal C): {verification:.6f}")
        # else:
        #     print("Solver did not converge.")

    except ValueError as e:
        print(
            f"Solver failed. This can happen if C is so large that no solution exists between 0 and 1."
        )
        print(f"Error: {e}")

    ε_g = sol.root  # Gas phase fraction
    ε_l = 1 - ε_g  # Liquid phase fraction

    # Calculate outlet pressure hydrostatically & check non-negative
    P_outlet = P_0 - (ρ_l * (1 - ε_g) * g * L)

    if P_outlet <= 0:
        raise ValueError(
            f"Calculated gas outlet pressure P_outlet must be positive, but got {P_outlet:.2e} Pa. Check P_0, rho_l, g, and L are realistic."
        )

    # Calculate interfacial area
    d_b = (
        26 * (Bn**-0.5) * (Ga**-0.12) * (Fr**-0.12)
    ) * D  # m, Mean bubble diameter AGREES WITH PAPER
    a = 6 * ε_g / d_b  # m^-1, Specific interfacial area, assuming spherical bubbles

    # Calculate volumetric mass transfer coefficient, liquid-gas
    h_l_a = (
        D_T * (0.6 * Sc**0.5 * Bn**0.62 * Ga**0.31 * ε_g**1.1) / (D**2)
    )  # Volumetric mass transfer coefficient, liquid-gas

    h_l = h_l_a / a  # Mass transfer coefficient

    # Calculate dimensionless values

    # Hydrostatic pressure ratio (Eq. 8.3) # MATCHES PAPER
    psi = (ρ_l * g * (1 - ε_g) * L) / P_0
    # Tritium partial pressure ratio (Eq. 8.5) # MATCHES PAPER
    nu = ((c_T_inlet / K_s) ** 2) / P_0
    # Bodenstein number, liquid phase (Eq. 8.9) # MATCHES PAPER
    Bo_l = u_l * L / (ε_l * E_l)
    # Transfer units parameter, liquid phase (Eq. 8.11) # MATCHES PAPER
    phi_l = a * h_l * L / u_l
    # Bodenstein number, gas phase (Eq. 8.10) # MATCHES PAPER ASSUMING u_g0
    Bo_g = u_g0 * L / (ε_g * E_g)
    # Transfer units parameter, gas phase (Eq. 8.12) # MATCHES PAPER
    phi_g = 0.5 * (R * T * c_T_inlet / P_0) * (a * h_l * L / u_g0)

    dimensionless_params = {
        "Bo_l": Bo_l,
        "phi_l": phi_l,
        "Bo_g": Bo_g,
        "phi_g": phi_g,
        "psi": psi,
        "nu": nu,
    }

    # Solve the model
    solution = solve_tritium_extraction(dimensionless_params, y_T2_in, elements)

    # --- Results ---
    if solution.success:
        # --- Dimensionless Results ---
        x_T_outlet_dimless = solution.y[0, 0]
        efficiency = 1 - x_T_outlet_dimless
        y_T2_outlet_gas = solution.y[2, -1]  # y_T2 at xi=1

        # --- Dimensional Results ---
        # Liquid concentration at outlet (xi=0)
        c_T_outlet = x_T_outlet_dimless * c_T_inlet

        # Gas partial pressure at outlet (xi=1)
        P_outlet = P_0 * (
            1 - dimensionless_params["psi"]
        )  # Derived from Eq. 8.4 at xi=1
        P_T2_out = y_T2_outlet_gas * P_outlet

        # Mass transfer consistency check
        N_A = const.N_A  # Avogadro's number, 1/mol
        # Tritium molar flow rate into the column via liquid
        n_T_in_liquid = c_T_inlet * Q_l * N_A  # Triton/s

        # Tritium molar flow rate out of the column via liquid
        n_T_out_liquid = c_T_outlet * Q_l * N_A  # Tritons/s

        # Tritium molar flow rate into the column via gas
        P_T2_in = y_T2_in * P_0  # [Pa]
        n_T2_in_gas = (P_T2_in * Q_g / (const.R * T)) * N_A  # T2/s
        n_T_in_gas = n_T2_in_gas * 2  # Triton/s

        # Calculate outlet gas volumetric flow rate (gas expands as pressure drops)
        Q_g_out = (P_0 * Q_g) / P_outlet
        # Tritium molar flow rate out of the column via gas
        n_T2_out_gas = (P_T2_out * Q_g_out / (const.R * T)) * N_A  # T2/s
        n_T_out_gas = n_T2_out_gas * 2  # Triton/s

        T_in = n_T_in_liquid + n_T_in_gas
        T_out = n_T_out_liquid + n_T_out_gas

        results = {
            "extraction_efficiency [%]": efficiency * 100,
            "c_T_inlet [mol/m^3]": c_T_inlet,
            "c_T_outlet [mol/m^3]": c_T_outlet,
            "liquid_vol_flow [m^3/s]": Q_l,
            "P_T2_inlet_gas [Pa]": P_T2_in,
            "P_T2_outlet_gas [Pa]": P_T2_out,
            "total_gas_P_outlet [Pa]": P_outlet,
            "gas_vol_flow [m^3/s]": Q_g,
            "tritium_out_liquid [mol/s]": n_T_out_liquid / N_A,
            "tritium_out_gas [mol/s]": n_T_out_gas / N_A,
        }
        return results
    else:
        raise RuntimeError("BVP solver did not converge.")


class GLC(pathsim.blocks.Function):
    """
    Gas Liquid Contactor model block. Inherits from Function block.
    Inputs: c_T_inlet [mol/m^3], P_T2_inlet [Pa]
    Outputs: n_T_out_liquid [mol/s], n_T_out_gas [mol/s]

    More details about the model can be found in: https://doi.org/10.13182/FST95-A30485

    Args:
        P_0: Inlet operating pressure [Pa]
        L: Column height [m]
        u_g0: Superficial gas inlet velocity [m/s]
        Q_l: Liquid volumetric flow rate [m^3/s]
        D: Column diameter [m]
        T: Temperature [K]
        g: Gravitational acceleration [m/s^2], default is 9.81
    """

    _port_map_in = {
        "c_T_inlet": 0,
        "y_T2_in": 1,
    }
    _port_map_out = {
        "c_T_outlet": 0,
        "P_T2_out_gas": 1,
        "efficiency": 2,
    }

    def __init__(
        self,
        P_0,
        L,
        u_g0,
        flow_l,
        D,
        T,
        g=const.g,
        initial_nb_of_elements=20,
    ):
        self.params = {
            "P_0": P_0,
            "L": L,
            "u_g0": u_g0,
            "flow_l": flow_l,
            "g": g,
            "D": D,
            "T": T,
            "elements": initial_nb_of_elements,
        }
        super().__init__(func=self.func)

    def func(self, c_T_inlet, y_T2_inlet):
        new_params = self.params.copy()
        new_params["c_T_inlet"] = c_T_inlet
        new_params["y_T2_in"] = y_T2_inlet

        res = solve(new_params)

        c_T_outlet = res["c_T_outlet [mol/m^3]"]
        P_T2_outlet = res["P_T2_outlet_gas [Pa]"]

        n_T_out_liquid = res["tritium_out_liquid [mol/s]"]
        n_T_out_gas = res["tritium_out_gas [mol/s]"]
        eff = res["extraction_efficiency [%]"]

        return c_T_outlet, P_T2_outlet, eff
