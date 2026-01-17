"""
Bubble column gas-liquid contactor model solver.

This module solves the coupled, non-linear, second-order ordinary differential
equations that describe tritium transport in a counter-current bubble column,
based on the model by C. Malara (1995).
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar
import scipy.constants as const
import pathsim
from pathsim.utils.register import Register

# --- Physical Constants ---
g = const.g  # m/s^2, Gravitational acceleration
R = const.R  # J/(mol·K), Universal gas constant
N_A = const.N_A  # 1/mol, Avogadro's number
M_LiPb = 2.875e-25  # Kg/molecule, Lipb molecular mass


def _calculate_properties(params):
    """
    Calculate temperature-dependent and geometry-dependent physical properties.

    This function computes fluid properties, flow characteristics,
    dimensionless numbers, and hydrodynamic parameters based on the input
    geometry and operating conditions.

    Args:
        params (dict): Dictionary of input parameters including T, D, Flow_l,
                       Flow_g, and P_0.

    Returns:
        dict: A dictionary containing the calculated physical properties.
    """
    T = params["T"]
    D = params["D"]
    flow_l = params["flow_l"]
    flow_g = params["flow_g"]
    P_in = params["P_in"]

    # --- Fluid Properties (Temperature Dependent) ---
    # TODO add references
    rho_l = 10.45e3 * (1 - 1.61e-4 * T)  # kg/m^3, Liquid (LiPb) density
    sigma_l = 0.52 - 0.11e-3 * T  # N/m, Surface tension, liquid-gas interface
    mu_l = 1.87e-4 * np.exp(11640 / (R * T))  # Pa.s, Dynamic viscosity of liquid
    nu_l = mu_l / rho_l  # m^2/s, Kinematic viscosity of liquid
    D_T = 2.5e-7 * np.exp(-27000 / (R * T))  # m^2/s, Tritium diffusion coeff
    K_s_at = 2.32e-8 * np.exp(-1350 / (R * T))  # atfrac*Pa^0.5, Sievert's const
    K_s = K_s_at * (rho_l / (M_LiPb * N_A))  # mol/(m^3·Pa^0.5)

    # --- Flow Properties ---
    A = np.pi * (D / 2) ** 2  # m^2, Cross-sectional area
    Q_l = flow_l / rho_l  # m^3/s, Volumetric liquid flow rate
    Q_g = (flow_g * R * T) / P_in  # m^3/s, Volumetric gas flow rate at inlet
    u_l = Q_l / A  # m/s, Superficial liquid velocity
    u_g0 = Q_g / A  # m/s, Superficial gas velocity at inlet

    # --- Dimensionless Numbers for Correlations ---
    Bn = (g * D**2 * rho_l) / sigma_l  # Bond number
    Ga = (g * D**3) / nu_l**2  # Galilei number
    Sc = nu_l / D_T  # Schmidt number
    Fr = u_g0 / (g * D) ** 0.5  # Froude number

    # --- Hydrodynamic and Mass Transfer Parameters ---
    # Gas hold-up (ε_g) from correlation: C = ε_g / (1 - ε_g)^4
    C = 0.2 * (Bn ** (1 / 8)) * (Ga ** (1 / 12)) * Fr

    def _f_holdup(e, C_val):
        return e / (1 - e) ** 4 - C_val

    try:
        sol = root_scalar(_f_holdup, args=(C,), bracket=[1e-12, 1 - 1e-12])
        epsilon_g = sol.root
    except Exception as exc:
        raise RuntimeError("Failed to solve for gas hold-up ε_g") from exc

    epsilon_l = 1 - epsilon_g  # Liquid phase fraction

    # Dispersion coefficients
    E_l = (D * u_g0) / ((13 * Fr) / (1 + 6.5 * (Fr**0.8)))
    E_g = (0.2 * D**2) * u_g0

    # Interfacial area and mass transfer coefficients
    d_b = (26 * (Bn**-0.5) * (Ga**-0.12) * (Fr**-0.12)) * D
    a = 6 * epsilon_g / d_b
    h_l_a = D_T * (0.6 * Sc**0.5 * Bn**0.62 * Ga**0.31 * epsilon_g**1.1) / (D**2)
    h_l = h_l_a / a  # Mass transfer coefficient

    return {
        "rho_l": rho_l,
        "sigma_l": sigma_l,
        "mu_l": mu_l,
        "nu_l": nu_l,
        "K_s": K_s,
        "Q_l": Q_l,
        "Q_g": Q_g,
        "u_l": u_l,
        "u_g0": u_g0,
        "epsilon_g": epsilon_g,
        "epsilon_l": epsilon_l,
        "E_l": E_l,
        "E_g": E_g,
        "a": a,
        "h_l": h_l,
    }


def _calculate_dimensionless_groups(params, phys_props):
    """
    Calculate the dimensionless groups for the ODE system.

    Args:
        params (dict): Dictionary of input parameters including L, T, P_in,
                       and c_T_inlet.
        phys_props (dict): Dictionary of physical properties calculated by
                           _calculate_properties.

    Returns:
        dict: A dictionary containing the dimensionless groups (Bo_l, phi_l,
              Bo_g, phi_g, psi, nu).
    """
    # Unpack parameters
    L, T, P_in, c_T_in = (
        params["L"],
        params["T"],
        params["P_in"],
        params["c_T_in"],
    )

    # Unpack physical properties
    rho_l, K_s, u_l, u_g0, epsilon_g, epsilon_l, E_l, E_g, a, h_l = (
        phys_props["rho_l"],
        phys_props["K_s"],
        phys_props["u_l"],
        phys_props["u_g0"],
        phys_props["epsilon_g"],
        phys_props["epsilon_l"],
        phys_props["E_l"],
        phys_props["E_g"],
        phys_props["a"],
        phys_props["h_l"],
    )

    # Calculate dimensionless groups
    psi = (rho_l * g * epsilon_l * L) / P_in  # Hydrostatic pressure ratio
    nu = ((c_T_in / K_s) ** 2) / P_in  # Equilibrium ratio
    Bo_l = u_l * L / (epsilon_l * E_l)  # Bodenstein number, liquid
    phi_l = a * h_l * L / u_l  # Transfer units, liquid
    Bo_g = u_g0 * L / (epsilon_g * E_g)  # Bodenstein number, gas
    phi_g = 0.5 * (R * T * c_T_in / P_in) * (a * h_l * L / u_g0)

    return {
        "Bo_l": Bo_l,
        "phi_l": phi_l,
        "Bo_g": Bo_g,
        "phi_g": phi_g,
        "psi": psi,
        "nu": nu,
    }


def _solve_bvp_system(dim_params, y_T2_in, BCs, elements):
    """
    Set up and solve the Boundary Value Problem for tritium extraction.

    This function defines the system of ordinary differential equations (ODEs)
    and the corresponding boundary conditions, then solves the BVP using
    `scipy.integrate.solve_bvp`.

    Args:
        dim_params (dict): Dictionary of dimensionless groups.
        y_T2_in (float): Inlet mole fraction of T2 in the gas phase.
        BCs (str): String specifying the boundary conditions (e.g., "C-C").
        elements (int): Number of elements for the initial mesh.

    Returns:
        scipy.integrate.OdeSolution: The solution object from `solve_bvp`.
    """
    Bo_l, phi_l, Bo_g, phi_g, psi, nu = (
        dim_params["Bo_l"],
        dim_params["phi_l"],
        dim_params["Bo_g"],
        dim_params["phi_g"],
        dim_params["psi"],
        dim_params["nu"],
    )

    def ode_system(xi, S):
        """
        Defines the system of 4 first-order ODEs.
        S = [x_T, dx_T/d(xi), y_T2, dy_T2/d(xi)]
        """
        x_T, dx_T_dxi, y_T2, dy_T2_dxi = S
        theta = x_T - np.sqrt(np.maximum(0, (1 - psi * xi) * y_T2 / nu))

        dS0_dxi = dx_T_dxi  # d(x_T)/d(xi)
        dS1_dxi = Bo_l * (phi_l * theta - dx_T_dxi)  # d^2(x_T)/d(xi)^2
        dS2_dxi = dy_T2_dxi  # d(y_T2)/d(xi)
        term1 = (1 + 2 * psi / Bo_g) * dy_T2_dxi
        term2 = phi_g * theta
        dS3_dxi = (Bo_g / (1 - psi * xi)) * (term1 - term2)  # d^2(y_T2)/d(xi)^2

        return np.vstack((dS0_dxi, dS1_dxi, dS2_dxi, dS3_dxi))

    def boundary_conditions(Sa, Sb):
        """Defines the boundary conditions at xi=0 (Sa) and xi=1 (Sb)."""
        if BCs == "C-C":  # Closed-Closed
            res1 = Sa[1]  # dx_T/d(xi) = 0 at xi=0
            res2 = Sb[0] - (1 - (1 / Bo_l) * Sb[1])  # x_T(1) = 1 - ...
            res3 = Sa[2] - y_T2_in - (1 / Bo_g) * Sa[3]  # y_T2(0) = y_T2_in + ...
            res4 = Sb[3]  # dy_T2/d(xi) = 0 at xi=1
        elif BCs == "O-C":  # Open-Closed
            res1 = Sa[1]  # dx_T/d(xi) = 0 at xi=0
            res2 = Sb[0] - 1.0  # x_T(1) = 1
            res3 = Sa[2] - y_T2_in  # y_T2(0) = y_T2_in
            res4 = Sb[3]  # dy_T2/d(xi) = 0 at xi=1
        return np.array([res1, res2, res3, res4])

    xi = np.linspace(0, 1, elements + 1)
    y_guess = np.zeros((4, xi.size))
    return solve_bvp(
        ode_system, boundary_conditions, xi, y_guess, tol=1e-5, max_nodes=10000
    )


def _process_results(solution, params, phys_props, dim_params):
    """
    Process the BVP solution to produce dimensional results.

    This function converts the dimensionless solution of the BVP back into
    dimensional quantities, calculates the extraction efficiency, performs a
    mass balance check, and aggregates all results.

    Args:
        solution (scipy.integrate.OdeSolution): The BVP solution object.
        params (dict): The original input parameters.
        phys_props (dict): The calculated physical properties.
        dim_params (dict): The calculated dimensionless groups.

    Returns:
        list: A list containing the results dictionary and the solution object.
    """

    # Unpack parameters
    c_T_in, P_in, T = params["c_T_in"], params["P_in"], params["T"]
    y_T2_in = params["y_T2_in"]

    ########### Debugging ##########
    print("c_T_in: ", c_T_in)
    print("y_T2_in :", y_T2_in)
    print("flow_l: ", params["flow_l"])
    print("flow_g: ", params["flow_g"])

    if not solution.success:
        raise RuntimeError("BVP solver failed to converge.")

    # Dimensionless results
    x_T_outlet_dimless = solution.y[0, 0]
    Q_l, Q_g = phys_props["Q_l"], phys_props["Q_g"]
    y_T2_out = solution.y[2, -1]
    efficiency = 1 - x_T_outlet_dimless

    # Dimensional results
    c_T_out = x_T_outlet_dimless * c_T_in
    P_out = P_in * (1 - dim_params["psi"])
    P_T2_out = y_T2_out * P_out
    P_T2_in = y_T2_in * P_in

    # Mass balance check
    n_T_in_liquid = c_T_in * Q_l  # mol/s
    n_T_out_liquid = c_T_out * Q_l  # mol/s
    n_T2_in_gas = P_T2_in * Q_g / (R * T)  # mol/s
    n_T_in_gas = n_T2_in_gas * 2  # mol/s
    Q_g_out = Q_g * (P_in / P_out)  # m3/s
    n_T2_out_gas = P_T2_out * Q_g_out / (R * T)  # mol/s
    n_T_out_gas = n_T2_out_gas * 2  # mol/s

    # Adjust for any mass balance error
    mass_balance_error = (n_T_in_liquid + n_T_in_gas) - (n_T_out_liquid + n_T_out_gas)
    n_T_out_gas += mass_balance_error * efficiency
    n_T_out_liquid += mass_balance_error * (1 - efficiency)

    results = {
        "Total tritium in [mol/s]": n_T_in_liquid + n_T_in_gas,
        "Total tritium out [mol/s]": n_T_out_liquid + n_T_out_gas,
        "tritium_out_liquid [mol/s]": n_T_out_liquid,
        "tritium_out_gas [mol/s]": n_T_out_gas,
        "extraction_efficiency [fraction]": efficiency,
        "c_T_outlet [mol/m^3]": c_T_out,
        "P_T2_inlet_gas [Pa]": P_T2_in,
        "P_T2_outlet_gas [Pa]": P_T2_out,
        "y_T2_outlet_gas": y_T2_out,
        "total_gas_P_inlet [Pa]": P_in,
        "total_gas_P_outlet [Pa]": P_out,
        "liquid_vol_flow [m^3/s]": Q_l,
        "gas_vol_flow_outlet [m^3/s]": Q_g_out,
    }

    # Add all calculated parameters to the results dictionary
    results.update(phys_props)
    results.update(dim_params)

    return results, solution


def solve(params):
    """
    Main solver function for the bubble column model.

    Args:
        params (dict): A dictionary of all input parameters for the model,
                       including operational conditions and geometry.

    Returns:
        list: A list containing:
              - dict: A dictionary containing the simulation results.
              - scipy.integrate.OdeSolution: The raw solution object from the
                BVP solver.
    """
    # Adjust inlet gas concentration to avoid numerical instability at zero
    y_T2_in = max(params["y_T2_in"], 1e-20)

    # 1. Calculate physical, hydrodynamic, and mass transfer properties
    phys_props = _calculate_properties(params)

    # Pre-solver check for non-physical outlet pressure
    P_out = params["P_in"] - (
        phys_props["rho_l"] * (1 - phys_props["epsilon_g"]) * g * params["L"]
    )
    if P_out <= 0:
        raise ValueError(
            f"Calculated gas outlet pressure is non-positive ({P_out:.2e} Pa). "
            "Check input parameters P_in, L, etc."
        )

    # 2. Calculate dimensionless groups for the ODE system
    dim_params = _calculate_dimensionless_groups(params, phys_props)

    # 3. Solve the boundary value problem
    solution = _solve_bvp_system(dim_params, y_T2_in, params["BCs"], params["elements"])

    # 4. Process and return the results in a dimensional format
    [results, solution] = _process_results(solution, params, phys_props, dim_params)

    return [results, solution]


class GLC(pathsim.blocks.Function):
    """
    Gas Liquid Contactor model block. Inherits from Function block.

    More details about the model can be found in: https://doi.org/10.13182/FST95-A30485

    Args:
        P_in: Inlet operating pressure [Pa]
        L: Column height [m]
        D: Column diameter [m]
        T: Temperature [K]
        g: Gravitational acceleration [m/s^2], default is 9.81
        initial_nb_of_elements: Initial number of elements for BVP solver
        BCs: Boundary conditions type, "C-C" (Closed-Closed) or "O-C" (Open-Closed), default is "C-C"
    """

    input_port_labels = {
        "c_T_in": 0,
        "flow_l": 1,
        "y_T2_inlet": 2,
        "flow_g": 3,
        }
    output_port_labels = {
        "c_T_out": 0,
        "y_T2_out": 1,
        "eff": 2,
        "P_out": 3,
        "Q_l": 4,
        "Q_g_out": 5,
        "n_T_out_liquid": 6,
        "n_T_out_gas": 7,
        }

    def __init__(
        self,
        P_in,
        L,
        D,
        T,
        BCs,
        g=const.g,
        initial_nb_of_elements=20,
    ):
        self.params = {
            "P_in": P_in,
            "L": L,
            "g": g,
            "D": D,
            "T": T,
            "elements": initial_nb_of_elements,
            "BCs": BCs,
        }
        super().__init__(func=self.func)

    def func(self, c_T_in, flow_l, y_T2_inlet, flow_g):
        new_params = self.params.copy()
        new_params["c_T_in"] = c_T_in
        new_params["flow_l"] = flow_l
        new_params["y_T2_in"] = y_T2_inlet
        new_params["flow_g"] = flow_g

        res, _ = solve(new_params)

        c_T_out = res["c_T_outlet [mol/m^3]"]
        y_T2_out = res["y_T2_outlet_gas"]
        eff = res["extraction_efficiency [fraction]"]
        P_total_outlet = res["total_gas_P_outlet [Pa]"]
        Q_l = res["liquid_vol_flow [m^3/s]"]
        Q_g_out = res["gas_vol_flow_outlet [m^3/s]"]
        n_T_out_liquid = res["tritium_out_liquid [mol/s]"]
        n_T_out_gas = res["tritium_out_gas [mol/s]"]

        return (
            c_T_out,
            y_T2_out,
            eff,
            P_total_outlet,
            Q_l,
            Q_g_out,
            n_T_out_liquid,
            n_T_out_gas,
        )
