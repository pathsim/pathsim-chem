
<p align="center">
  <img src="https://raw.githubusercontent.com/pathsim/pathsim-chem/main/docs/source/logos/chem_logo.png" width="300" alt="PathSim-Chem Logo" />
</p>

<p align="center">
  <strong>Chemical engineering blocks for PathSim</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pathsim-chem/"><img src="https://img.shields.io/pypi/v/pathsim-chem" alt="PyPI"></a>
  <img src="https://img.shields.io/github/license/pathsim/pathsim-chem" alt="License">
</p>

<p align="center">
  <a href="https://docs.pathsim.org/chem">Documentation</a> &bull;
  <a href="https://pathsim.org">PathSim Homepage</a> &bull;
  <a href="https://github.com/pathsim/pathsim-chem">GitHub</a>
</p>

---

PathSim-Chem extends the [PathSim](https://github.com/pathsim/pathsim) simulation framework with blocks for chemical engineering and thermodynamic property calculations. All blocks follow the standard PathSim block interface and can be connected into simulation diagrams.

## Features

- **IK-CAPE Thermodynamics** — 50+ blocks implementing the DECHEMA IK-CAPE standard for thermodynamic property calculations
- **Pure Component Correlations** — Antoine, Wagner, Kirchhoff, Rackett, Aly-Lee, DIPPR, and 10 more temperature-dependent property correlations
- **Mixing Rules** — Linear, quadratic, Lorentz-Berthelot, and other calculation-of-averages rules for mixture properties
- **Activity Coefficients** — NRTL, Wilson, UNIQUAC, and Flory-Huggins models for liquid-phase non-ideality
- **Equations of State** — Peng-Robinson and Soave-Redlich-Kwong cubic EoS with mixture support
- **Fugacity Coefficients** — EoS-based and virial equation fugacity calculations
- **Excess Enthalpy & Departure** — NRTL, UNIQUAC, Wilson, Redlich-Kister excess enthalpy and EoS departure functions
- **Chemical Reactions** — Equilibrium constants, kinetic rate constants, and power-law rate expressions
- **Tritium Processing** — GLC columns, TCAP cascades, bubblers, and splitters for tritium separation

## Install

```bash
pip install pathsim-chem
```

## Quick Example

Compute the vapor pressure of water at 100 °C using the Antoine equation:

```python
from pathsim_chem.thermodynamics import Antoine

# Antoine coefficients for water (NIST)
antoine = Antoine(a0=23.2256, a1=3835.18, a2=-45.343)

# Evaluate at 373.15 K
antoine.inputs[0] = 373.15
antoine.update(None)
P_sat = antoine.outputs[0]  # ≈ 101325 Pa
```

Use activity coefficients in a simulation:

```python
from pathsim_chem.thermodynamics import NRTL

# Ethanol-water NRTL model
nrtl = NRTL(
    x=[0.4, 0.6],
    a=[[0, -0.801], [3.458, 0]],
    c=[[0, 0.3], [0.3, 0]],
)

# Evaluate at 350 K
nrtl.inputs[0] = 350
nrtl.update(None)
gamma_ethanol = nrtl.outputs[0]
gamma_water = nrtl.outputs[1]
```

## Modules

| Module | Description |
|---|---|
| `pathsim_chem.thermodynamics.correlations` | 16 pure component property correlations (Antoine, Wagner, etc.) |
| `pathsim_chem.thermodynamics.averages` | 10 mixing rules for calculation of averages |
| `pathsim_chem.thermodynamics.activity_coefficients` | NRTL, Wilson, UNIQUAC, Flory-Huggins |
| `pathsim_chem.thermodynamics.equations_of_state` | Peng-Robinson, Soave-Redlich-Kwong |
| `pathsim_chem.thermodynamics.corrections` | Poynting correction, Henry's law |
| `pathsim_chem.thermodynamics.fugacity_coefficients` | RKS, PR, and virial fugacity coefficients |
| `pathsim_chem.thermodynamics.enthalpy` | Excess enthalpy and enthalpy departure functions |
| `pathsim_chem.thermodynamics.reactions` | Equilibrium constants, rate constants, power-law rates |
| `pathsim_chem.tritium` | GLC, TCAP, bubbler, splitter blocks for tritium processing |

## Learn More

- [Documentation](https://docs.pathsim.org/chem) — API reference and examples
- [PathSim](https://github.com/pathsim/pathsim) — the core simulation framework
- [Contributing](https://docs.pathsim.org/pathsim/latest/contributing) — how to contribute

## License

MIT
