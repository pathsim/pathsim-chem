
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

## License

MIT
