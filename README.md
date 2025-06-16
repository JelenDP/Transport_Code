# Molecular Electron Transport using Quantum Transport Theory

This repository provides Python scripts for calculating the **zero-bias transmission function** of **Single-Molecule Junctions (SMJs)** based on the **Non-Equilibrium Green's Function (NEGF)** formalism, combined with **high-level** ***ab initio*** **quantum chemistry** calculations.

## Main Scripts

### `negf.py`
Calculates the zero-bias transmission function using the **standard NEGF approach**.  
For theoretical background, see [1].

### `negf_emb_mrcc.py`
Calculates the transmission function based on **embedding calculations** from **MRCC**, supporting:
- **DFT-in-DFT**
- **WFT-in-DFT**  
The embedding scheme is based on the **Huzinaga equation** (i.e. **Projection-based Embedding, PbE**).  
For methodological details, see [2].

## Necessary Calculations

In line with the NEGF methodology:

### 1) **Lead**

**a. Geometry**  
- Described by periodic calculations; the unit cell must contain **two principal layers (PLs)**.

**b. Electronic Structure**  
- Requires a periodic **DFT calculation**.  
- Required quantities: Fock and overlap matrices in the **atomic orbital (AO) basis**.  
- Supported packages:
  - **TURBOMOLE** (requires a **custom-modified version** capable of exporting Fock and overlap matrices), or
  - **BAND** (part of the SCM suite; the **standard version is sufficient**).

### 2) **Extended Molecule (EM)**

**a. Geometry**  
- Built from the molecule and parts of the electrodes. The electrodes may include surface layers and tips for connecting to the molecule but must contain at least **one PL from the lead on both sides**.

**b. Electronic Structure**

- **i. DFT Calculation (Periodic or Non-Periodic)**  
  - Required quantities: Fock and overlap matrices in the **AO basis**.  
  - Supported packages:
    - **TURBOMOLE** (requires a **custom-modified version**),
    - **BAND**,
    - **MRCC** (requires a **custom-modified version**).

- **ii. Embedding Scheme**  
  - The **environment** must include the electrode atoms, and its geometry must match that used in the periodic lead calculation.  
  - The **active subsystem** may include only the molecule or the molecule plus parts of the electrodes.  
  - Supported embedding methods:
    - **DFT-in-DFT**
    - **WFT-in-DFT** (requires **Dyson orbitals**)  
  - Supported packages:
    - **MRCC** (custom-modified)

## References
- [1]: Jeremy Taylor et al., ***Phys. Rev. B***, **63**, 245407 (2001). DOI: 10.1103/PhysRevB.63.245407.
- [2]: Dávid P. Jelenfi, Attila Tajti and Péter G. Szalay, ***J. Chem. Phys.***, **162**, 034101 (2025). DOI: 10.1063/5.0238014.
