<div align="left">
  <img src="./docs/logo/logo-xtddft.jpg" height="80px"/>
</div>

# XTDDFT_dev

Spin-adapted time-dependent density functional theory for open-shell systems.
The current development code is on the `main` branch. Response classes are
in `XTDDFT/`, shared numerical helpers are in `utils/`, and runnable
examples are in `examples/`.

## Installation

The package requires Python 3.9 or newer. For a CPU installation:

```bash
pip install -e .
pip install -e ".[test]"       # optional: pytest
```

GPU calculations use CuPy and GPU4PySCF. Select the extra matching the
installed CUDA runtime:

```bash
pip install -e ".[gpu-cuda11x]"
# or
pip install -e ".[gpu-cuda12x]"
```

The CUDA version of CuPy, GPU4PySCF, cuTENSOR, and the GPU4PySCF LibXC
extension must be mutually compatible. If the CUDA LibXC wheel is unavailable
for the chosen runtime, install the matching GPU4PySCF LibXC extension from
the GPU4PySCF source tree.

## Backend selection

Select the numerical backend before constructing a reference or response
object:

```python
from XTDDFT_dev.utils.backend import set_backend

set_backend("cpu")       # NumPy/PySCF
# set_backend("gpu")     # CuPy/GPU4PySCF
```

Response constructors also accept `davidson_backend="cpu"` or
`davidson_backend="gpu"`. The GPU option requires a GPU4PySCF reference
object and a working CUDA runtime. The repository pytest configuration forces
tests to use the CPU backend for reproducible local and GitHub Actions runs.

## Response methods

All three solvers accept a converged PySCF ROKS or UKS reference. Their
`kernel(nstates=...)` method returns excitation energies in eV and the
corresponding eigenvectors.

### XTDA: spin-conserving response

```python
from XTDDFT_dev.XTDDFT.xtda import XTDA

td = XTDA(mf, method=0, davidson=True, davidson_backend="cpu")
energies_ev, vectors = td.kernel(nstates=3)
td.analyse()
```

XTDA uses `method=0` and supports Davidson or dense diagonalization for
spin-conserving excitations ($S_f=S_i$). `save=True` or
`td.save_results("xtda.npz")` writes compressed excitation results.

### SF_TDA_up: spin-flip-up response

```python
from XTDDFT_dev.XTDDFT.sf_tda_up import SF_TDA_up

td = SF_TDA_up(mf, method=1, collinear_samples=20)
energies_ev, vectors = td.kernel(nstates=3)
```

`method=0` selects ALDA0 and `method=1` selects the multicollinear
approximation (MCOL). These are spin-flip-up excitations
($S_f=S_i+1$).

### XSF_TDA_down: spin-flip-down response

```python
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down

td = XSF_TDA_down(mf, method=2, SA=3, collinear_samples=60)
energies_ev, vectors = td.kernel(nstates=3)
```

`method=0`, `1`, and `2` select ALDA0, MCOL, and the collinear
approximation (COL). `SA` controls the spin-adaptation correction: 0 keeps
the SF-TDA block, while 1--3 add progressively more $\Delta A$ terms.
These are spin-flip-down excitations ($S_f=S_i-1$).

## Transition densities, dipoles, and NTOs

XTDA uses physical state labels: state 0 is the reference ground state and
excited states are 1, 2, ...:

```python
gamma_10 = td.transition_density_matrix(state_f=1, state_i=0)
mu_0n = td.transition_dipoles_ground()
mu_mn = td.transition_dipole_matrix()
singular_values, holes, particles = td.nto(state_f=1, state_i=0)
blocks = td.block_nto(state=0)
```

`transition_density_matrix`, `transition_dipole_matrix`, `nto`, and
`block_nto` are also available on the SF-TDA and XSF-TDA solvers. SF-TDA
uses zero-based excited-root indices and accepts `None` for the reference;
its spin-independent reference-to-SF transition density is zero. XSF-TDA
uses physical excited-state labels 1, 2, ...; its spin-free transition-density
representation does not include state 0. Returned matrices use restricted
C|O|V MO order or unrestricted alpha|beta spin-MO order, as appropriate.

## Excited-state gradients

Analytic nuclear gradients are available for molecular CPU calculations:

```python
gradient = td.nuc_grad_method(state=1).kernel()
```

The state number is one-based and refers to an excited state. Implemented
routes are XTDA `method=0`, SF-TDA `method=1`, and XSF-TDA `method=1` or
`2`. Periodic and GPU references are not supported by the analytic gradient
driver.

## Non-adiabatic couplings

The finite-difference NAC driver is implemented for molecular CPU ROKS + XTDA
(`method=0`). State 0 is the ground state, so ground/excited and
excited/excited pairs can be requested. All atoms are always included:

```python
td = XTDA(mf, method=0, davidson_backend="cpu")
td.kernel(nstates=3)

nac = td.nac_method(
    pairs=[(0, 1), (1, 2)],
    step=1.0e-4,             # displacement in Angstrom
)
vectors = nac.kernel()
coupling_01 = vectors[(0, 1)]  # shape: (mol.natm, 3), unit: bohr^-1
```

The driver aligns molecular orbitals and excited-state amplitudes at displaced
geometries before applying the central finite difference. Requested physical
state pairs must be covered by the roots from `td.kernel()`.

## Periodic calculations

The response solvers support molecular systems and the Gamma-point periodic
paths implemented in the corresponding class. NACs and analytic excited-state
gradients currently support molecular systems only.

## Minimal complete example

```python
from pyscf import dft, gto
from XTDDFT_dev.XTDDFT.xtda import XTDA
from XTDDFT_dev.utils.backend import set_backend

set_backend("cpu")
mol = gto.M(
    atom="C 0 0 0; O 0 0 1.2; H 0 1.0 -0.5; H 0 -1.0 -0.5",
    basis="6-31g",
    spin=2,
)
mf = dft.ROKS(mol, xc="b3lyp")
mf.kernel()
td = XTDA(mf, method=0)
energies_ev, vectors = td.kernel(nstates=3, save=True, save_file="xtda.npz")
td.analyse()
```

## How to cite

When using XTDDFT, please cite

```bash

@article{li2010spin,
  title={Spin-adapted open-shell random phase approximation and time-dependent density functional theory. I. Theory},
  author={Li, Zhendong and Liu, Wenjian},
  journal={The Journal of chemical physics},
  volume={133},
  number={6},
  year={2010},
  publisher={AIP Publishing}
}

@article{zhao2026spin,
  title={Spin-adapted open-shell time-dependent density functional theory: towards a simple and accurate method for spin-flip-down excitations},
  author={Zhao, Hewang and Li, Zhendong},
  journal={Molecular Physics},
  pages={e2631735},
  year={2026},
  publisher={Taylor \& Francis}
}

@article{li2013combining,
  title={Combining spin-adapted open-shell TD-DFT with spin--orbit coupling},
  author={Li, Zhendong and Suo, Bingbing and Zhang, Yong and Xiao, Yunlong and Liu, Wenjian},
  journal={Molecular Physics},
  volume={111},
  number={24},
  pages={3741--3755},
  year={2013},
  publisher={Taylor \& Francis}
}
```

## License

[Apache License 2.0](LICENSE)
