# XTDDFT Analytic Gradient Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the six existing CPU molecular analytic-gradient implementations available through `td.nuc_grad_method(state=1)` with pytest coverage and a runnable molecular example.

**Architecture:** Add a lazy dispatcher under `XTDDFT/grad`, delegate to it from `XTDDFT_base`, and retain one formula module per reference/response combination. Normalize the six wrappers and their access to current solver state without rewriting the mathematical kernels or mutating the original mean-field object.

**Tech Stack:** Python 3.9+, NumPy, SciPy, PySCF, pytest, setuptools via `pyproject.toml`.

**Spec:** `docs/superpowers/specs/2026-09-02-gradient-integration-design.md`

## Global Constraints

- Public state numbers are one-based integers.
- Analytic gradients are molecular and CPU-only in this release.
- Supported routes are XTDA method 0, SF_TDA_up method 1, and XSF_TDA_down methods 1 and 2, each with ROKS or UKS references.
- Spin-flip method 0, GPU, PBC, and non-ROKS/UKS references fail explicitly; there is no silent fallback or backend conversion.
- `Gradients.kernel()` returns a NumPy array in Hartree/Bohr.
- Gradient construction and execution do not mutate SCF orbitals, occupations, energies, or TD vectors.
- `finite_difference.py` is not installed or exposed as public API.
- Preserve the user's existing uncommitted change in `XTDDFT/xtda.py` and do not revert unrelated files.

---

### Task 1: Public Dispatcher and Package Registration

**Files:**
- Create: `tests/test_gradient_dispatch.py`
- Create: `XTDDFT/grad/__init__.py`
- Modify: `XTDDFT/base.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: completed `XTDA`, `SF_TDA_up`, or `XSF_TDA_down` objects exposing `mf`, `method`, `e`, and `v`.
- Produces: `XTDDFT.grad.nuc_grad_method(td, state=1)` and `XTDDFT_base.nuc_grad_method(state=1)` returning the selected module's `Gradients(td, state=state)` object.

- [ ] **Step 1: Write dispatcher tests with lightweight solver instances**

Use `object.__new__` so the tests exercise exact solver types without running SCF. Replace the lazy importer with a recorder and parameterize the six module routes:

```python
import types
import numpy as np
import pytest

from XTDDFT_dev.XTDDFT import grad
from XTDDFT_dev.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT_dev.XTDDFT.xtda import XTDA


class ROKS:
    cell = None
    mo_coeff = np.zeros((4, 4))


class UKS:
    cell = None
    mo_coeff = np.zeros((2, 4, 4))


def solved(solver_cls, mf_cls, method):
    td = object.__new__(solver_cls)
    td.mf = mf_cls()
    td.method = method
    td.e = np.zeros(2)
    td.v = np.zeros((5, 2))
    return td


@pytest.mark.parametrize(
    "solver_cls,mf_cls,method,module_tail",
    [
        (XTDA, ROKS, 0, "gradient_roks_sc"),
        (XTDA, UKS, 0, "gradient_uks_sc"),
        (SF_TDA_up, ROKS, 1, "gradient_roks_sfu"),
        (SF_TDA_up, UKS, 1, "gradient_uks_sfu"),
        (XSF_TDA_down, ROKS, 1, "gradient_roks_sfd"),
        (XSF_TDA_down, UKS, 2, "gradient_uks_sfd"),
    ],
)
def test_dispatches_to_expected_module(monkeypatch, solver_cls, mf_cls, method, module_tail):
    calls = []

    class FakeGradients:
        def __init__(self, td, state):
            self.base = td
            self.state = state

    monkeypatch.setattr(
        grad,
        "import_module",
        lambda name: calls.append(name) or types.SimpleNamespace(Gradients=FakeGradients),
    )
    td = solved(solver_cls, mf_cls, method)
    result = grad.nuc_grad_method(td, state=2)
    assert calls == [f"XTDDFT_dev.XTDDFT.grad.{module_tail}"]
    assert result.base is td
    assert result.state == 2
```

Add tests for `state=0`, `state=True`, missing `e`/`v`, state 3 with only two roots, method 0 spin-flip, unknown solver, an MF class whose module begins with `gpu4pyscf`, a fake with non-`None` `cell`, and an RHF-like reference. Assert the exception type and that each message names the remedy or available roots. Add one test calling `td.nuc_grad_method()` to prove the base-class delegation uses the dispatcher.

- [ ] **Step 2: Run the dispatcher test and verify it fails**

Run:

```bash
conda run -n xtddft python -m pytest tests/test_gradient_dispatch.py -q
```

Expected: collection fails because `XTDDFT.grad` is not yet a package and `XTDDFT_base` has no `nuc_grad_method`.

- [ ] **Step 3: Implement the lazy dispatcher**

Create `XTDDFT/grad/__init__.py` with no eager imports of formula modules. Use this structure:

```python
from importlib import import_module
from numbers import Integral

from ..base import _is_gpu_mf, _is_pbc_mf


def _solver_family(td):
    from ..sf_tda_up import SF_TDA_up
    from ..xsf_tda_down import XSF_TDA_down
    from ..xtda import XTDA

    if isinstance(td, XTDA) and td.method == 0:
        return "sc"
    if isinstance(td, SF_TDA_up) and td.method == 1:
        return "sfu"
    if isinstance(td, XSF_TDA_down) and td.method in (1, 2):
        return "sfd"
    if isinstance(td, (SF_TDA_up, XSF_TDA_down)) and td.method == 0:
        raise NotImplementedError("Spin-flip method=0 analytic gradients are not implemented.")
    raise NotImplementedError(
        f"Analytic gradients are not implemented for {type(td).__name__} method={getattr(td, 'method', None)!r}."
    )


def _reference_family(mf):
    names = {cls.__name__.upper() for cls in type(mf).__mro__}
    if "ROKS" in names:
        return "roks"
    if "UKS" in names:
        return "uks"
    raise NotImplementedError("Analytic gradients require a molecular ROKS or UKS reference.")


def nuc_grad_method(td, state=1):
    if isinstance(state, bool) or not isinstance(state, Integral) or state < 1:
        raise ValueError("state must be a one-based positive integer.")
    if _is_gpu_mf(td.mf):
        raise NotImplementedError("GPU analytic gradients are not implemented; run the calculation with a CPU mean-field object.")
    if _is_pbc_mf(td.mf):
        raise NotImplementedError("PBC analytic gradients are not implemented; use a molecular ROKS or UKS reference.")
    if getattr(td, "e", None) is None or getattr(td, "v", None) is None:
        raise RuntimeError("Run td.kernel() before requesting an analytic gradient.")
    nroots = int(td.v.shape[1])
    if state > nroots:
        raise ValueError(f"state={state} requested, but only {nroots} roots are available.")
    family = _solver_family(td)
    reference = _reference_family(td.mf)
    module = import_module(f"{__name__}.gradient_{reference}_{family}")
    return module.Gradients(td, state=int(state))


__all__ = ["nuc_grad_method"]
```

If real PySCF subclasses do not expose `ROKS`/`UKS` in their MRO names, extend `_reference_family` using PySCF's concrete base classes and add the corresponding test; do not fall back to shape-only acceptance of arbitrary references.

- [ ] **Step 4: Add the solver convenience method and package entry**

Add to `XTDDFT_base`:

```python
    def nuc_grad_method(self, state=1):
        from .grad import nuc_grad_method
        return nuc_grad_method(self, state=state)
```

Append `"XTDDFT_dev.XTDDFT.grad"` to the explicit `[tool.setuptools] packages` list. Do not add `finite_difference.py` to any export.

- [ ] **Step 5: Run the focused test and commit**

Run:

```bash
conda run -n xtddft python -m pytest tests/test_gradient_dispatch.py -q
```

Expected: all dispatcher tests pass without importing a formula module.

Commit:

```bash
git add tests/test_gradient_dispatch.py XTDDFT/grad/__init__.py XTDDFT/base.py pyproject.toml
git commit -m "feat: add analytic gradient dispatcher"
```

---

### Task 2: Normalize Spin-Conserving Gradient Modules

**Files:**
- Create: `tests/test_gradient_module_contract.py`
- Modify: `tests/test_xtda_tensor_basis.py`
- Modify: `XTDDFT/xtda.py`
- Modify: `XTDDFT/grad/gradient_roks_sc.py`
- Modify: `XTDDFT/grad/gradient_uks_sc.py`

**Interfaces:**
- Consumes: solved `XTDA` with `method=0`, `state`, `mf`, normalized solver orbital fields, `order`, and `use_delta_a`.
- Produces: `XTDA._spin_orbital_vectors()`, `gradient_roks_sc.Gradients`, and `gradient_uks_sc.Gradients`, with each gradient class exposing `grad_elec(atmlst=None)` and `kernel(atmlst=None)`.

- [ ] **Step 1: Write module contract tests**

In `tests/test_gradient_module_contract.py`, import each module in a subprocess after setting sentinel values for `OMP_NUM_THREADS`, `OMP_DYNAMIC`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`. Assert the values do not change and each module exposes `Gradients`. Parse the source with `ast` and assert there is no `if __name__ == "__main__"` block and no import of `finite_difference`, legacy `XTDA`, `UTDA`, `SF_TDA`, `XSF_TDA`, or top-level `utils`.

Add a wrapper test with a minimal solved-TD fake and monkeypatch module-level `grad_elec` plus the inherited `grad_nuc`. Assert:

```python
gradient = module.Gradients(td, state=2)
actual = gradient.kernel(atmlst=[0])
np.testing.assert_allclose(actual, electronic + nuclear)
assert gradient.base is td
assert gradient.state == 2
assert gradient.unit == "au"
```

Extend `tests/test_xtda_tensor_basis.py` with an exact transform round-trip. For
a restricted solver holding spin-tensor vectors, assert
`_spin_orbital_vectors()` equals the original `CVa|OVa|COb|CVb` matrix used to
construct them. For an unrestricted solver, assert the method returns `v`
unchanged.

- [ ] **Step 2: Run contract tests and verify they fail**

Run:

```bash
conda run -n xtddft python -m pytest tests/test_gradient_module_contract.py -q
```

Expected: imports fail on legacy absolute imports and the environment/main-guard assertions fail.

- [ ] **Step 3: Clean both spin-conserving modules without changing formula algebra**

For both modules:

- Remove the shebang, `os.environ` assignments, unused demo imports, `finite_difference` imports, and the complete executable block.
- Replace every formula access `td.base._scf` with `td.base.mf`.
- Rename `SC_gradient` to `Gradients`.
- Initialize `base`, `mol`, `state`, `de`, `atmlst`, `verbose`, `stdout`, `max_memory`, and `unit="au"` from `td`/`td.mf`; do not assign `td._scf` or mutate `td.mf`.
- Remove the implicit `td.kernel(...)` branch from `grad_elec`; dispatcher validation owns that error.
- In the ROKS module replace `X = td.base.X` with `use_delta_a = bool(td.base.use_delta_a)` and use that name at both correction branches.
- In the ROKS module obtain the state vector from
  `td.base._spin_orbital_vectors()[:, td.state - 1]`; its formula expects
  `CVa|OVa|COb|CVb`, whereas the maintained restricted solver exposes
  `CV(0)|CO(0)|OV(0)|CV(1)` in `td.v`. The UKS formula continues to consume
  `td.v` directly because UKS results are already spin-orbital ordered.
- Refactor `_RO2U` so it owns `mf.copy()` and writes doubled `mo_coeff`, `mo_occ`, and `mo_energy` only to that copy. The original `td.mf` must never be assigned through the adapter.

Add this method to `XTDA` next to the existing vector-layout helpers:

```python
    def _spin_orbital_vectors(self):
        vectors = np.asarray(_asnumpy(self.v))
        if self.type_u:
            return vectors
        transform = _so2st_matrix(self.nc, self.no, self.nv)
        return transform.T @ vectors
```

The round-trip test must establish that the existing transform is orthogonal
for the active dimensions before the gradient relies on its transpose.

Use this wrapper shape in each module:

```python
class Gradients(<existing PySCF gradient base>):
    def __init__(self, td, state=1):
        self.base = td
        self.mol = td.mol
        self.state = state
        self.de = None
        self.atmlst = None
        self.verbose = getattr(td.mf, "verbose", logger.INFO)
        self.stdout = getattr(td.mf, "stdout", None)
        self.max_memory = getattr(td.mf, "max_memory", 2000)
        self.unit = "au"

    def grad_elec(self, atmlst=None):
        return grad_elec(self, atmlst=atmlst)

    def kernel(self, atmlst=None):
        self.atmlst = self.atmlst if atmlst is None else atmlst
        self.de = self.grad_elec(self.atmlst) + self.grad_nuc(atmlst=self.atmlst)
        self._finalize()
        return self.de
```

Preserve the existing PySCF base (`rohf_grad.Gradients` or `uhf_grad.Gradients`) and module-specific CPHF configuration.

- [ ] **Step 4: Run contract and dispatcher tests**

Run:

```bash
conda run -n xtddft python -m pytest tests/test_gradient_module_contract.py tests/test_gradient_dispatch.py tests/test_xtda_tensor_basis.py -q
```

Expected: both pass, and importing either SC module leaves process environment unchanged.

- [ ] **Step 5: Commit**

```bash
git add tests/test_gradient_module_contract.py tests/test_xtda_tensor_basis.py XTDDFT/grad/gradient_roks_sc.py XTDDFT/grad/gradient_uks_sc.py
git add -p XTDDFT/xtda.py
git commit -m "refactor: integrate spin-conserving gradients"
```

When interactively staging `XTDDFT/xtda.py`, include only the new
`_spin_orbital_vectors` method and leave the user's pre-existing `_order_pyscf2my`
hunk unstaged.

---

### Task 3: Normalize Spin-Flip-Up Gradient Modules and Solver Sampling

**Files:**
- Modify: `tests/test_gradient_module_contract.py`
- Create: `tests/test_sf_tda_up_gradient_config.py`
- Modify: `XTDDFT/sf_tda_up.py`
- Modify: `XTDDFT/grad/gradient_roks_sfu.py`
- Modify: `XTDDFT/grad/gradient_uks_sfu.py`

**Interfaces:**
- Consumes: solved `SF_TDA_up(method=1)` and the solver's `collinear_samples`.
- Produces: `gradient_roks_sfu.Gradients`, `gradient_uks_sfu.Gradients`, and a single `SF_TDA_up.collinear_samples` setting shared by energy and gradient response calculations.

- [ ] **Step 1: Extend failing contract/configuration tests**

Parameterize the Task 2 module tests over the SFU modules. In `tests/test_sf_tda_up_gradient_config.py`, monkeypatch the response builder and assert that a solver created with `collinear_samples=24` passes exactly `24` to both dense MCOL construction and Davidson response construction. Assert non-integer or zero sample counts raise `ValueError` for method 1.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
conda run -n xtddft python -m pytest tests/test_gradient_module_contract.py tests/test_sf_tda_up_gradient_config.py -q
```

Expected: legacy imports/environment writes fail, and `SF_TDA_up` does not yet store a unified sampling value.

- [ ] **Step 3: Make collinear sampling solver-owned**

Change the constructor to accept `collinear_samples=50`, validate it for `method=1`, and assign `self.collinear_samples`. Replace the hard-coded `50` in Davidson response creation with `self.collinear_samples`, and make dense MCOL construction use the same stored value when called from `get_Amat()`.

```python
if method == 1 and (
    isinstance(collinear_samples, bool)
    or not isinstance(collinear_samples, Integral)
    or collinear_samples < 1
):
    raise ValueError("collinear_samples must be a positive integer for method=1.")
self.collinear_samples = int(collinear_samples)
```

Import `Integral` from `numbers`. Do not change ALDA0 response algebra.

- [ ] **Step 4: Clean both SFU modules**

Apply the same environment/import/main-block/wrapper cleanup from Task 2. Replace all `td.base._scf` uses with `td.base.mf`; rename `SFU_gradient` to `Gradients`; remove the wrapper's independent `method` and `collinear_samples` choices. Formula code must read `td.base.method` and `td.base.collinear_samples` so the energy and gradient share settings.

For ROKS `_RO2U`, create a private SCF copy before installing the solver's doubled spin-orbital arrays. Do not write `mo_occ` or `mo_coeff` onto `td.mf`.

- [ ] **Step 5: Run focused tests and commit**

```bash
conda run -n xtddft python -m pytest tests/test_gradient_dispatch.py tests/test_gradient_module_contract.py tests/test_sf_tda_up_gradient_config.py -q
```

Expected: all pass.

```bash
git add tests/test_gradient_module_contract.py tests/test_sf_tda_up_gradient_config.py XTDDFT/sf_tda_up.py XTDDFT/grad/gradient_roks_sfu.py XTDDFT/grad/gradient_uks_sfu.py
git commit -m "refactor: integrate spin-flip-up gradients"
```

---

### Task 4: Normalize Spin-Flip-Down Gradient Modules

**Files:**
- Modify: `tests/test_gradient_module_contract.py`
- Modify: `XTDDFT/grad/gradient_roks_sfd.py`
- Modify: `XTDDFT/grad/gradient_uks_sfd.py`
- Delete: `XTDDFT/grad/finite_difference.py`

**Interfaces:**
- Consumes: solved `XSF_TDA_down(method=1|2)`, including `SA`, `re`, `vects` when applicable, and `collinear_samples`.
- Produces: `gradient_roks_sfd.Gradients` and `gradient_uks_sfd.Gradients` with no duplicated method configuration.

- [ ] **Step 1: Extend module contract tests to SFD**

Add both SFD modules to the import, side-effect, no-demo, and wrapper parameter sets. Add a dispatcher test proving both method 1 and method 2 select the same reference-specific SFD module and that method 0 is rejected before import.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
conda run -n xtddft python -m pytest tests/test_gradient_dispatch.py tests/test_gradient_module_contract.py -q
```

Expected: both SFD imports fail on legacy paths and their wrappers violate the common contract.

- [ ] **Step 3: Clean both SFD modules**

Apply the same cleanup rules from Tasks 2 and 3. Rename `SFD_gradient` to `Gradients`, remove the independent `method` argument, and read `td.base.method` and `td.base.collinear_samples`. Replace the stale `self.td.kernel(states=...)`/`self.base.kernel(states=...)` paths by dispatcher-owned precondition checks.

For the ROKS adapter, copy the mean-field object before installing doubled arrays. Preserve use of the solver's `re` and `vects` state, but add a targeted `RuntimeError` if the selected SA path requires `vects` and the completed solver did not populate it. Do not synthesize or reorder TD roots in the gradient layer.

After every formula module has dropped its legacy finite-difference import,
delete `XTDDFT/grad/finite_difference.py`. Add a contract assertion that the
file is absent; numerical finite differences belong exclusively to
`tests/test_analytic_gradients.py`.

- [ ] **Step 4: Run focused tests and commit**

```bash
conda run -n xtddft python -m pytest tests/test_gradient_dispatch.py tests/test_gradient_module_contract.py tests/test_sf_tda_up_gradient_config.py -q
```

Expected: all pass.

```bash
git add tests/test_gradient_dispatch.py tests/test_gradient_module_contract.py XTDDFT/grad/gradient_roks_sfd.py XTDDFT/grad/gradient_uks_sfd.py
git commit -m "refactor: integrate spin-flip-down gradients"
```

---

### Task 5: Molecular Smoke and Numerical Gradient Tests

**Files:**
- Create: `tests/test_analytic_gradients.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: all six dispatcher routes and total excited-state energy `mf.e_tot + td.e[state - 1]`.
- Produces: pytest smoke coverage plus slow central-finite-difference regression coverage for one Cartesian component per route.

- [ ] **Step 1: Register the slow marker and write reusable fixtures**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
  "slow: computational molecular regression tests",
]
```

In `tests/test_analytic_gradients.py`, use `pytest.importorskip("pyscf")`, force the package backend to CPU, and define one small open-shell molecule fixture (start with bent triplet CH2, `spin=2`, `sto-3g`, no symmetry). Build ROKS and UKS fixtures with the same XC functional, convergence thresholds, and grid level. Copy `mo_coeff`, `mo_occ`, `mo_energy`, and `td.v` before every gradient run.

- [ ] **Step 2: Write the six-route smoke test and verify initial failures**

Parameterize factories for `XTDA(method=0)`, `SF_TDA_up(method=1)`, and `XSF_TDA_down(method=1, SA=3 for ROKS and SA=0 for UKS)` across ROKS and UKS. Run enough roots for `state=1`, then assert:

```python
de = td.nuc_grad_method(state=1).kernel()
assert isinstance(de, np.ndarray)
assert de.shape == (mol.natm, 3)
assert np.isfinite(de).all()
np.testing.assert_allclose(de.sum(axis=0), 0.0, atol=2e-5)
np.testing.assert_array_equal(mf.mo_coeff, mo_coeff_before)
np.testing.assert_array_equal(mf.mo_occ, mo_occ_before)
np.testing.assert_array_equal(mf.mo_energy, mo_energy_before)
np.testing.assert_array_equal(td.v, vectors_before)
```

Run:

```bash
conda run -n xtddft python -m pytest tests/test_analytic_gradients.py -m "not slow" -x -q
```

Expected: the first incompatible formula assumption fails with a concrete traceback.

- [ ] **Step 3: Reconcile formula inputs route by route**

Apply only the following compatibility map, then rerun each single pytest
parameter before moving to the next route:

| Formula assumption | Maintained solver source |
| --- | --- |
| `td.base._scf` | `td.base.mf` |
| ROKS SC spin-orbital vector | `td.base._spin_orbital_vectors()[:, td.state - 1]` |
| UKS SC spin-orbital vector | `td.v[:, td.state - 1]` |
| old `td.base.X` flag | `td.base.use_delta_a` |
| SFU sampling chosen by gradient | `td.base.collinear_samples` |
| SFD method chosen by gradient | `td.base.method` |
| ROKS SFD removed-OO expansion | the completed solver's `re` and `vects` |

Do not apply another orbital permutation to UKS SC vectors: the maintained
solver already stores `CVa|OVa|COb|CVb`. Do not change tensor coefficients to
make a regression pass. If a failure remains after this map, stop that route,
capture the first mismatching intermediate, and diagnose it against the
corresponding solver block definition before editing algebra. After each route
passes, run all fast gradient tests.

- [ ] **Step 4: Add one-coordinate central finite differences**

Mark the numerical parameter set with `@pytest.mark.slow`. For each route, displace a symmetry-breaking Cartesian coordinate by `±1e-3` Bohr, rebuild SCF and TD objects with identical settings, and compute:

```python
numeric = (energy_plus - energy_minus) / (2.0e-3)
analytic = td.nuc_grad_method(state=1).kernel()[atom_index, axis]
assert analytic == pytest.approx(numeric, abs=5e-4, rel=5e-3)
```

If root order changes, select the displaced root by maximum overlap with the undisplaced transition vector and document that selection in the helper. Never loosen tolerance before checking the finite-difference step, grid, SCF convergence, and root identity.

- [ ] **Step 5: Run fast and slow gradient tests**

```bash
conda run -n xtddft python -m pytest tests/test_gradient_dispatch.py tests/test_gradient_module_contract.py tests/test_sf_tda_up_gradient_config.py tests/test_analytic_gradients.py -m "not slow" -q
conda run -n xtddft python -m pytest tests/test_analytic_gradients.py -m slow -q
```

Expected: all tests pass. Record the final molecule, functional, grid, displacement, and observed maximum analytic/numerical error in the commit message body or task notes.

- [ ] **Step 6: Commit**

```bash
git add tests/test_analytic_gradients.py pyproject.toml XTDDFT/grad XTDDFT/sf_tda_up.py XTDDFT/xsf_tda_down.py
git commit -m "test: validate analytic gradients numerically"
```

Before staging `XTDDFT/xtda.py`, inspect `git diff` and stage only integration edits, preserving the user's pre-existing ordering change as a separate unstaged hunk unless the user explicitly asks to include it.

---

### Task 6: Runnable Example, README, and Full Verification

**Files:**
- Create: `examples/molecule/run_excited_state_gradient.py`
- Create: `tests/test_excited_state_gradient_script.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: installed `XTDDFT_dev`, PySCF ROKS, and `XTDA`.
- Produces: one import-safe executable example and user documentation for support, state numbering, and units.

- [ ] **Step 1: Write the example contract test**

Follow existing script tests. Read the source and assert it imports `XTDA`, calls `td.kernel(nstates=`, calls `td.nuc_grad_method(state=1)`, calls `gradient.kernel()`, mentions `Hartree/Bohr`, and guards execution with `if __name__ == "__main__":`. Import it through `importlib.util.spec_from_file_location` while monkeypatching PySCF execution and assert importing performs no calculation.

- [ ] **Step 2: Run the script test and verify failure**

```bash
conda run -n xtddft python -m pytest tests/test_excited_state_gradient_script.py -q
```

Expected: fail because the example does not exist.

- [ ] **Step 3: Add a concrete molecular example**

Implement `main()` using a small triplet molecule, CPU backend, ROKS, and `XTDA(method=0)`. The essential call sequence is:

```python
from pyscf import dft, gto

from XTDDFT_dev.XTDDFT.xtda import XTDA
from XTDDFT_dev.utils.backend import set_backend


def main():
    set_backend("cpu")
    mol = gto.M(
        atom="""
        C  0.000000  0.000000  0.000000
        H  0.000000  1.050000  0.000000
        H  0.000000 -1.050000  0.000000
        """,
        basis="sto-3g",
        spin=2,
        charge=0,
        unit="Angstrom",
    )
    mf = dft.ROKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-10
    mf.grids.level = 3
    mf.kernel()

    td = XTDA(mf, method=0, davidson=True)
    td.kernel(nstates=3)
    gradient = td.nuc_grad_method(state=1)
    de = gradient.kernel()
    print("First excited-state gradient (Hartree/Bohr):")
    print(de)


if __name__ == "__main__":
    main()
```

Use a geometry confirmed by Task 5 rather than retaining the provisional CH2 geometry if it is not numerically stable.

- [ ] **Step 4: Update README**

Add an “Analytic gradients” section containing the same four-line public call, a compact supported-method table, the one-based state convention, Hartree/Bohr units, and the CPU molecular limitation. Link to `examples/molecule/run_excited_state_gradient.py`.

- [ ] **Step 5: Run the example contract and the example itself**

```bash
conda run -n xtddft python -m pytest tests/test_excited_state_gradient_script.py -q
conda run -n xtddft python examples/molecule/run_excited_state_gradient.py
```

Expected: the test passes and the example prints a finite `(natm, 3)` gradient.

- [ ] **Step 6: Run complete verification**

```bash
conda run -n xtddft python -m pytest tests -m "not slow" -q
conda run -n xtddft python -m pytest tests/test_analytic_gradients.py -m slow -q
wheel_dir=$(mktemp -d)
conda run -n xtddft python -m pip wheel . --no-deps --wheel-dir "$wheel_dir"
conda run -n xtddft python -m zipfile -l "$wheel_dir"/*.whl
```

Expected: all pytest suites pass and the wheel builds successfully. Inspect the
wheel listing and confirm it contains `XTDDFT_dev/XTDDFT/grad/__init__.py` and
all six analytic modules, but not `finite_difference.py`.

- [ ] **Step 7: Review the final diff and commit**

```bash
git diff --check
git status --short
git diff -- README.md examples/molecule/run_excited_state_gradient.py tests/test_excited_state_gradient_script.py
git add README.md examples/molecule/run_excited_state_gradient.py tests/test_excited_state_gradient_script.py
git commit -m "docs: add analytic gradient example"
```

Confirm that `.vscode/`, unrelated generated files, and the user's pre-existing `XTDDFT/xtda.py` hunk remain uncommitted unless explicitly authorized.
