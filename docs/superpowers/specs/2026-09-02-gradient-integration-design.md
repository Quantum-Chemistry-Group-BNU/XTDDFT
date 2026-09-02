# XTDDFT Analytic Gradient Integration Design

## Goal

Integrate the six existing molecular analytic-gradient implementations under
`XTDDFT/grad/` into the maintained `XTDDFT_dev` package while preserving the
project's current public API and module style. Users will obtain gradients from
an already-computed excited-state solver through a PySCF-style interface:

```python
td.kernel(nstates=3)
grad = td.nuc_grad_method(state=1)
de = grad.kernel()
```

The first release is deliberately limited to CPU molecular calculations. The
standalone finite-difference script is not part of the public package API;
finite differences are used only as a test oracle.

## Supported Matrix

The dispatcher exposes only combinations with an existing analytic path:

| Excited-state solver | Solver method | Reference | Gradient implementation |
| --- | --- | --- | --- |
| `XTDA` | `0` | ROKS | ROKS spin-conserving |
| `XTDA` | `0` | UKS | UKS spin-conserving |
| `SF_TDA_up` | `1` | ROKS | ROKS spin-flip-up |
| `SF_TDA_up` | `1` | UKS | UKS spin-flip-up |
| `XSF_TDA_down` | `1` or `2` | ROKS | ROKS spin-flip-down |
| `XSF_TDA_down` | `1` or `2` | UKS | UKS spin-flip-down |

Spin-flip `method=0`, GPU mean-field objects, periodic systems, and reference
types other than ROKS or UKS raise `NotImplementedError` with a concise remedy.
The implementation must not silently move data between CPU and GPU.

## Package Architecture

`XTDDFT/grad/__init__.py` is the single dispatch entry point. It exports
`nuc_grad_method(td, state=1)`, validates the solver and calculation state, and
imports only the selected implementation. Lazy imports keep the top-level
package lightweight and isolate optional PySCF gradient internals until a
gradient is requested.

`XTDDFT_base.nuc_grad_method(state=1)` delegates to this dispatcher. The six
formula modules remain separate:

- `gradient_roks_sc.py`
- `gradient_uks_sc.py`
- `gradient_roks_sfu.py`
- `gradient_uks_sfu.py`
- `gradient_roks_sfd.py`
- `gradient_uks_sfd.py`

Each module exports its implementation as `Gradients`. The mathematical
formulae stay local to the existing modules; this integration does not attempt
to merge their repeated code. Engineering cleanup includes package-relative
imports, removal of executable demonstration blocks, removal of module-level
thread environment changes, and correction of stale solver attribute and
keyword names.

`pyproject.toml` includes `XTDDFT_dev.XTDDFT.grad` in the explicit setuptools
package list. `XTDDFT/__init__.py` does not eagerly import the six formula
modules.

## Dispatch and Data Flow

The public `state` argument is a positive, one-based state number, matching the
existing formulae's `td.v[:, state - 1]` indexing. Before constructing a
gradient object, the dispatcher verifies that `td.kernel()` has completed and
that the stored eigenvector matrix contains the requested root. It does not
rerun the excited-state calculation implicitly because doing so could change
solver settings or root ordering.

The dispatcher identifies the response family from the concrete solver class
and the reference family from the original CPU mean-field object. A two-
dimensional `mo_coeff` selects the restricted-open-shell path; a two-spin,
three-dimensional `mo_coeff` selects the unrestricted path. The gradient object
inherits the solver's `method`, `collinear_samples`, molecular object,
eigenvectors, logging settings, and SCF reference instead of asking the user to
repeat them.

`Gradients.kernel(atmlst=None)` computes the electronic contribution for the
selected state, adds the SCF nuclear contribution, stores the result in `de`,
and returns a finite NumPy array with shape `(len(atmlst), 3)` or `(mol.natm,
3)`. The unit is Hartree/Bohr.

ROKS formulae need a two-spin view of restricted orbitals. That view is created
locally or within a restoration-safe context. Gradient construction and
execution must leave `td.mf.mo_coeff`, `td.mf.mo_occ`, and the stored TD
eigenvectors unchanged.

## Validation and Errors

The dispatcher rejects invalid requests before importing or running a formula:

- `state` is not an integer greater than or equal to one: `ValueError`.
- `td.kernel()` has not populated energies and vectors: `RuntimeError` with an
  instruction to run it first.
- The requested state exceeds the stored roots: `ValueError` containing the
  available root count.
- The calculation uses a GPU backend, PBC object, unsupported solver, reference,
  or method: `NotImplementedError` naming the unsupported combination.

Existing formula-level limitations, including unavailable functional
derivatives and NLC contributions, retain explicit `NotImplementedError`
behavior. No fallback to a different functional, method, or numerical gradient
is permitted.

## Tests

All tests use `pytest` and live in the existing `tests/` directory.

Fast interface tests cover the six dispatch routes, one-based state validation,
missing or insufficient roots, unsupported method combinations, GPU and PBC
rejection, lazy imports, packaging, and preservation of thread-related
environment variables during import. These tests may use small fakes and
monkeypatching so they run without SCF calculations.

Molecular smoke tests run each of the six implementation routes on a small,
stable open-shell system. They check shape, dtype, finite values, approximate
translational invariance, and preservation of the mean-field orbitals and TD
vectors. Numerical regression tests compare at least one Cartesian component
from every route against a central finite difference of
`mf.e_tot + td.e[state - 1]`. Expensive numerical cases receive a pytest slow
marker so the fast suite can run independently.

Verification consists of the gradient fast tests, the marked numerical tests,
and the repository's existing full pytest suite. Tolerances are stated
explicitly in each numerical test and chosen to accommodate the selected DFT
grid and finite-difference step without masking sign, state-index, or unit
errors.

## Example and Documentation

`examples/molecule/run_excited_state_gradient.py` provides a complete CPU
molecular example: construct an open-shell molecule, run ROKS or UKS, construct
one supported XTDDFT solver, compute enough excited states, request
`state=1`, run the gradient, and print the Hartree/Bohr result. The example has
an executable `main()` guard and performs no calculation when imported.

The README documents the supported matrix, the PySCF-style call, one-based
state convention, output unit, and the CPU/molecule-only limitation. It also
states that spin-flip `method=0`, GPU, and PBC gradients are not implemented in
this release.

## Out of Scope

- Publishing `finite_difference.py` as a user-facing module.
- GPU or PBC analytic gradients.
- Implicit CPU conversion of GPU calculations.
- Geometry-optimization scanners or nonadiabatic couplings.
- Algebraic consolidation or redesign of the six gradient formulae.
- Unrelated changes to solver behavior or the user's existing `xtda.py` work.
