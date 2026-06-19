# Excited-State Symmetry Analysis Design

## Goal

Add an independent symmetry-analysis module that assigns point-group irreducible-representation weights to the high-spin reference state and to XTDA, XSF-TDA-down, and SF-TDA-up excited states by projecting their wavefunction or excitation-vector representations under molecular point-group operations generated with libmsym.

## Scope

The first implementation targets:

- Molecular calculations.
- Gamma-point point-defect supercells treated as finite supercell clusters.
- XTDA, XSF_TDA_down, and SF_TDA_up result objects.
- Offline analysis from a reconstructed PySCF mean-field object plus saved `vectors` and energies.
- Ground-state/reference analysis for the single-determinant high-spin reference used by the excited-state method.

The supercell analysis is a finite point-group analysis of the supercell atomic geometry. It is not a full periodic space-group or k-point little-group analysis. This is appropriate for localized defect states in large Gamma-point supercells, but the report must state this limitation.

## Non-Goals

- Non-Gamma k-point calculations.
- Full periodic space-group symmetry.
- Automatic repair of symmetry-broken SCF solutions.
- Assignment from transition dipoles or dominant configurations alone.
- Exact symmetry labels when roots belonging to a degenerate irrep are not all present in the solved state manifold.

## Architecture

The module will use three layers.

1. `GeometrySymmetry`
   - Builds a finite point-group model from PySCF `mol` or Gamma-point `cell` atom coordinates.
   - Uses libmsym when available to detect symmetry operations and character tables.
   - Exposes operation matrices in Cartesian space and the point-group irreps.

2. Method adapters
   - Convert each TD object's stored eigenvector into one or more named excitation blocks.
   - Provide occupied and virtual MO indices for each block.
   - Supported adapters:
     - `SF_TDA_up`: `b -> a` spin-flip block with shape `nc * nv`.
     - `XSF_TDA_down`: `CV|CO|OV|OO` spin-adapted blocks, honoring `remove=True` OO compression.
     - `XTDA`: restricted `CV(0)|CO(0)|OV(0)|CV(1)` and unrestricted `CVa|OVa|COb|CVb`.

3. Projector
   - Builds AO representation matrices for each symmetry operation.
   - Transforms MO coefficients to obtain MO-space operation matrices.
   - Lifts MO-space transformations to each method's excitation-vector space.
   - Computes characters for individual roots and near-degenerate root subspaces.
   - Decomposes characters into irrep weights with the point group's character table.

4. Reference-state analyzer
   - Checks whether the spin density and total density are invariant under each point-group operation.
   - Computes the symmetry of the reference determinant from occupied-orbital subspace transformations.
   - For restricted open-shell references, separates doubly occupied closed-shell orbitals from singly occupied open-shell orbitals. Closed-shell pairs should contribute a totally symmetric factor when their occupied subspace is invariant. The high-spin open-shell determinant contributes `det(U_open(g))`.
   - For unrestricted references, computes `det(U_occ_alpha(g)) * det(U_occ_beta(g))` for the spin-orbital determinant.
   - Reports approximate weights rather than a hard label if the occupied subspace is not closed under the detected point group.

## Core Formula

For a root vector `X_s` and a symmetry operation `g`, the module computes:

```text
chi_s(g) = <X_s | U_exc(g) | X_s>
```

For a near-degenerate subspace `S`, it computes:

```text
S_mn(g) = <X_m | U_exc(g) | X_n>
chi_S(g) = trace(S(g))
```

For a single normalized root, projector weights are:

```text
w_Gamma = d_Gamma / |G| * sum_g conj(chi_Gamma(g)) * chi(g)
```

For a complete representation or degenerate subspace character, the irrep multiplicity is:

```text
a_Gamma = 1 / |G| * sum_classes n_c conj(chi_Gamma(c)) * chi(c)
```

The report labels a state as assigned only when one irrep weight is clearly dominant and the residual is small.

For the reference determinant, the character is computed from the determinant representation of the occupied spin-orbital subspace:

```text
chi_ref(g) = det(U_occ_alpha(g)) * det(U_occ_beta(g))
```

For a restricted open-shell high-spin reference, the closed-shell contribution is expected to be totally symmetric, and the open-shell contribution can be reported separately:

```text
chi_open(g) = det(U_open(g))
```

This is deliberately distinct from density symmetry. A reference density can be totally symmetric even when the open-shell high-spin determinant transforms as a non-totally-symmetric irrep.

## Numerical Checks

The module should report:

- Detected point group and tolerance.
- Whether the AO overlap is invariant under each operation.
- Whether the reference density is invariant under each operation.
- Ground/reference determinant irrep weights.
- Open-shell determinant irrep weights for restricted open-shell references.
- Root grouping used for near-degenerate subspaces.
- Irrep weights and residual weight for each root or subspace.

If the reference density or AO transformation is not symmetric within tolerance, the module should downgrade the assignment to "approximate".

## Public API

The first API should be object-based:

```python
from XTDDFT_dev.XTDDFT.symmetry import analyze_excited_state_symmetry

report = analyze_excited_state_symmetry(
    td_obj,
    energy_tol=1.0e-5,
    symmetry_tol=1.0e-5,
    point_group=None,
)
print(report.format_text())
```

The module should also expose lower-level helpers for scripts:

```python
groups = group_degenerate_roots(energies_ha, tol=1.0e-5)
adapter = make_method_adapter(td_obj)
```

Offline scripts can reconstruct `td_obj` the same way existing NTO scripts do, then call the object API.

## Fallback Behavior

If libmsym is unavailable, the module should fail with an actionable import error rather than silently falling back to PySCF's Abelian subgroup labels.

If libmsym cannot identify the intended high-symmetry group because the supercell or relaxed geometry is slightly distorted, the caller can pass a looser tolerance or a requested point group once that support is added. The first version can expose tolerance control and report the detected group.

## Testing Strategy

Use small synthetic tests for the adapter layer so tests do not require expensive TD calculations:

- Verify each adapter splits vectors into expected blocks.
- Verify degenerate grouping by energy tolerance.
- Verify projector decomposition on a manually supplied small character table and representation characters.
- Verify reference determinant characters using small manually supplied occupied-space operation matrices.

Add import-level tests that the public module can be imported without libmsym for type definitions, but calling symmetry detection raises a clear message if libmsym is missing.

Integration examples can be added under `experiment/molecule` after the core module is stable.
