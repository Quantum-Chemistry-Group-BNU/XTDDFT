"""Legacy grid-projection symmetry helpers.

The main XTDDFT symmetry path uses AO permutation backends.  This module keeps
the older numerical grid projection available for debugging loose geometries,
but it is intentionally outside the core package because it is much slower and
can allocate very large AO-by-grid arrays for large supercells.
"""

from __future__ import annotations

import numpy as np


def build_ao_operation_matrices(
    mol,
    geometry,
    grid_level: int = 5,
    grid_coords=None,
    grid_weights=None,
    grid_block_size: int = 20000,
):
    """Build AO representation matrices by numerical integration on grids."""

    if grid_coords is None or grid_weights is None:
        try:
            from pyscf.dft import gen_grid
        except ImportError as err:
            raise ImportError("PySCF is required to build AO operation matrices") from err

        grids = gen_grid.Grids(mol)
        grids.level = int(grid_level)
        grids.build()
        coords = np.asarray(grids.coords)
        weights = np.asarray(grids.weights)
    else:
        coords = np.asarray(grid_coords)
        weights = np.asarray(grid_weights)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"grid_coords must have shape (ngrids, 3); got {coords.shape}")
        if weights.ndim != 1 or weights.shape[0] != coords.shape[0]:
            raise ValueError(
                "grid_weights must be a 1D array with the same length as grid_coords; "
                f"got weights={weights.shape}, coords={coords.shape}"
            )

    overlap = _overlap_matrix(mol)
    ops = []
    for rotation in geometry.cartesian_operations:
        projected_overlap = np.zeros_like(overlap, dtype=np.result_type(overlap, complex))
        for p0 in range(0, coords.shape[0], int(grid_block_size)):
            p1 = min(p0 + int(grid_block_size), coords.shape[0])
            coord_block = coords[p0:p1]
            weight_block = weights[p0:p1]
            inverse_coords = (coord_block - geometry.origin) @ rotation + geometry.origin
            ao = _eval_gto_values(mol, coord_block)
            ao_rot = _eval_gto_values(mol, inverse_coords)
            projected_overlap += ao.conj().T @ (weight_block[:, None] * ao_rot)
        ops.append(np.linalg.solve(overlap, projected_overlap))
    return tuple(ops)


def _eval_gto_values(system, coords) -> np.ndarray:
    if hasattr(system, "pbc_eval_gto"):
        values = system.pbc_eval_gto("GTOval", coords)
    else:
        values = system.eval_gto("GTOval", coords)
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    return values


def _overlap_matrix(system) -> np.ndarray:
    if hasattr(system, "pbc_intor"):
        overlap = system.pbc_intor("int1e_ovlp", hermi=1)
    else:
        overlap = system.intor_symmetric("int1e_ovlp")
    overlap = np.asarray(overlap)
    if overlap.ndim == 3 and overlap.shape[0] == 1:
        overlap = overlap[0]
    return overlap
