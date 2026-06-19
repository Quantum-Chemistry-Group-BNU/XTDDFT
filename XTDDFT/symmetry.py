"""Point-group symmetry helpers for XTDDFT excited-state analyses.

This module is intentionally split into a tested, lightweight layer and a
future libmsym-backed geometry layer.  The lightweight layer covers method
vector adapters, root grouping, determinant characters, character projection,
and report formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CharacterTable:
    """Character table sampled over explicit symmetry operations."""

    group_name: str
    operation_labels: tuple[str, ...]
    irreps: Mapping[str, Sequence[complex]]
    class_counts: tuple[int, ...] | None = None


@dataclass(frozen=True)
class GeometrySymmetry:
    """Finite-geometry point-group information detected by libmsym."""

    point_group: str
    character_table: CharacterTable
    symmetry_operation_labels: tuple[str, ...]
    cartesian_operations: tuple[np.ndarray, ...]
    operation_class_indices: tuple[int, ...]
    origin: np.ndarray
    fractional_operations: tuple[np.ndarray, ...] | None = None
    fractional_translations: tuple[np.ndarray, ...] | None = None
    lattice_vectors: np.ndarray | None = None


@dataclass(frozen=True)
class MOOperationMatrices:
    """Spin-resolved MO representation matrices for symmetry operations."""

    alpha: tuple[np.ndarray, ...]
    beta: tuple[np.ndarray, ...]
    ao: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SubspaceOperationMatrices:
    """MO operation matrices for selected orbital subspaces only."""

    matrices: Mapping[tuple[str, str], tuple[np.ndarray, ...]]
    spaces: Mapping[tuple[str, str], tuple[int, ...]]


@dataclass(frozen=True)
class ActiveOrbitalSelection:
    """Orbital subspaces selected from important excitation amplitudes."""

    spaces: Mapping[tuple[str, str], tuple[int, ...]]
    analyzed_weight: float
    total_weight: float


@dataclass(frozen=True)
class ExcitationBlockSpec:
    """Method-specific meaning of one excitation-amplitude block."""

    name: str
    left_key: tuple[str, str]
    right_key: tuple[str, str]
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]


@dataclass(frozen=True)
class ExcitationBlock:
    """One named block of a method-specific excitation vector."""

    name: str
    amplitudes: np.ndarray


@dataclass(frozen=True)
class SymmetryAnalysisReport:
    """Formatted result of a point-group symmetry analysis."""

    group_name: str
    finite_supercell: bool
    root_groups: Sequence[tuple[int, ...]]
    assignments: Sequence[Mapping[str, float]]
    reference_assignment: Mapping[str, float] | None = None
    open_shell_assignment: Mapping[str, float] | None = None
    approximate: bool = False
    notes: Sequence[str] = ()

    def format_text(self) -> str:
        lines = [f"Point group: {self.group_name}"]
        if self.finite_supercell:
            lines.append(
                "Scope: finite supercell point-group analysis "
                "(not a full periodic space-group analysis)"
            )
        lines.append(f"Assignment quality: {'approximate' if self.approximate else 'strict'}")
        if self.reference_assignment is not None:
            lines.append("reference: " + _format_weights(self.reference_assignment))
        if self.open_shell_assignment is not None:
            lines.append("open shell: " + _format_weights(self.open_shell_assignment))
        for group, weights in zip(self.root_groups, self.assignments):
            label = ",".join(str(i + 1) for i in group)
            lines.append(f"roots {label}: " + _format_weights(weights))
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


class BaseMethodAdapter:
    """Base class for method-specific stored-vector layouts."""

    def __init__(self, td_obj):
        self.td_obj = td_obj

    @property
    def vectors(self) -> np.ndarray:
        vectors = np.asarray(self.td_obj.v)
        if vectors.ndim != 2:
            raise ValueError(f"td_obj.v must be a 2D array; got shape={vectors.shape}")
        return vectors

    def _state_vector(self, state: int) -> np.ndarray:
        vectors = self.vectors
        state = int(state)
        if state < 0:
            state += vectors.shape[1]
        if state < 0 or state >= vectors.shape[1]:
            raise IndexError(f"state index {state} out of range for {vectors.shape[1]} states")
        return np.asarray(vectors[:, state])

    def state_blocks(self, state: int) -> list[ExcitationBlock]:
        raise NotImplementedError


class SfTdaUpAdapter(BaseMethodAdapter):
    """Adapter for SF_TDA_up stored as one b -> a block."""

    def state_blocks(self, state: int) -> list[ExcitationBlock]:
        td = self.td_obj
        vec = self._state_vector(state)
        return [ExcitationBlock("b2a", vec.reshape(int(td.nc), int(td.nv)))]


class XsfTdaDownAdapter(BaseMethodAdapter):
    """Adapter for XSF_TDA_down stored in CV|CO|OV|OO order."""

    def state_blocks(self, state: int) -> list[ExcitationBlock]:
        td = self.td_obj
        vec = self._state_vector(state)
        nc, no, nv = int(td.nc), int(td.no), int(td.nv)
        dim1 = nc * nv
        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv
        cv = vec[:dim1].reshape(nc, nv)
        co = vec[dim1:dim2].reshape(nc, no)
        ov = vec[dim2:dim3].reshape(no, nv)
        oo = vec[dim3:]
        if bool(getattr(td, "re", False)):
            if not hasattr(td, "vects"):
                raise AttributeError("XSF_TDA_down object with re=True must provide vects")
            oo = np.asarray(td.vects) @ oo
        oo = oo.reshape(no, no)
        return [
            ExcitationBlock("CV", cv),
            ExcitationBlock("CO", co),
            ExcitationBlock("OV", ov),
            ExcitationBlock("OO", oo),
        ]


class XtdaAdapter(BaseMethodAdapter):
    """Adapter for XTDA restricted spin-tensor or unrestricted spin-orbital vectors."""

    def state_blocks(self, state: int) -> list[ExcitationBlock]:
        td = self.td_obj
        vec = self._state_vector(state)
        nc, no, nv = int(td.nc), int(td.no), int(td.nv)
        dim1 = nc * nv
        if bool(getattr(td, "type_u", False)):
            dim2 = dim1 + no * nv
            dim3 = dim2 + nc * no
            return [
                ExcitationBlock("CVa", vec[:dim1].reshape(nc, nv)),
                ExcitationBlock("OVa", vec[dim1:dim2].reshape(no, nv)),
                ExcitationBlock("COb", vec[dim2:dim3].reshape(nc, no)),
                ExcitationBlock("CVb", vec[dim3:].reshape(nc, nv)),
            ]

        dim2 = dim1 + nc * no
        dim3 = dim2 + no * nv
        return [
            ExcitationBlock("CV(0)", vec[:dim1].reshape(nc, nv)),
            ExcitationBlock("CO(0)", vec[dim1:dim2].reshape(nc, no)),
            ExcitationBlock("OV(0)", vec[dim2:dim3].reshape(no, nv)),
            ExcitationBlock("CV(1)", vec[dim3:].reshape(nc, nv)),
        ]


def group_degenerate_roots(energies, tol: float = 1.0e-5) -> list[tuple[int, ...]]:
    """Group consecutive roots whose energies differ by at most ``tol``."""

    values = np.asarray(energies, dtype=float).reshape(-1)
    if values.size == 0:
        return []
    groups: list[list[int]] = [[0]]
    group_ref = values[0]
    for idx in range(1, values.size):
        if abs(values[idx] - group_ref) <= tol:
            groups[-1].append(idx)
        else:
            groups.append([idx])
            group_ref = values[idx]
    return [tuple(group) for group in groups]


def decompose_characters(characters, table: CharacterTable) -> dict[str, float]:
    """Project representation characters onto irreps of ``table``."""

    chars = np.asarray(characters, dtype=complex).reshape(-1)
    nclasses = len(table.operation_labels)
    if chars.size != nclasses:
        raise ValueError(
            f"characters length {chars.size} does not match operation count {nclasses}"
        )
    if table.class_counts is None:
        class_counts = np.ones(nclasses, dtype=float)
    else:
        class_counts = np.asarray(table.class_counts, dtype=float).reshape(-1)
        if class_counts.size != nclasses:
            raise ValueError(
                f"class_counts length {class_counts.size} does not match "
                f"operation label count {nclasses}"
            )
    group_order = float(np.sum(class_counts))

    weights: dict[str, float] = {}
    for label, irrep_chars0 in table.irreps.items():
        irrep_chars = np.asarray(irrep_chars0, dtype=complex).reshape(-1)
        if irrep_chars.size != nclasses:
            raise ValueError(
                f"irrep {label!r} has {irrep_chars.size} characters, expected {nclasses}"
            )
        value = np.sum(class_counts * np.conj(irrep_chars) * chars) / group_order
        value = np.real_if_close(value, tol=1000)
        if np.iscomplexobj(value):
            weights[label] = float(np.real(value))
        else:
            weights[label] = float(value)
        if abs(weights[label]) < 1.0e-12:
            weights[label] = 0.0
        elif abs(weights[label] - round(weights[label])) < 1.0e-12:
            weights[label] = float(round(weights[label]))
    return weights


def determinant_characters(operation_matrices) -> np.ndarray:
    """Return determinant characters for occupied-subspace operation matrices."""

    chars = [np.linalg.det(np.asarray(op, dtype=complex)) for op in operation_matrices]
    return np.real_if_close(np.asarray(chars), tol=1000)


def make_method_adapter(td_obj, method_name: str | None = None) -> BaseMethodAdapter:
    """Create a method adapter for a TD object or an explicitly named method."""

    name = method_name or td_obj.__class__.__name__
    if name == "XTDA":
        return XtdaAdapter(td_obj)
    if name == "SF_TDA_up":
        return SfTdaUpAdapter(td_obj)
    if name == "XSF_TDA_down":
        return XsfTdaDownAdapter(td_obj)
    raise ValueError(f"Unsupported excited-state method for symmetry analysis: {name}")


def select_active_orbital_spaces(
    td_obj,
    root_group,
    amplitude_weight_cutoff: float = 1.0e-6,
    cumulative_weight_cutoff: float = 0.995,
) -> ActiveOrbitalSelection:
    """Select active orbital subspaces from important excitation amplitudes."""

    adapter = make_method_adapter(td_obj)
    specs = {spec.name: spec for spec in _method_block_specs(td_obj)}
    entries = []
    total_weight = 0.0

    for state in root_group:
        for block in adapter.state_blocks(state):
            spec = specs[block.name]
            weights = np.abs(block.amplitudes) ** 2
            total_weight += float(np.sum(weights))
            for left_pos, right_pos in zip(*np.where(weights > 0.0)):
                entries.append(
                    (
                        float(weights[left_pos, right_pos]),
                        spec,
                        int(left_pos),
                        int(right_pos),
                    )
                )

    active = {spec.left_key: set() for spec in specs.values()}
    active.update({spec.right_key: set() for spec in specs.values()})
    if total_weight <= 0.0 or not entries:
        return ActiveOrbitalSelection(
            spaces={key: tuple() for key in active},
            analyzed_weight=0.0,
            total_weight=0.0,
        )

    entries.sort(key=lambda item: item[0], reverse=True)
    selected_entries = []
    cumulative = 0.0
    for entry in entries:
        weight, _spec, _left_pos, _right_pos = entry
        if weight < amplitude_weight_cutoff and cumulative / total_weight >= cumulative_weight_cutoff:
            break
        selected_entries.append(entry)
        cumulative += weight
        if cumulative / total_weight >= cumulative_weight_cutoff and weight < amplitude_weight_cutoff:
            break

    if not selected_entries:
        selected_entries = [entries[0]]
        cumulative = entries[0][0]

    for weight, spec, left_pos, right_pos in selected_entries:
        del weight
        active[spec.left_key].add(spec.left_indices[left_pos])
        active[spec.right_key].add(spec.right_indices[right_pos])

    spaces = {key: tuple(sorted(values)) for key, values in active.items()}
    analyzed_weight = _active_amplitude_weight(td_obj, root_group, spaces, specs)
    return ActiveOrbitalSelection(
        spaces=spaces,
        analyzed_weight=analyzed_weight,
        total_weight=total_weight,
    )


def _method_block_specs(td_obj) -> list[ExcitationBlockSpec]:
    name = td_obj.__class__.__name__
    nc, no, nv = int(td_obj.nc), int(getattr(td_obj, "no", 0)), int(td_obj.nv)

    if name == "XSF_TDA_down":
        occ_a = _index_array(td_obj, "occidx_a", range(nc + no))
        vir_b = _index_array(td_obj, "viridx_b", range(nc, nc + no + nv))
        c_a = tuple(int(i) for i in occ_a[:nc])
        o_a = tuple(int(i) for i in occ_a[nc:nc + no])
        o_b = tuple(int(i) for i in vir_b[:no])
        v_b = tuple(int(i) for i in vir_b[no:no + nv])
        return [
            ExcitationBlockSpec("CV", ("alpha", "C"), ("beta", "V"), c_a, v_b),
            ExcitationBlockSpec("CO", ("alpha", "C"), ("beta", "O"), c_a, o_b),
            ExcitationBlockSpec("OV", ("alpha", "O"), ("beta", "V"), o_a, v_b),
            ExcitationBlockSpec("OO", ("alpha", "O"), ("beta", "O"), o_a, o_b),
        ]

    if name == "SF_TDA_up":
        occ_b = _index_array(td_obj, "occidx_b", range(nc))
        vir_a = _index_array(td_obj, "viridx_a", range(nc + no, nc + no + nv))
        return [
            ExcitationBlockSpec(
                "b2a",
                ("beta", "C"),
                ("alpha", "V"),
                tuple(int(i) for i in occ_b[:nc]),
                tuple(int(i) for i in vir_a[:nv]),
            )
        ]

    if name == "XTDA" and bool(getattr(td_obj, "type_u", False)):
        occ_a = _index_array(td_obj, "occidx_a", range(nc + no))
        vir_a = _index_array(td_obj, "viridx_a", range(nc + no, nc + no + nv))
        occ_b = _index_array(td_obj, "occidx_b", range(nc))
        vir_b = _index_array(td_obj, "viridx_b", range(nc, nc + no + nv))
        c_a = tuple(int(i) for i in occ_a[:nc])
        o_a = tuple(int(i) for i in occ_a[nc:nc + no])
        v_a = tuple(int(i) for i in vir_a[:nv])
        c_b = tuple(int(i) for i in occ_b[:nc])
        o_b = tuple(int(i) for i in vir_b[:no])
        v_b = tuple(int(i) for i in vir_b[no:no + nv])
        return [
            ExcitationBlockSpec("CVa", ("alpha", "C"), ("alpha", "V"), c_a, v_a),
            ExcitationBlockSpec("OVa", ("alpha", "O"), ("alpha", "V"), o_a, v_a),
            ExcitationBlockSpec("COb", ("beta", "C"), ("beta", "O"), c_b, o_b),
            ExcitationBlockSpec("CVb", ("beta", "C"), ("beta", "V"), c_b, v_b),
        ]

    if name == "XTDA":
        c = tuple(range(nc))
        o = tuple(range(nc, nc + no))
        v = tuple(range(nc + no, nc + no + nv))
        return [
            ExcitationBlockSpec("CV(0)", ("alpha", "C"), ("alpha", "V"), c, v),
            ExcitationBlockSpec("CO(0)", ("alpha", "C"), ("alpha", "O"), c, o),
            ExcitationBlockSpec("OV(0)", ("alpha", "O"), ("alpha", "V"), o, v),
            ExcitationBlockSpec("CV(1)", ("alpha", "C"), ("alpha", "V"), c, v),
        ]

    raise ValueError(f"Unsupported excited-state method for symmetry analysis: {name}")


def _active_amplitude_weight(td_obj, root_group, spaces, specs_by_name) -> float:
    adapter = make_method_adapter(td_obj)
    kept = 0.0
    total = 0.0
    for state in root_group:
        for block in adapter.state_blocks(state):
            spec = specs_by_name[block.name]
            weights = np.abs(block.amplitudes) ** 2
            total += float(np.sum(weights))
            left_active = set(spaces.get(spec.left_key, ()))
            right_active = set(spaces.get(spec.right_key, ()))
            left_mask = [idx for idx, orb in enumerate(spec.left_indices) if orb in left_active]
            right_mask = [idx for idx, orb in enumerate(spec.right_indices) if orb in right_active]
            if left_mask and right_mask:
                kept += float(np.sum(weights[np.ix_(left_mask, right_mask)]))
    return 0.0 if total <= 0.0 else kept / total


def _index_array(obj, name: str, fallback) -> np.ndarray:
    return np.asarray(getattr(obj, name, list(fallback)), dtype=int)


def detect_geometry_symmetry(
    system,
    threshold: float = 1.0e-3,
    point_group: str | None = None,
) -> GeometrySymmetry:
    """Detect finite point-group symmetry and extract a character table.

    ``system`` can be any PySCF-like object exposing ``natm``,
    ``atom_symbol(i)``, and ``atom_coords()``.  For Gamma-point supercells this
    intentionally treats the cell as a finite cluster of atoms.
    """

    msym = _require_libmsym()
    symbols_coords = _system_symbols_coords(system)
    origin = np.mean([coords for _symbol, coords in symbols_coords], axis=0)
    elements = [
        msym.Element(name=symbol, coordinates=coords - origin)
        for symbol, coords in symbols_coords
    ]
    with msym.Context(elements=elements) as ctx:
        ctx.set_thresholds(
            geometry=float(threshold),
            equivalence=float(threshold),
            permutation=float(threshold),
        )
        if point_group is None:
            detected_group = ctx.find_symmetry()
        else:
            ctx.point_group = point_group
            detected_group = ctx.point_group
        table = _character_table_from_libmsym(detected_group, ctx.character_table)
        operations = tuple(ctx.symmetry_operations)
        operation_labels = tuple(_operation_label(op) for op in operations)
        cartesian_operations = tuple(_cartesian_matrix_from_libmsym(op) for op in operations)
        class_indices = tuple(int(op.conjugacy_class) for op in operations)
    return GeometrySymmetry(
        point_group=detected_group,
        character_table=table,
        symmetry_operation_labels=operation_labels,
        cartesian_operations=cartesian_operations,
        operation_class_indices=class_indices,
        origin=origin,
    )


def detect_pbc_spglib_symmetry(
    system,
    symprec: float = 1.0e-3,
    point_group: str | None = None,
) -> GeometrySymmetry:
    """Detect periodic point-group operations with spglib."""

    try:
        import spglib
    except ImportError as err:
        raise ImportError("spglib is required for projection_backend='spglib_ao_permutation'") from err

    lattice = np.asarray(system.lattice_vectors(), dtype=float)
    inv_lattice = np.linalg.inv(lattice)
    coords = np.asarray(system.atom_coords(), dtype=float)
    frac = coords @ inv_lattice
    frac = frac - np.floor(frac)
    numbers = _system_atomic_numbers(system)

    dataset = spglib.get_symmetry_dataset(
        (lattice, frac, numbers),
        symprec=float(symprec),
    )
    if dataset is None:
        raise ValueError(f"spglib could not detect symmetry with symprec={symprec}")

    rotations = np.asarray(_spglib_dataset_value(dataset, "rotations"), dtype=int)
    translations = np.asarray(_spglib_dataset_value(dataset, "translations"), dtype=float)
    hm_point_group = str(_spglib_dataset_value(dataset, "pointgroup"))
    schoenflies = _normalize_point_group_label(hm_point_group)
    if point_group is not None:
        requested = _normalize_point_group_label(point_group)
        if requested != schoenflies:
            raise ValueError(
                "spglib detected point group "
                f"{hm_point_group} ({schoenflies}), but point_group={point_group!r} "
                "was requested. Increase symmetry_tol/symprec or inspect the structure."
            )
        schoenflies = requested

    unique_rotations = []
    unique_translations = []
    seen = set()
    for rotation, translation in zip(rotations, translations):
        key = tuple(int(x) for x in rotation.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        unique_rotations.append(rotation)
        unique_translations.append(translation - np.floor(translation))

    cartesian_ops = tuple(
        inv_lattice @ rotation.T @ lattice
        for rotation in unique_rotations
    )
    table = _character_table_for_point_group(schoenflies)
    class_indices = tuple(
        _operation_class_index_for_point_group(schoenflies, op)
        for op in cartesian_ops
    )
    operation_labels = tuple(
        table.operation_labels[class_index]
        for class_index in class_indices
    )

    return GeometrySymmetry(
        point_group=schoenflies,
        character_table=table,
        symmetry_operation_labels=operation_labels,
        cartesian_operations=cartesian_ops,
        operation_class_indices=class_indices,
        origin=np.zeros(3),
        fractional_operations=tuple(unique_rotations),
        fractional_translations=tuple(unique_translations),
        lattice_vectors=lattice,
    )


def build_ao_operation_matrices(
    mol,
    geometry: GeometrySymmetry,
    grid_level: int = 5,
    grid_coords=None,
    grid_weights=None,
    grid_block_size: int = 20000,
):
    """Build AO representation matrices for the detected symmetry operations."""

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


def build_ao_permutation_operation_matrices(
    mol,
    geometry: GeometrySymmetry,
    tol: float = 1.0e-5,
):
    """Build AO operation matrices from atom permutation and shell rotations."""

    ao_loc = np.asarray(mol.ao_loc_nr(), dtype=int)
    nao = int(ao_loc[-1])
    ops = []
    for rotation in geometry.cartesian_operations:
        atom_map = _atom_map_for_operation(mol, geometry.origin, rotation, tol=tol)
        shell_map = _shell_map_for_atom_map(mol, atom_map)
        shell_rotations = _shell_rotation_matrices(mol, rotation)
        uao = np.zeros((nao, nao))
        for ib, jb in enumerate(shell_map):
            l = int(mol.bas_angular(ib))
            nctr = int(mol.bas_nctr(ib))
            block = np.kron(np.eye(nctr), shell_rotations[l])
            i0, i1 = int(ao_loc[ib]), int(ao_loc[ib + 1])
            j0, j1 = int(ao_loc[jb]), int(ao_loc[jb + 1])
            uao[j0:j1, i0:i1] = block
        ops.append(uao)
    return tuple(ops)


def build_ao_spglib_operation_matrices(
    mol,
    geometry: GeometrySymmetry,
    tol: float = 1.0e-5,
):
    """Build AO operation matrices from spglib fractional operations."""

    if geometry.fractional_operations is None or geometry.fractional_translations is None:
        raise ValueError("geometry does not contain spglib fractional operations")

    ao_loc = np.asarray(mol.ao_loc_nr(), dtype=int)
    nao = int(ao_loc[-1])
    ops = []
    for frac_rotation, frac_translation, cart_rotation in zip(
        geometry.fractional_operations,
        geometry.fractional_translations,
        geometry.cartesian_operations,
    ):
        atom_map = _atom_map_for_fractional_operation(
            mol,
            frac_rotation,
            frac_translation,
            tol=tol,
        )
        shell_map = _shell_map_for_atom_map(mol, atom_map)
        shell_rotations = _shell_rotation_matrices(mol, cart_rotation)
        uao = np.zeros((nao, nao))
        for ib, jb in enumerate(shell_map):
            l = int(mol.bas_angular(ib))
            nctr = int(mol.bas_nctr(ib))
            block = np.kron(np.eye(nctr), shell_rotations[l])
            i0, i1 = int(ao_loc[ib]), int(ao_loc[ib + 1])
            j0, j1 = int(ao_loc[jb]), int(ao_loc[jb + 1])
            uao[j0:j1, i0:i1] = block
        ops.append(uao)
    return tuple(ops)


def build_mo_operation_matrices(
    mf,
    geometry: GeometrySymmetry,
    grid_level: int = 5,
    grid_coords=None,
    grid_weights=None,
    grid_block_size: int = 20000,
    projection_backend: str = "grid",
    operation_tol: float = 1.0e-5,
):
    """Build spin-resolved MO representation matrices from AO operations."""

    system = getattr(mf, "mol", None)
    if system is None:
        system = getattr(mf, "cell", None)
    if system is None:
        raise TypeError("mf must expose either .mol or .cell")
    if projection_backend == "grid":
        ao_ops = build_ao_operation_matrices(
            system,
            geometry,
            grid_level=grid_level,
            grid_coords=grid_coords,
            grid_weights=grid_weights,
            grid_block_size=grid_block_size,
        )
    elif projection_backend == "ao_permutation":
        ao_ops = build_ao_permutation_operation_matrices(
            system,
            geometry,
            tol=operation_tol,
        )
    elif projection_backend == "spglib_ao_permutation":
        ao_ops = build_ao_spglib_operation_matrices(
            system,
            geometry,
            tol=operation_tol,
        )
    else:
        raise ValueError(
            "projection_backend must be 'grid', 'ao_permutation', or "
            "'spglib_ao_permutation'"
        )
    overlap = _overlap_matrix(system)
    coeff_alpha, coeff_beta = _spin_mo_coefficients(mf)
    alpha = tuple(coeff_alpha.conj().T @ overlap @ op @ coeff_alpha for op in ao_ops)
    beta = tuple(coeff_beta.conj().T @ overlap @ op @ coeff_beta for op in ao_ops)
    return MOOperationMatrices(alpha=alpha, beta=beta, ao=ao_ops)


def build_mo_subspace_operation_matrices(
    mf,
    geometry: GeometrySymmetry,
    spaces: Mapping[tuple[str, str], Sequence[int]],
    grid_level: int = 5,
    grid_coords=None,
    grid_weights=None,
    grid_block_size: int = 20000,
    projection_backend: str = "grid",
    operation_tol: float = 1.0e-5,
) -> SubspaceOperationMatrices:
    """Build MO operation matrices only for selected orbital subspaces."""

    system = getattr(mf, "mol", None)
    if system is None:
        system = getattr(mf, "cell", None)
    if system is None:
        raise TypeError("mf must expose either .mol or .cell")

    coeff_alpha, coeff_beta = _spin_mo_coefficients(mf)
    coeff_by_spin = {"alpha": coeff_alpha, "beta": coeff_beta}
    normalized_spaces = {
        key: tuple(int(i) for i in indices)
        for key, indices in spaces.items()
        if len(tuple(indices)) > 0
    }

    if projection_backend in ("ao_permutation", "spglib_ao_permutation"):
        overlap = _overlap_matrix(system)
        if projection_backend == "ao_permutation":
            ao_ops = build_ao_permutation_operation_matrices(
                system,
                geometry,
                tol=operation_tol,
            )
        else:
            ao_ops = build_ao_spglib_operation_matrices(
                system,
                geometry,
                tol=operation_tol,
            )
        matrices = {key: [] for key in normalized_spaces}
        for op in ao_ops:
            for key, indices in normalized_spaces.items():
                spin, _label = key
                coeff = coeff_by_spin[spin][:, indices]
                subspace_op = coeff.conj().T @ overlap @ op @ coeff
                matrices[key].append(np.real_if_close(subspace_op, tol=1000))
        return SubspaceOperationMatrices(
            matrices={key: tuple(values) for key, values in matrices.items()},
            spaces=normalized_spaces,
        )

    if projection_backend != "grid":
        raise ValueError(
            "projection_backend must be 'grid', 'ao_permutation', or "
            "'spglib_ao_permutation'"
        )

    if grid_coords is None or grid_weights is None:
        try:
            from pyscf.dft import gen_grid
        except ImportError as err:
            raise ImportError("PySCF is required to build MO subspace matrices") from err
        grids = gen_grid.Grids(system)
        grids.level = int(grid_level)
        grids.build()
        coords = np.asarray(grids.coords)
        weights = np.asarray(grids.weights)
    else:
        coords = np.asarray(grid_coords)
        weights = np.asarray(grid_weights)

    matrices = {key: [] for key in normalized_spaces}

    for rotation in geometry.cartesian_operations:
        accum = {
            key: np.zeros((len(indices), len(indices)), dtype=complex)
            for key, indices in normalized_spaces.items()
        }
        for p0 in range(0, coords.shape[0], int(grid_block_size)):
            p1 = min(p0 + int(grid_block_size), coords.shape[0])
            coord_block = coords[p0:p1]
            weight_block = weights[p0:p1]
            inverse_coords = (coord_block - geometry.origin) @ rotation + geometry.origin
            ao = _eval_gto_values(system, coord_block)
            ao_rot = _eval_gto_values(system, inverse_coords)
            for key, indices in normalized_spaces.items():
                spin, _label = key
                coeff = coeff_by_spin[spin][:, indices]
                phi = ao @ coeff
                phi_rot = ao_rot @ coeff
                accum[key] += phi.conj().T @ (weight_block[:, None] * phi_rot)
        for key in normalized_spaces:
            matrices[key].append(np.real_if_close(accum[key], tol=1000))

    return SubspaceOperationMatrices(
        matrices={key: tuple(values) for key, values in matrices.items()},
        spaces=normalized_spaces,
    )


def analyze_excited_state_symmetry(
    td_obj,
    energy_tol: float = 1.0e-5,
    symmetry_tol: float = 1.0e-5,
    point_group: str | None = None,
    grid_level: int = 5,
    grid_coords=None,
    grid_weights=None,
    grid_block_size: int = 20000,
    projection_backend: str = "grid",
    analysis_mode: str = "active",
    reference_mode: str = "open_shell",
    amplitude_weight_cutoff: float = 1.0e-6,
    cumulative_weight_cutoff: float = 0.995,
    min_analyzed_weight: float = 0.95,
    active_roots=None,
):
    """Analyze reference and excited-state symmetry for a TD object.

    ``projection_backend="grid"`` uses numerical AO projection on PySCF grids.
    ``projection_backend="ao_permutation"`` uses atom permutations and AO shell
    rotations, which is much faster when the geometry is symmetry-adapted.
    """

    mf = getattr(td_obj, "mf", None)
    if mf is None:
        raise TypeError("td_obj must expose .mf for symmetry analysis")
    system = getattr(mf, "mol", None)
    finite_supercell = False
    if system is None:
        system = getattr(mf, "cell", None)
        finite_supercell = system is not None
    if system is None:
        raise TypeError("td_obj.mf must expose either .mol or .cell")

    if projection_backend == "spglib_ao_permutation":
        geometry = detect_pbc_spglib_symmetry(
            system,
            symprec=symmetry_tol,
            point_group=point_group,
        )
    else:
        geometry = detect_geometry_symmetry(
            system,
            threshold=symmetry_tol,
            point_group=point_group,
        )
    energies = np.asarray(getattr(td_obj, "e"), dtype=float).reshape(-1)
    groups = group_degenerate_roots(energies, tol=energy_tol)
    if active_roots is not None:
        active_roots = {int(root) for root in active_roots}
        groups = [group for group in groups if any(root in active_roots for root in group)]

    notes = []
    notes.append(f"projection backend: {projection_backend}")
    if analysis_mode != "active":
        mo_ops = build_mo_operation_matrices(
            mf,
            geometry,
            grid_level=grid_level,
            grid_coords=grid_coords,
            grid_weights=grid_weights,
            grid_block_size=grid_block_size,
            projection_backend=projection_backend,
            operation_tol=symmetry_tol,
        )
        assignments = [
            decompose_characters(
                _characters_by_class(
                    _excited_subspace_operation_characters(td_obj, mo_ops, group),
                    geometry,
                ),
                geometry.character_table,
            )
            for group in groups
        ]
        reference_chars, open_shell_chars = _reference_operation_characters(mf, mo_ops)
        reference_assignment = decompose_characters(
            _characters_by_class(reference_chars, geometry),
            geometry.character_table,
        )
        open_shell_assignment = None
        if open_shell_chars is not None:
            open_shell_assignment = decompose_characters(
                _characters_by_class(open_shell_chars, geometry),
                geometry.character_table,
            )

        overlap = _overlap_matrix(system)
        max_ao_error = max(
            np.linalg.norm(op.conj().T @ overlap @ op - overlap)
            for op in mo_ops.ao
        )
        notes.append(f"max AO overlap invariance error: {max_ao_error:.3e}")
        approximate = max_ao_error > symmetry_tol
    else:
        selections = [
            select_active_orbital_spaces(
                td_obj,
                group,
                amplitude_weight_cutoff=amplitude_weight_cutoff,
                cumulative_weight_cutoff=cumulative_weight_cutoff,
            )
            for group in groups
        ]
        reference_spaces = _reference_active_spaces(mf, reference_mode=reference_mode)
        all_spaces = _merge_active_spaces([sel.spaces for sel in selections] + [reference_spaces])
        subspace_ops = build_mo_subspace_operation_matrices(
            mf,
            geometry,
            all_spaces,
            grid_level=grid_level,
            grid_coords=grid_coords,
            grid_weights=grid_weights,
            grid_block_size=grid_block_size,
            projection_backend=projection_backend,
            operation_tol=symmetry_tol,
        )
        assignments = [
            decompose_characters(
                _characters_by_class(
                    _active_excited_subspace_operation_characters(
                        td_obj, group, selection, subspace_ops
                    ),
                    geometry,
                ),
                geometry.character_table,
            )
            for group, selection in zip(groups, selections)
        ]
        reference_chars = _reference_active_operation_characters(reference_spaces, subspace_ops)
        reference_assignment = decompose_characters(
            _characters_by_class(reference_chars, geometry),
            geometry.character_table,
        )
        open_shell_assignment = reference_assignment if reference_mode == "open_shell" else None
        approximate = False
        for group, selection in zip(groups, selections):
            label = ",".join(str(i + 1) for i in group)
            notes.append(
                f"roots {label} analyzed amplitude weight: {selection.analyzed_weight:.6f}"
            )
            notes.append(
                f"roots {label} active orbitals: "
                + " ".join(
                    f"{spin}:{space}={len(indices)}"
                    for (spin, space), indices in sorted(selection.spaces.items())
                    if len(indices) > 0
                )
            )
            if selection.analyzed_weight < min_analyzed_weight:
                approximate = True
    return SymmetryAnalysisReport(
        group_name=geometry.point_group,
        finite_supercell=finite_supercell,
        root_groups=groups,
        assignments=assignments,
        reference_assignment=reference_assignment,
        open_shell_assignment=open_shell_assignment,
        approximate=approximate,
        notes=tuple(notes),
    )


def _spin_mo_coefficients(mf) -> tuple[np.ndarray, np.ndarray]:
    coeff = np.asarray(mf.mo_coeff)
    if coeff.ndim == 3:
        return np.asarray(coeff[0]), np.asarray(coeff[1])
    return coeff, coeff


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


def _spin_mo_occupations(mf) -> tuple[np.ndarray, np.ndarray]:
    occ = np.asarray(mf.mo_occ)
    if occ.ndim == 2:
        return np.asarray(occ[0]), np.asarray(occ[1])
    occ_alpha = np.zeros_like(occ, dtype=float)
    occ_beta = np.zeros_like(occ, dtype=float)
    occ_alpha[occ >= 1.0e-12] = 1.0
    occ_beta[occ > 1.0 + 1.0e-12] = 1.0
    return occ_alpha, occ_beta


def _reference_operation_characters(mf, mo_ops: MOOperationMatrices):
    occ_alpha, occ_beta = _spin_mo_occupations(mf)
    idx_alpha = np.where(occ_alpha > 0.5)[0]
    idx_beta = np.where(occ_beta > 0.5)[0]
    chars = []
    open_chars = []
    open_idx = np.where((occ_alpha > 0.5) & (occ_beta < 0.5))[0]
    for ua, ub in zip(mo_ops.alpha, mo_ops.beta):
        det_a = np.linalg.det(ua[np.ix_(idx_alpha, idx_alpha)]) if idx_alpha.size else 1.0
        det_b = np.linalg.det(ub[np.ix_(idx_beta, idx_beta)]) if idx_beta.size else 1.0
        chars.append(det_a * det_b)
        if open_idx.size:
            open_chars.append(np.linalg.det(ua[np.ix_(open_idx, open_idx)]))
    ref = np.real_if_close(np.asarray(chars), tol=1000)
    if open_idx.size:
        return ref, np.real_if_close(np.asarray(open_chars), tol=1000)
    return ref, None


def _reference_active_spaces(mf, reference_mode: str = "open_shell"):
    occ_alpha, occ_beta = _spin_mo_occupations(mf)
    if reference_mode == "open_shell":
        open_idx = tuple(int(i) for i in np.where((occ_alpha > 0.5) & (occ_beta < 0.5))[0])
        return {("alpha", "open"): open_idx}
    if reference_mode == "full_occ":
        alpha_idx = tuple(int(i) for i in np.where(occ_alpha > 0.5)[0])
        beta_idx = tuple(int(i) for i in np.where(occ_beta > 0.5)[0])
        return {("alpha", "occ"): alpha_idx, ("beta", "occ"): beta_idx}
    raise ValueError("reference_mode must be 'open_shell' or 'full_occ'")


def _reference_active_operation_characters(reference_spaces, subspace_ops):
    if not reference_spaces:
        nops = _nops_from_subspace_ops(subspace_ops)
        return np.ones(nops)
    nops = _nops_from_subspace_ops(subspace_ops)
    chars = []
    for op_index in range(nops):
        value = 1.0 + 0.0j
        for key, indices in reference_spaces.items():
            if len(indices) == 0:
                continue
            value *= np.linalg.det(subspace_ops.matrices[key][op_index])
        chars.append(value)
    return np.real_if_close(np.asarray(chars), tol=1000)


def _excited_subspace_operation_characters(td_obj, mo_ops: MOOperationMatrices, group):
    adapter = make_method_adapter(td_obj)
    chars = []
    for op_index in range(len(mo_ops.alpha)):
        total = 0.0j
        for state in group:
            total += _state_overlap_with_transformed(td_obj, adapter, state, mo_ops, op_index)
        chars.append(total)
    return np.real_if_close(np.asarray(chars), tol=1000)


def _active_excited_subspace_operation_characters(
    td_obj,
    root_group,
    selection: ActiveOrbitalSelection,
    subspace_ops: SubspaceOperationMatrices,
):
    adapter = make_method_adapter(td_obj)
    specs = {spec.name: spec for spec in _method_block_specs(td_obj)}
    nops = _nops_from_subspace_ops(subspace_ops)
    chars = []
    for op_index in range(nops):
        total = 0.0j
        for state in root_group:
            blocks = {block.name: block.amplitudes for block in adapter.state_blocks(state)}
            for block_name, amplitudes in blocks.items():
                spec = specs[block_name]
                left_orbitals = selection.spaces.get(spec.left_key, ())
                right_orbitals = selection.spaces.get(spec.right_key, ())
                if len(left_orbitals) == 0 or len(right_orbitals) == 0:
                    continue
                left_positions = [spec.left_indices.index(orb) for orb in left_orbitals]
                right_positions = [spec.right_indices.index(orb) for orb in right_orbitals]
                x_block = amplitudes[np.ix_(left_positions, right_positions)]
                global_left = subspace_ops.spaces[spec.left_key]
                global_right = subspace_ops.spaces[spec.right_key]
                left_subpos = [global_left.index(orb) for orb in left_orbitals]
                right_subpos = [global_right.index(orb) for orb in right_orbitals]
                u_left_all = subspace_ops.matrices[spec.left_key][op_index]
                u_right_all = subspace_ops.matrices[spec.right_key][op_index]
                u_left = u_left_all[np.ix_(left_subpos, left_subpos)]
                u_right = u_right_all[np.ix_(right_subpos, right_subpos)]
                transformed = u_left.conj() @ x_block @ u_right.T
                total += np.vdot(x_block, transformed)
        chars.append(total)
    return np.real_if_close(np.asarray(chars), tol=1000)


def _merge_active_spaces(space_maps) -> dict[tuple[str, str], tuple[int, ...]]:
    merged: dict[tuple[str, str], set[int]] = {}
    for spaces in space_maps:
        for key, indices in spaces.items():
            merged.setdefault(key, set()).update(int(i) for i in indices)
    return {key: tuple(sorted(indices)) for key, indices in merged.items()}


def _nops_from_subspace_ops(subspace_ops: SubspaceOperationMatrices) -> int:
    if not subspace_ops.matrices:
        return 1
    first = next(iter(subspace_ops.matrices.values()))
    return len(first)


def _state_overlap_with_transformed(td_obj, adapter, state, mo_ops, op_index):
    name = td_obj.__class__.__name__
    if name == "XTDA":
        return _xtda_state_overlap(td_obj, adapter, state, mo_ops, op_index)
    if name == "XSF_TDA_down":
        return _xsf_down_state_overlap(td_obj, adapter, state, mo_ops, op_index)
    if name == "SF_TDA_up":
        return _sf_up_state_overlap(td_obj, adapter, state, mo_ops, op_index)
    raise ValueError(f"Unsupported excited-state method for symmetry analysis: {name}")


def _xtda_state_overlap(td, adapter, state, mo_ops, op_index):
    blocks = {block.name: block.amplitudes for block in adapter.state_blocks(state)}
    if bool(getattr(td, "type_u", False)):
        ua = mo_ops.alpha[op_index]
        ub = mo_ops.beta[op_index]
        nc, no, nv = int(td.nc), int(td.no), int(td.nv)
        pieces = (
            ("CVa", ua[:nc, :nc], ua[nc + no:, nc + no:]),
            ("OVa", ua[nc:nc + no, nc:nc + no], ua[nc + no:, nc + no:]),
            ("COb", ub[:nc, :nc], ub[nc:nc + no, nc:nc + no]),
            ("CVb", ub[:nc, :nc], ub[nc + no:, nc + no:]),
        )
    else:
        u = mo_ops.alpha[op_index]
        nc, no, nv = int(td.nc), int(td.no), int(td.nv)
        c = slice(0, nc)
        o = slice(nc, nc + no)
        v = slice(nc + no, nc + no + nv)
        pieces = (
            ("CV(0)", u[c, c], u[v, v]),
            ("CO(0)", u[c, c], u[o, o]),
            ("OV(0)", u[o, o], u[v, v]),
            ("CV(1)", u[c, c], u[v, v]),
        )
    return _block_overlap_sum(blocks, pieces)


def _xsf_down_state_overlap(td, adapter, state, mo_ops, op_index):
    blocks = {block.name: block.amplitudes for block in adapter.state_blocks(state)}
    ua = mo_ops.alpha[op_index]
    ub = mo_ops.beta[op_index]
    nc, no, nv = int(td.nc), int(td.no), int(td.nv)
    c = slice(0, nc)
    o = slice(nc, nc + no)
    v = slice(nc + no, nc + no + nv)
    pieces = (
        ("CV", ua[c, c], ub[v, v]),
        ("CO", ua[c, c], ub[o, o]),
        ("OV", ua[o, o], ub[v, v]),
        ("OO", ua[o, o], ub[o, o]),
    )
    return _block_overlap_sum(blocks, pieces)


def _sf_up_state_overlap(td, adapter, state, mo_ops, op_index):
    blocks = {block.name: block.amplitudes for block in adapter.state_blocks(state)}
    ub = mo_ops.beta[op_index]
    ua = mo_ops.alpha[op_index]
    nc, no, nv = int(td.nc), int(getattr(td, "no", 0)), int(td.nv)
    occ_b = ub[:nc, :nc]
    vir_a = ua[nc + no:nc + no + nv, nc + no:nc + no + nv]
    return _block_overlap_sum(blocks, (("b2a", occ_b, vir_a),))


def _block_overlap_sum(blocks, pieces):
    value = 0.0j
    for name, left, right in pieces:
        amplitudes = np.asarray(blocks[name])
        transformed = left.conj() @ amplitudes @ right.T
        value += np.vdot(amplitudes, transformed)
    return value


def _characters_by_class(operation_characters, geometry: GeometrySymmetry) -> np.ndarray:
    op_chars = np.asarray(operation_characters, dtype=complex)
    nclasses = len(geometry.character_table.operation_labels)
    class_chars = np.zeros(nclasses, dtype=complex)
    counts = np.zeros(nclasses, dtype=float)
    for value, cls in zip(op_chars, geometry.operation_class_indices):
        class_chars[int(cls)] += value
        counts[int(cls)] += 1.0
    nonzero = counts > 0
    class_chars[nonzero] /= counts[nonzero]
    return np.real_if_close(class_chars, tol=1000)


def _format_weights(weights: Mapping[str, float]) -> str:
    return " ".join(f"{label}: {value:.6f}" for label, value in weights.items())


def _require_libmsym():
    try:
        import libmsym
    except ImportError as err:
        raise ImportError(
            "libmsym is required for geometry symmetry detection and AO projection. "
            "Install libmsym or run only the lightweight adapter/projection helpers."
        ) from err
    return libmsym


def _system_symbols_coords(system) -> list[tuple[str, np.ndarray]]:
    if not hasattr(system, "natm"):
        raise TypeError("system must expose PySCF-like natm, atom_symbol, and atom_coords")
    coords = np.asarray(system.atom_coords(), dtype=float)
    if coords.shape != (int(system.natm), 3):
        raise ValueError(f"atom_coords must have shape ({system.natm}, 3); got {coords.shape}")
    return [
        (str(system.atom_symbol(i)), np.asarray(coords[i], dtype=float))
        for i in range(int(system.natm))
    ]


def _system_atomic_numbers(system) -> list[int]:
    try:
        from pyscf.data import elements
    except ImportError as err:
        raise ImportError("PySCF is required to convert atom symbols to atomic numbers") from err
    return [int(elements.charge(str(system.atom_symbol(i)))) for i in range(int(system.natm))]


def _spglib_dataset_value(dataset, key: str):
    if hasattr(dataset, key):
        return getattr(dataset, key)
    return dataset[key]


def _normalize_point_group_label(label: str) -> str:
    compact = str(label).replace(" ", "")
    mapping = {
        "1": "C1",
        "C1": "C1",
        "-1": "Ci",
        "Ci": "Ci",
        "m": "Cs",
        "Cs": "Cs",
        "3": "C3",
        "C3": "C3",
        "3m": "C3v",
        "C3v": "C3v",
        "32": "D3",
        "D3": "D3",
        "-3m": "D3d",
        "D3d": "D3d",
        "m-3m": "Oh",
        "Oh": "Oh",
    }
    return mapping.get(compact, compact)


def _character_table_for_point_group(point_group: str) -> CharacterTable:
    group = _normalize_point_group_label(point_group)
    if group in {"C1", "Ci", "Cs", "C3v", "D3d"}:
        msym = _require_libmsym()
        with msym.Context(elements=[msym.Element(name="H", coordinates=[0.0, 0.0, 0.0])]) as ctx:
            ctx.point_group = group
            return _character_table_from_libmsym(group, ctx.character_table)
    raise ValueError(
        f"Unsupported spglib point group {point_group!r} for character analysis. "
        "Add a class classifier and character table for this group."
    )


def _operation_class_index_for_point_group(point_group: str, cartesian_operation) -> int:
    group = _normalize_point_group_label(point_group)
    op = np.asarray(cartesian_operation, dtype=float)
    det = round(float(np.linalg.det(op)))
    trace = round(float(np.trace(op)))
    if group == "C1":
        return 0
    if group == "Ci":
        return 0 if det > 0 else 1
    if group == "Cs":
        return 0 if det > 0 else 1
    if group == "C3v":
        if det > 0 and trace == 3:
            return 0
        if det > 0:
            return 1
        return 2
    if group == "D3d":
        # libmsym D3d class order is E, 2S6, 2C3, i, 3sigma_d, 3C2'.
        if det > 0 and trace == 3:
            return 0
        if det < 0 and trace == 0:
            return 1
        if det > 0 and trace == 0:
            return 2
        if det < 0 and trace == -3:
            return 3
        if det < 0 and trace == 1:
            return 4
        if det > 0 and trace == -1:
            return 5
    raise ValueError(
        f"Cannot classify operation for point group {point_group!r}: "
        f"det={det}, trace={trace}"
    )


def _atom_map_for_fractional_operation(
    system,
    fractional_rotation,
    fractional_translation,
    tol: float = 1.0e-5,
) -> tuple[int, ...]:
    symbols_coords = _system_symbols_coords(system)
    symbols = [symbol for symbol, _coords in symbols_coords]
    coords = np.asarray([coords for _symbol, coords in symbols_coords])
    lattice = np.asarray(system.lattice_vectors(), dtype=float)
    inv_lattice = np.linalg.inv(lattice)
    frac_atoms = coords @ inv_lattice
    frac_atoms = frac_atoms - np.floor(frac_atoms)
    rotation = np.asarray(fractional_rotation, dtype=int)
    translation = np.asarray(fractional_translation, dtype=float)
    frac_rot = frac_atoms @ rotation.T + translation
    frac_rot = frac_rot - np.floor(frac_rot)

    atom_map = []
    used = set()
    for i, frot in enumerate(frac_rot):
        best = None
        best_dist = None
        for j, fatom in enumerate(frac_atoms):
            if j in used or symbols[j] != symbols[i]:
                continue
            diff = frot - fatom
            diff -= np.rint(diff)
            dist = float(np.linalg.norm(diff @ lattice))
            if best is None or dist < best_dist:
                best = j
                best_dist = dist
        if best is None or best_dist is None or best_dist > tol:
            raise ValueError(
                f"Could not map atom {i} under spglib operation; "
                f"best distance={best_dist}"
            )
        used.add(best)
        atom_map.append(best)
    return tuple(atom_map)


def _atom_map_for_operation(system, origin, rotation, tol: float = 1.0e-5) -> tuple[int, ...]:
    symbols_coords = _system_symbols_coords(system)
    symbols = [symbol for symbol, _coords in symbols_coords]
    coords = np.asarray([coords for _symbol, coords in symbols_coords])
    rotated = (coords - origin) @ rotation + origin

    if hasattr(system, "lattice_vectors"):
        lattice = np.asarray(system.lattice_vectors())
        inv_lattice = np.linalg.inv(lattice)
        frac_atoms = coords @ inv_lattice
        frac_rot = rotated @ inv_lattice
        frac_atoms = frac_atoms - np.floor(frac_atoms)
        frac_rot = frac_rot - np.floor(frac_rot)
        atom_map = []
        used = set()
        for i, frot in enumerate(frac_rot):
            best = None
            best_dist = None
            for j, fatom in enumerate(frac_atoms):
                if j in used or symbols[j] != symbols[i]:
                    continue
                diff = frot - fatom
                diff -= np.rint(diff)
                cart_diff = diff @ lattice
                dist = float(np.linalg.norm(cart_diff))
                if best is None or dist < best_dist:
                    best = j
                    best_dist = dist
            if best is None or best_dist is None or best_dist > tol:
                raise ValueError(
                    f"Could not map atom {i} under symmetry operation; "
                    f"best distance={best_dist}"
                )
            used.add(best)
            atom_map.append(best)
        return tuple(atom_map)

    atom_map = []
    used = set()
    for i, rrot in enumerate(rotated):
        distances = np.linalg.norm(coords - rrot, axis=1)
        candidates = [
            (distances[j], j)
            for j in range(len(symbols))
            if j not in used and symbols[j] == symbols[i]
        ]
        if not candidates:
            raise ValueError(f"Could not map atom {i}: no same-symbol candidates")
        dist, best = min(candidates, key=lambda item: item[0])
        if dist > tol:
            raise ValueError(
                f"Could not map atom {i} under symmetry operation; "
                f"best distance={dist}"
            )
        used.add(best)
        atom_map.append(best)
    return tuple(atom_map)


def _shell_map_for_atom_map(mol, atom_map) -> tuple[int, ...]:
    shells_by_atom = [[] for _ in range(int(mol.natm))]
    for ib in range(int(mol.nbas)):
        shells_by_atom[int(mol.bas_atom(ib))].append(ib)

    shell_map = []
    for ib in range(int(mol.nbas)):
        atom_i = int(mol.bas_atom(ib))
        atom_j = int(atom_map[atom_i])
        signature = _shell_signature(mol, ib)
        same_before = [
            kb for kb in shells_by_atom[atom_i]
            if kb <= ib and _shell_signature(mol, kb) == signature
        ]
        occurrence = len(same_before) - 1
        candidates = [
            jb for jb in shells_by_atom[atom_j]
            if _shell_signature(mol, jb) == signature
        ]
        if occurrence >= len(candidates):
            raise ValueError(
                f"Could not find matching shell for shell {ib} on atom {atom_j}"
            )
        shell_map.append(candidates[occurrence])
    return tuple(shell_map)


def _shell_signature(mol, ib: int):
    return (
        int(mol.bas_angular(ib)),
        int(mol.bas_nprim(ib)),
        int(mol.bas_nctr(ib)),
        tuple(np.round(np.asarray(mol.bas_exp(ib), dtype=float), 12)),
        tuple(np.round(np.asarray(mol.bas_ctr_coeff(ib), dtype=float).reshape(-1), 12)),
    )


def _shell_rotation_matrices(mol, rotation) -> list[np.ndarray]:
    from pyscf import lib
    from pyscf.symm.Dmatrix import Dmatrix, get_euler_angles

    lmax = max(int(mol.bas_angular(ib)) for ib in range(int(mol.nbas)))
    det = float(np.linalg.det(rotation))
    if det < 0:
        proper = -np.asarray(rotation)
    else:
        proper = np.asarray(rotation)
    alpha, beta, gamma = get_euler_angles(np.eye(3), proper)

    if not getattr(mol, "cart", False):
        matrices = []
        for l in range(lmax + 1):
            mat = Dmatrix(l, alpha, beta, gamma, reorder_p=True)
            if det < 0:
                mat = ((-1) ** l) * mat
            matrices.append(np.asarray(mat))
        return matrices

    pmat = Dmatrix(1, alpha, beta, gamma, reorder_p=True)
    if det < 0:
        pmat = -pmat
    matrices = [np.ones((1, 1))]
    for l in range(1, lmax + 1):
        cidx = np.sort(lib.cartesian_prod([(0, 1, 2)] * l), axis=1)
        addr = 0
        affine = np.ones((1, 1))
        for i in range(l):
            nd = affine.shape[0] * 3
            affine = np.einsum("ik,jl->ijkl", affine, pmat).reshape(nd, nd)
            addr = addr * 3 + cidx[:, i]
        uniq_addr, rev_addr = np.unique(addr, return_inverse=True)
        ncart = (l + 1) * (l + 2) // 2
        if ncart != uniq_addr.size:
            raise ValueError(f"Unexpected Cartesian shell size for l={l}")
        trans = np.zeros((ncart, ncart))
        for i, k in enumerate(rev_addr):
            trans[k] += affine[i, uniq_addr]
        matrices.append(trans)
    return matrices


def _character_table_from_libmsym(group_name: str, libmsym_table) -> CharacterTable:
    operation_labels = tuple(_operation_label(op) for op in libmsym_table.symmetry_operations)
    species = list(libmsym_table.symmetry_species)
    values = np.asarray(libmsym_table.table, dtype=float)
    irreps = {
        str(spec.name): tuple(values[i, :].tolist())
        for i, spec in enumerate(species)
    }
    return CharacterTable(
        group_name=group_name,
        operation_labels=operation_labels,
        class_counts=tuple(int(x) for x in libmsym_table.class_count),
        irreps=irreps,
    )


def _operation_label(operation) -> str:
    text = str(operation)
    if "( " in text and ", conjugacy class:" in text:
        return text.split("( ", 1)[1].split(", conjugacy class:", 1)[0].strip()
    return text


def _cartesian_matrix_from_libmsym(operation) -> np.ndarray:
    op_type = int(operation.type)
    if op_type == 0:
        return np.eye(3)
    if op_type == 1:
        return _rotation_matrix(np.asarray(operation.vector, dtype=float), operation.power, operation.order)
    if op_type == 2:
        axis = np.asarray(operation.vector, dtype=float)
        rotation = _rotation_matrix(axis, operation.power, operation.order)
        reflection = np.eye(3) - 2.0 * np.outer(_unit(axis), _unit(axis))
        return reflection @ rotation
    if op_type == 3:
        normal = _unit(np.asarray(operation.vector, dtype=float))
        return np.eye(3) - 2.0 * np.outer(normal, normal)
    if op_type == 4:
        return -np.eye(3)
    raise ValueError(f"Unsupported libmsym operation type: {op_type}")


def _rotation_matrix(axis: np.ndarray, power: int, order: int) -> np.ndarray:
    n = _unit(axis)
    angle = 2.0 * np.pi * float(power) / float(order)
    nx = np.array(
        [
            [0.0, -n[2], n[1]],
            [n[2], 0.0, -n[0]],
            [-n[1], n[0], 0.0],
        ]
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(n, n)
        + np.sin(angle) * nx
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1.0e-14:
        raise ValueError("Symmetry operation axis/normal has near-zero norm")
    return vector / norm
