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


@dataclass(frozen=True)
class MOOperationMatrices:
    """Spin-resolved MO representation matrices for symmetry operations."""

    alpha: tuple[np.ndarray, ...]
    beta: tuple[np.ndarray, ...]
    ao: tuple[np.ndarray, ...]


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

    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"))
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


def build_mo_operation_matrices(
    mf,
    geometry: GeometrySymmetry,
    grid_level: int = 5,
    grid_coords=None,
    grid_weights=None,
    grid_block_size: int = 20000,
):
    """Build spin-resolved MO representation matrices from AO operations."""

    system = getattr(mf, "mol", None)
    if system is None:
        system = getattr(mf, "cell", None)
    if system is None:
        raise TypeError("mf must expose either .mol or .cell")
    ao_ops = build_ao_operation_matrices(
        system,
        geometry,
        grid_level=grid_level,
        grid_coords=grid_coords,
        grid_weights=grid_weights,
        grid_block_size=grid_block_size,
    )
    overlap = np.asarray(system.intor_symmetric("int1e_ovlp"))
    coeff_alpha, coeff_beta = _spin_mo_coefficients(mf)
    alpha = tuple(coeff_alpha.conj().T @ overlap @ op @ coeff_alpha for op in ao_ops)
    beta = tuple(coeff_beta.conj().T @ overlap @ op @ coeff_beta for op in ao_ops)
    return MOOperationMatrices(alpha=alpha, beta=beta, ao=ao_ops)


def analyze_excited_state_symmetry(
    td_obj,
    energy_tol: float = 1.0e-5,
    symmetry_tol: float = 1.0e-5,
    point_group: str | None = None,
    grid_level: int = 5,
    grid_coords=None,
    grid_weights=None,
    grid_block_size: int = 20000,
):
    """Analyze reference and excited-state symmetry for a TD object.

    This first complete path uses numerical AO projection on PySCF grids.  It
    is intended for molecular and finite Gamma-supercell cluster analyses.
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

    geometry = detect_geometry_symmetry(
        system,
        threshold=symmetry_tol,
        point_group=point_group,
    )
    mo_ops = build_mo_operation_matrices(
        mf,
        geometry,
        grid_level=grid_level,
        grid_coords=grid_coords,
        grid_weights=grid_weights,
        grid_block_size=grid_block_size,
    )
    energies = np.asarray(getattr(td_obj, "e"), dtype=float).reshape(-1)
    groups = group_degenerate_roots(energies, tol=energy_tol)
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

    notes = []
    max_ao_error = max(
        np.linalg.norm(op.T @ np.asarray(system.intor_symmetric("int1e_ovlp")) @ op
                       - np.asarray(system.intor_symmetric("int1e_ovlp")))
        for op in mo_ops.ao
    )
    notes.append(f"max AO overlap invariance error: {max_ao_error:.3e}")
    return SymmetryAnalysisReport(
        group_name=geometry.point_group,
        finite_supercell=finite_supercell,
        root_groups=groups,
        assignments=assignments,
        reference_assignment=reference_assignment,
        open_shell_assignment=open_shell_assignment,
        approximate=max_ao_error > symmetry_tol,
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


def _excited_subspace_operation_characters(td_obj, mo_ops: MOOperationMatrices, group):
    adapter = make_method_adapter(td_obj)
    chars = []
    for op_index in range(len(mo_ops.alpha)):
        total = 0.0j
        for state in group:
            total += _state_overlap_with_transformed(td_obj, adapter, state, mo_ops, op_index)
        chars.append(total)
    return np.real_if_close(np.asarray(chars), tol=1000)


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
