from types import SimpleNamespace

import numpy as np
import pytest

from XTDDFT_dev.XTDDFT.symmetry import (
    CharacterTable,
    SymmetryAnalysisReport,
    analyze_excited_state_symmetry,
    build_ao_operation_matrices,
    build_mo_operation_matrices,
    decompose_characters,
    detect_geometry_symmetry,
    determinant_characters,
    group_degenerate_roots,
    make_method_adapter,
    MOOperationMatrices,
    _state_overlap_with_transformed,
)


def test_group_degenerate_roots_groups_close_energies():
    energies = np.array([0.10, 0.1000004, 0.25, 0.40, 0.4000002])
    assert group_degenerate_roots(energies, tol=1.0e-5) == [(0, 1), (2,), (3, 4)]


def test_decompose_characters_for_c2v_a1_representation():
    table = CharacterTable(
        group_name="C2v",
        operation_labels=("E", "C2", "sv_xz", "sv_yz"),
        irreps={
            "A1": (1, 1, 1, 1),
            "A2": (1, 1, -1, -1),
            "B1": (1, -1, 1, -1),
            "B2": (1, -1, -1, 1),
        },
    )
    weights = decompose_characters(np.array([1, 1, 1, 1]), table)
    assert weights["A1"] == 1.0
    assert weights["A2"] == 0.0
    assert weights["B1"] == 0.0
    assert weights["B2"] == 0.0


def test_decompose_characters_uses_conjugacy_class_counts():
    table = CharacterTable(
        group_name="C3v",
        operation_labels=("E", "2C3", "3sv"),
        class_counts=(1, 2, 3),
        irreps={
            "A1": (1, 1, 1),
            "A2": (1, 1, -1),
            "E": (2, -1, 0),
        },
    )
    weights = decompose_characters(np.array([2, -1, 0]), table)
    assert weights["A1"] == 0.0
    assert weights["A2"] == 0.0
    assert weights["E"] == 1.0


def test_detect_geometry_symmetry_extracts_libmsym_character_table():
    pytest.importorskip("libmsym")

    class FakeMol:
        natm = 3

        def atom_symbol(self, index):
            return ("O", "H", "H")[index]

        def atom_coords(self):
            return np.array(
                [
                    [0.000, 0.000, 0.000],
                    [0.758, 0.000, 0.504],
                    [-0.758, 0.000, 0.504],
                ]
            )

    geometry = detect_geometry_symmetry(FakeMol())
    assert geometry.point_group == "C2v"
    assert geometry.character_table.group_name == "C2v"
    assert set(geometry.character_table.irreps) == {"A1", "A2", "B1", "B2"}


def test_nh3_c3v_ao_operation_matrices_preserve_overlap():
    pytest.importorskip("libmsym")
    pytest.importorskip("pyscf")
    from pyscf import gto

    mol = gto.M(
        atom="""
        N   0.0000000000   0.0000000000   0.1160000000
        H   0.0000000000   0.9377000000  -0.2706666667
        H   0.8124219201  -0.4688500000  -0.2706666667
        H  -0.8124219201  -0.4688500000  -0.2706666667
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    geometry = detect_geometry_symmetry(mol)
    ao_ops = build_ao_operation_matrices(mol, geometry, grid_level=4)
    overlap = mol.intor_symmetric("int1e_ovlp")
    for op in ao_ops:
        assert np.linalg.norm(op.T @ overlap @ op - overlap) < 5.0e-3


def test_mo_operation_matrices_have_expected_spin_shapes_for_roks_nh3():
    pytest.importorskip("libmsym")
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    mol = gto.M(
        atom="""
        N   0.0000000000   0.0000000000   0.1160000000
        H   0.0000000000   0.9377000000  -0.2706666667
        H   0.8124219201  -0.4688500000  -0.2706666667
        H  -0.8124219201  -0.4688500000  -0.2706666667
        """,
        basis="sto-3g",
        unit="Angstrom",
        spin=2,
        verbose=0,
    )
    mf = dft.ROKS(mol)
    mf.xc = "lda,vwn"
    mf.kernel()
    geometry = detect_geometry_symmetry(mol)
    mo_ops = build_mo_operation_matrices(mf, geometry, grid_level=4)
    assert len(mo_ops.alpha) == len(geometry.symmetry_operation_labels)
    assert mo_ops.alpha[0].shape == (mf.mo_coeff.shape[1], mf.mo_coeff.shape[1])
    assert mo_ops.beta[0].shape == (mf.mo_coeff.shape[1], mf.mo_coeff.shape[1])


def test_nh3_c3v_xsf_and_xtda_reports_assign_reference_and_e_pairs():
    pytest.importorskip("libmsym")
    pytest.importorskip("pyscf")
    from pyscf import dft, gto, lib

    from XTDDFT_dev.utils.backend import set_backend
    from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down
    from XTDDFT_dev.XTDDFT.xtda import XTDA

    set_backend("cpu")
    lib.num_threads(2)
    mol = gto.M(
        atom="""
        N   0.0000000000   0.0000000000   0.1160000000
        H   0.0000000000   0.9377000000  -0.2706666667
        H   0.8124219201  -0.4688500000  -0.2706666667
        H  -0.8124219201  -0.4688500000  -0.2706666667
        """,
        basis="sto-3g",
        unit="Angstrom",
        spin=2,
        verbose=0,
    )
    mf = dft.ROKS(mol)
    mf.xc = "lda,vwn"
    mf.kernel()
    assert mf.converged

    xsf = XSF_TDA_down(mf, method=0, SA=3, davidson=False)
    xsf.kernel(nstates=4, remove=None, save=False)
    xsf_report = analyze_excited_state_symmetry(
        xsf, energy_tol=1.0e-4, symmetry_tol=1.0e-3, grid_level=4
    )
    assert xsf_report.group_name == "C3v"
    assert xsf_report.reference_assignment["A1"] > 0.99
    assert xsf_report.assignments[2]["E1"] > 0.99

    xtda = XTDA(mf, method=0, davidson=False, so2st=False)
    xtda.kernel(nstates=4, save=False)
    xtda_report = analyze_excited_state_symmetry(
        xtda, energy_tol=1.0e-2, symmetry_tol=1.0e-3, grid_level=4
    )
    assert xtda_report.group_name == "C3v"
    assert xtda_report.reference_assignment["A1"] > 0.99
    assert xtda_report.assignments[0]["E1"] > 0.99


def test_xtda_restricted_adapter_splits_spin_tensor_blocks():
    td = SimpleNamespace(
        type_u=False,
        nc=1,
        no=2,
        nv=3,
        v=np.arange(14, dtype=float).reshape(14, 1),
    )
    adapter = make_method_adapter(td, method_name="XTDA")
    blocks = adapter.state_blocks(0)
    assert [block.name for block in blocks] == ["CV(0)", "CO(0)", "OV(0)", "CV(1)"]
    assert blocks[0].amplitudes.shape == (1, 3)
    assert blocks[1].amplitudes.shape == (1, 2)
    assert blocks[2].amplitudes.shape == (2, 3)
    assert blocks[3].amplitudes.shape == (1, 3)


def test_sf_tda_up_adapter_splits_single_block():
    td = SimpleNamespace(nc=2, nv=3, v=np.arange(6, dtype=float).reshape(6, 1))
    adapter = make_method_adapter(td, method_name="SF_TDA_up")
    blocks = adapter.state_blocks(0)
    assert [block.name for block in blocks] == ["b2a"]
    assert blocks[0].amplitudes.shape == (2, 3)


def test_xsf_tda_down_adapter_expands_removed_oo_block():
    vects = np.eye(4)[:, :3]
    td = SimpleNamespace(
        nc=1,
        no=2,
        nv=1,
        re=True,
        vects=vects,
        v=np.arange(8, dtype=float).reshape(8, 1),
    )
    adapter = make_method_adapter(td, method_name="XSF_TDA_down")
    blocks = adapter.state_blocks(0)
    assert [block.name for block in blocks] == ["CV", "CO", "OV", "OO"]
    assert blocks[3].amplitudes.shape == (2, 2)


def test_unrestricted_xsf_down_projection_uses_beta_virtual_rotation():
    XSF_TDA_down = type("XSF_TDA_down", (), {})
    td = XSF_TDA_down()
    td.nc = 1
    td.no = 0
    td.nv = 1
    td.re = False
    td.type_u = True
    td.v = np.array([[1.0]])
    mo_ops = MOOperationMatrices(
        alpha=(np.array([[1.0, 0.0], [0.0, 1.0]]),),
        beta=(np.array([[1.0, 0.0], [0.0, -1.0]]),),
        ao=(np.eye(2),),
    )
    adapter = make_method_adapter(td, method_name="XSF_TDA_down")
    value = _state_overlap_with_transformed(td, adapter, 0, mo_ops, 0)
    assert value == -1.0


def test_report_format_text_contains_scope_warning_for_supercell():
    report = SymmetryAnalysisReport(
        group_name="C3v",
        finite_supercell=True,
        root_groups=[(0, 1)],
        assignments=[{"A1": 0.0, "E": 1.0}],
        reference_assignment={"A1": 0.0, "E": 1.0},
        approximate=False,
        notes=("density invariant",),
    )
    text = report.format_text()
    assert "C3v" in text
    assert "finite supercell point-group analysis" in text
    assert "reference" in text
    assert "roots 1,2" in text
    assert "E: 1.000000" in text


def test_analyze_excited_state_symmetry_requires_td_object_with_mf():
    with pytest.raises(TypeError, match="must expose .mf"):
        analyze_excited_state_symmetry(object())


def test_determinant_characters_for_single_open_shell_b1():
    operations = [
        np.array([[1.0]]),
        np.array([[-1.0]]),
        np.array([[1.0]]),
        np.array([[-1.0]]),
    ]
    chars = determinant_characters(operations)
    assert np.allclose(chars, [1.0, -1.0, 1.0, -1.0])


def test_determinant_characters_for_two_dimensional_open_shell_wedge():
    operations = [
        np.eye(2),
        np.diag([1.0, -1.0]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    ]
    chars = determinant_characters(operations)
    assert np.allclose(chars, [1.0, -1.0, -1.0])
