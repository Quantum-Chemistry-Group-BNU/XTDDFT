# Excited-State Symmetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-pass independent point-group symmetry analyzer for high-spin reference states and XTDA, XSF_TDA_down, and SF_TDA_up excited-state vectors.

**Architecture:** Add one focused module, `XTDDFT/symmetry.py`, with method adapters, reference determinant character analysis, root grouping, character projection, and a public object-based analysis entry point. Keep expensive libmsym/AO-operation integration behind optional functions so unit tests can cover adapter and projection logic without requiring libmsym.

**Tech Stack:** Python, NumPy, PySCF objects, optional libmsym, existing XTDDFT result objects.

---

### Task 1: Root Grouping And Irrep Projection

**Files:**
- Create: `tests/test_symmetry_analysis.py`
- Create: `XTDDFT/symmetry.py`

- [ ] **Step 1: Write failing tests for root grouping and character projection**

```python
import numpy as np

from XTDDFT_dev.utils.symmetry import (
    CharacterTable,
    decompose_characters,
    group_degenerate_roots,
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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python -m pytest tests/test_symmetry_analysis.py -q
```

Expected: import fails because `XTDDFT_dev.utils.symmetry` does not exist.

- [ ] **Step 3: Implement minimal data types and functions**

Add `CharacterTable`, `group_degenerate_roots`, and `decompose_characters` in `XTDDFT/symmetry.py`.

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python -m pytest tests/test_symmetry_analysis.py -q
```

Expected: 2 tests pass.

### Task 2: Method Adapter Detection And Vector Blocks

**Files:**
- Modify: `tests/test_symmetry_analysis.py`
- Modify: `XTDDFT/symmetry.py`

- [ ] **Step 1: Write failing tests for XTDA, XSF_TDA_down, and SF_TDA_up adapters**

Use lightweight `types.SimpleNamespace` objects with the same attributes used by the real classes:

```python
from types import SimpleNamespace
import numpy as np

from XTDDFT_dev.utils.symmetry import make_method_adapter


def test_xtda_restricted_adapter_splits_spin_tensor_blocks():
    td = SimpleNamespace(
        __class__=SimpleNamespace(__name__="XTDA"),
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
```

- [ ] **Step 2: Run tests and verify adapter tests fail**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python -m pytest tests/test_symmetry_analysis.py -q
```

Expected: adapter imports or behavior fail because adapters are not implemented.

- [ ] **Step 3: Implement adapters**

Add `ExcitationBlock`, `BaseMethodAdapter`, `XtdaAdapter`, `SfTdaUpAdapter`, `XsfTdaDownAdapter`, and `make_method_adapter`.

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python -m pytest tests/test_symmetry_analysis.py -q
```

Expected: all adapter and projection tests pass.

### Task 3: Report Object And Public Entry Point

**Files:**
- Modify: `tests/test_symmetry_analysis.py`
- Modify: `XTDDFT/symmetry.py`

- [ ] **Step 1: Write failing tests for report formatting and libmsym failure message**

```python
import pytest

from XTDDFT_dev.utils.symmetry import (
    SymmetryAnalysisReport,
    analyze_excited_state_symmetry,
)


def test_report_format_text_contains_scope_warning_for_supercell():
    report = SymmetryAnalysisReport(
        group_name="C3v",
        finite_supercell=True,
        root_groups=[(0, 1)],
        assignments=[{"A1": 0.0, "E": 1.0}],
        approximate=False,
        notes=("density invariant",),
    )
    text = report.format_text()
    assert "C3v" in text
    assert "finite supercell point-group analysis" in text
    assert "roots 1,2" in text
    assert "E: 1.000000" in text


def test_analyze_excited_state_symmetry_requires_geometry_backend():
    with pytest.raises(ImportError, match="libmsym"):
        analyze_excited_state_symmetry(object())
```

- [ ] **Step 2: Run tests and verify report tests fail**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python -m pytest tests/test_symmetry_analysis.py -q
```

Expected: report class or API behavior is missing.

- [ ] **Step 3: Implement report and public entry point skeleton**

Add `SymmetryAnalysisReport` and `analyze_excited_state_symmetry`. The public entry point should validate the TD object, build root groups from `td_obj.e`, instantiate an adapter, then raise a clear libmsym `ImportError` until the geometry backend is implemented.

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python -m pytest tests/test_symmetry_analysis.py -q
```

Expected: all tests pass.

### Task 4: Reference Determinant Symmetry Helpers

**Files:**
- Modify: `tests/test_symmetry_analysis.py`
- Modify: `XTDDFT/symmetry.py`

- [ ] **Step 1: Write failing tests for determinant characters**

```python
import numpy as np

from XTDDFT_dev.utils.symmetry import determinant_characters


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
```

- [ ] **Step 2: Run tests and verify determinant tests fail**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python -m pytest tests/test_symmetry_analysis.py -q
```

Expected: `determinant_characters` import or behavior is missing.

- [ ] **Step 3: Implement determinant helper**

Add `determinant_characters(operation_matrices)` returning real-if-close determinant characters.

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python -m pytest tests/test_symmetry_analysis.py -q
```

Expected: all tests pass.

### Task 5: Package Import Check

**Files:**
- Modify: `XTDDFT/__init__.py` only if needed.

- [ ] **Step 1: Run import smoke test**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python - <<'PY'
from XTDDFT_dev.utils.symmetry import group_degenerate_roots, make_method_adapter
print(group_degenerate_roots([0.0, 1.0], tol=1e-5))
PY
```

Expected: prints `[(0,), (1,)]`.

- [ ] **Step 2: Run focused test suite**

Run:

```bash
/home/chang/soft/miniconda3/envs/clf_quimb/bin/python -m pytest tests/test_symmetry_analysis.py -q
```

Expected: all tests pass.

### Task 6: Follow-Up Geometry Backend

**Files:**
- Future modify: `XTDDFT/symmetry.py`
- Future create: `experiment/molecule/analyze_excited_state_symmetry.py`

- [ ] **Step 1: Integrate libmsym**

Discover installed Python API shape for libmsym in the target environment. Implement `GeometrySymmetry` only after confirming the available API.

- [ ] **Step 2: Add AO operation projection**

Use PySCF AO evaluation/overlap projection to build AO representation matrices for each Cartesian operation.

- [ ] **Step 3: Add end-to-end example**

Reconstruct a TD object from `chkfile + results.npz`, call `analyze_excited_state_symmetry`, and print the formatted report.
