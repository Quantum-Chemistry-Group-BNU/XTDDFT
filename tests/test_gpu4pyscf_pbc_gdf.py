#!/usr/bin/env python3
"""Manual GPU4PySCF PBC GDF/FFTDF smoke test.

Run from the project root:

    python tests/test_gpu4pyscf_pbc_gdf.py

This script is intentionally standalone rather than a pytest test.  It prints
environment details and full tracebacks so it can be copied to another server
to diagnose whether GPU4PySCF PBC GDF/int3c2e works there.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_cell():
    import pyscf

    return pyscf.M(
        atom="""
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
""",
        basis="gth-szv",
        pseudo="gth-pade",
        a="""
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000""",
        unit="B",
        verbose=0,
    )


def print_environment() -> None:
    print("== Environment ==")
    print("python", sys.version.replace("\n", " "))
    print("executable", sys.executable)
    print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
    try:
        import numpy
        import scipy
        import pyscf
        import cupy
        import gpu4pyscf

        print("numpy", numpy.__version__)
        print("scipy", scipy.__version__)
        print("pyscf", pyscf.__version__)
        print("cupy", cupy.__version__)
        print("gpu4pyscf", getattr(gpu4pyscf, "__version__", "unknown"))
        print("cuda runtime", cupy.cuda.runtime.runtimeGetVersion())
        print("cuda driver", cupy.cuda.runtime.driverGetVersion())
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print("device", name, "cc", f"{props['major']}.{props['minor']}")
    except Exception:
        traceback.print_exc()
    print()


def check_cupy_fft() -> bool:
    print("== CuPy FFT check ==")
    try:
        import cupy as cp

        x = cp.zeros((47, 47, 47), dtype=cp.complex128)
        y = cp.fft.ifftn(x)
        cp.cuda.Stream.null.synchronize()
        print("OK", y.shape)
        return True
    except Exception:
        traceback.print_exc()
        return False
    finally:
        print()


def run_case(name, factory, test_project_fock=True) -> bool:
    print(f"== {name} ==")
    try:
        from utils.backend import asnumpy, set_backend
        from XTDDFT.base import _get_mo_fock, mf_info

        set_backend("gpu")
        mf = factory()
        print("mf", mf.__class__)
        print("with_df", getattr(mf, "with_df", None).__class__)
        print("converged", getattr(mf, "converged", None))
        print("e_tot", getattr(mf, "e_tot", None))

        dm = mf.make_rdm1()
        print("dm", getattr(dm, "shape", None), type(dm).__module__)

        try:
            veff = mf.get_veff(mf.cell, dm, kpt=getattr(mf, "kpt", None))
        except TypeError:
            veff = mf.get_veff(mf.cell, dm, kpts=getattr(mf, "kpts", None))
        print("get_veff OK", getattr(veff, "shape", None), type(veff).__module__)

        if test_project_fock:
            mo_energy, mo_occ, mo_coeff = mf_info(mf)
            fa, fb = _get_mo_fock(mf, mo_coeff, mo_occ)
            print("_get_mo_fock OK", fa.shape, fb.shape, type(fa).__module__)
            print("diag alpha", asnumpy(fa).diagonal()[:8])
        print()
        return True
    except Exception:
        traceback.print_exc()
        print()
        return False


def main() -> int:
    print_environment()
    check_cupy_fft()

    cell = make_cell()
    print("== Cell ==")
    print("nao", cell.nao_nr())
    print("mesh", cell.mesh)
    print("volume", cell.vol)
    print()

    cases = [
        (
            "FFTDF RKS gamma to_gpu run",
            lambda: cell.RKS(xc="pbe").to_gpu().run(),
            True,
        ),
        (
            "GDF RKS gamma CPU run then to_gpu",
            lambda: cell.RKS(xc="pbe").density_fit().run().to_gpu(),
            True,
        ),
        (
            "GDF RKS gamma to_gpu run",
            lambda: cell.RKS(xc="pbe").density_fit().to_gpu().run(),
            False,
        ),
        (
            "GDF RHF gamma CPU run then to_gpu",
            lambda: cell.RHF(exxdiv=None).density_fit().run().to_gpu(),
            True,
        ),
        (
            "GDF RHF gamma to_gpu run",
            lambda: cell.RHF(exxdiv=None).density_fit().to_gpu().run(),
            False,
        ),
    ]

    results = []
    for name, factory, test_project_fock in cases:
        results.append((name, run_case(name, factory, test_project_fock)))

    print("== Summary ==")
    for name, ok in results:
        print(("OK   " if ok else "FAIL "), name)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

