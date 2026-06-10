#!/usr/bin/env python3
"""Server-side GPU4PySCF PBC GDF smoke test for a small solid-carbon cell.

Run on the GPU server from the project root:

    /path/to/XSFTDA/bin/python tests/test_gpu4pyscf_pbc_c_solid_gdf.py

The script intentionally prints full diagnostics and tracebacks.  It does not
depend on XTDDFT internals; it isolates whether GPU4PySCF PBC GDF can build
CDERI, stream CDERI blocks, build range-separated DF objects, and run a small
JK contraction for a gamma-point periodic carbon cell.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT.df_cderi_cache import (
    attach_pbc_gamma_cderi_cache,
    build_pbc_gamma_cderi_cache,
)
from XTDDFT_dev.XTDDFT.pbc_cderi_outcore import build_pbc_gamma_cderi_cache_outcore


def make_diamond_carbon_cell():
    from pyscf.pbc import gto

    cell = gto.Cell()
    cell.atom = """
C 0.000000000000 0.000000000000 0.000000000000
C 0.891700000000 0.891700000000 0.891700000000
"""
    cell.a = """
0.000000000000 1.783400000000 1.783400000000
1.783400000000 0.000000000000 1.783400000000
1.783400000000 1.783400000000 0.000000000000
"""
    cell.unit = "A"
    cell.basis = "cc-pvdz"
    cell.verbose = 4
    cell.build()
    return cell


def print_environment():
    print("== Environment ==")
    print("python", sys.version.replace("\n", " "))
    print("executable", sys.executable)
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
    try:
        import cupy as cp
        import gpu4pyscf
        import pyscf

        print("pyscf", pyscf.__version__)
        print("cupy", cp.__version__)
        print("gpu4pyscf", getattr(gpu4pyscf, "__version__", "unknown"))
        print("cuda runtime", cp.cuda.runtime.runtimeGetVersion())
        print("cuda driver", cp.cuda.runtime.driverGetVersion())
        count = cp.cuda.runtime.getDeviceCount()
        print("cuda device count", count)
        for idev in range(count):
            props = cp.cuda.runtime.getDeviceProperties(idev)
            name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            print("device", idev, name, "cc", f"{props['major']}.{props['minor']}")
    except Exception:
        traceback.print_exc()
    print()


def describe_df(dfobj, label):
    print(f"== {label} ==")
    print("df class", dfobj.__class__)
    print("is_gamma_point", getattr(dfobj, "is_gamma_point", None))
    print("auxbasis", getattr(dfobj, "auxbasis", None))
    print("mesh", getattr(getattr(dfobj, "cell", None), "mesh", None))
    print("_omega", getattr(dfobj, "_omega", None))
    print("_cderi is None", getattr(dfobj, "_cderi", None) is None)
    if getattr(dfobj, "_cderi", None) is not None:
        print("_cderi keys", list(dfobj._cderi.keys()))
        for key, value in dfobj._cderi.items():
            print("  _cderi", key, "shape", getattr(value, "shape", None), "type", type(value))
    if getattr(dfobj, "_cderi_idx", None) is not None:
        print("_cderi_idx shapes", [getattr(x, "shape", None) for x in dfobj._cderi_idx])
    print()


def test_loop_gamma_point(dfobj, blksize=8, max_blocks=4):
    import cupy as cp

    print(f"== loop_gamma_point blksize={blksize} ==")
    nblocks = 0
    total_aux = 0
    for cderi, cderi_sparse, sign in dfobj.loop_gamma_point(blksize=blksize, unpack=True):
        nblocks += 1
        total_aux += int(cderi.shape[0])
        print(
            "block", nblocks,
            "sign", sign,
            "cderi", cderi.shape,
            "sparse", cderi_sparse.shape,
            "norm", float(cp.linalg.norm(cderi).get()),
        )
        cp.cuda.Stream.null.synchronize()
        if nblocks >= max_blocks:
            break
    print("reported_blocks", nblocks, "reported_aux_rows", total_aux)
    print()


def collect_loop_gamma_norms(dfobj, blksize=8, max_blocks=4):
    import cupy as cp

    out = []
    for cderi, cderi_sparse, sign in dfobj.loop_gamma_point(blksize=blksize, unpack=True):
        out.append((
            sign,
            tuple(cderi.shape),
            tuple(cderi_sparse.shape),
            float(cp.linalg.norm(cderi).get()),
            float(cp.linalg.norm(cderi_sparse).get()),
        ))
        cp.cuda.Stream.null.synchronize()
        if len(out) >= max_blocks:
            break
    return out


def test_get_jk(dfobj, nao):
    import cupy as cp

    print("== get_jk identity-density smoke ==")
    dm = np.eye(nao, dtype=np.float64)
    vj, vk = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
    cp.cuda.Stream.null.synchronize()
    print("vj shape", getattr(vj, "shape", None), "type", type(vj), "norm", float(cp.linalg.norm(vj).get()))
    print("vk shape", getattr(vk, "shape", None), "type", type(vk), "norm", float(cp.linalg.norm(vk).get()))
    print()


def compare_get_jk(df_a, df_b, nao, label):
    import cupy as cp

    print(f"== compare get_jk {label} ==")
    dm = np.eye(nao, dtype=np.float64)
    vj_a, vk_a = df_a.get_jk(dm, hermi=1, with_j=True, with_k=True)
    vj_b, vk_b = df_b.get_jk(dm, hermi=1, with_j=True, with_k=True)
    cp.cuda.Stream.null.synchronize()
    dvj = float(cp.linalg.norm(vj_a - vj_b).get())
    dvk = float(cp.linalg.norm(vk_a - vk_b).get())
    print("delta vj norm", dvj)
    print("delta vk norm", dvk)
    if dvj > 1e-8 or dvk > 1e-8:
        raise RuntimeError(f"cached {label} get_jk mismatch: dJ={dvj}, dK={dvk}")
    print()


def test_cached_pbc_gamma_df(cell, source_df, label):
    print(f"== HDF5 cache attach {label} ==")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / f"{label.replace(' ', '_')}.h5"
        info = build_pbc_gamma_cderi_cache(source_df, cache_path)
        print("cache path", cache_path)
        print("cache naux", info.naux, "nao_pair", info.nao_pair, "dataset", info.dataset)

        cached_mf = cell.RKS(xc="pbe").density_fit().to_gpu()
        cached_mf.with_df.is_gamma_point = True
        handle = attach_pbc_gamma_cderi_cache(cached_mf.with_df, cache_path)
        try:
            describe_df(cached_mf.with_df, f"Cached {label}")
            ref_norms = collect_loop_gamma_norms(source_df, blksize=8, max_blocks=4)
            cached_norms = collect_loop_gamma_norms(cached_mf.with_df, blksize=8, max_blocks=4)
            print("reference loop norms", ref_norms)
            print("cached loop norms", cached_norms)
            if len(ref_norms) != len(cached_norms):
                raise RuntimeError(f"cached {label} loop_gamma_point block count mismatch")
            for i, (ref, cached) in enumerate(zip(ref_norms, cached_norms), start=1):
                if ref[:3] != cached[:3] or abs(ref[3] - cached[3]) > 1e-10 or abs(ref[4] - cached[4]) > 1e-10:
                    raise RuntimeError(f"cached {label} loop_gamma_point block {i} mismatch")
            compare_get_jk(source_df, cached_mf.with_df, cell.nao_nr(), label)
        finally:
            handle.close()


def test_outcore_cached_pbc_gamma_df(cell, source_df, label):
    print(f"== out-of-core HDF5 cache build {label} ==")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / f"{label.replace(' ', '_')}_outcore.h5"

        outcore_mf = cell.RKS(xc="pbe").density_fit().to_gpu()
        outcore_mf.with_df.is_gamma_point = True
        info = build_pbc_gamma_cderi_cache_outcore(outcore_mf.with_df, cache_path)
        print("outcore cache path", cache_path)
        print("outcore cache naux", info.naux, "nao_pair", info.nao_pair, "dataset", info.dataset)

        import h5py

        with h5py.File(cache_path, "r") as handle:
            print("outcore builder attr", handle["pbc_gamma"].attrs.get("builder"))
            if handle["pbc_gamma"].attrs.get("builder") != "xtddft_outcore_column_blocks":
                raise RuntimeError("outcore cache did not record the xtddft outcore builder")

        cached_mf = cell.RKS(xc="pbe").density_fit().to_gpu()
        cached_mf.with_df.is_gamma_point = True
        handle = attach_pbc_gamma_cderi_cache(cached_mf.with_df, cache_path)
        try:
            describe_df(cached_mf.with_df, f"Outcore cached {label}")
            ref_norms = collect_loop_gamma_norms(source_df, blksize=8, max_blocks=4)
            cached_norms = collect_loop_gamma_norms(cached_mf.with_df, blksize=8, max_blocks=4)
            print("reference loop norms", ref_norms)
            print("outcore cached loop norms", cached_norms)
            if len(ref_norms) != len(cached_norms):
                raise RuntimeError(f"outcore cached {label} loop_gamma_point block count mismatch")
            for i, (ref, cached) in enumerate(zip(ref_norms, cached_norms), start=1):
                if ref[:3] != cached[:3] or abs(ref[3] - cached[3]) > 1e-10 or abs(ref[4] - cached[4]) > 1e-10:
                    raise RuntimeError(f"outcore cached {label} loop_gamma_point block {i} mismatch")
            compare_get_jk(source_df, cached_mf.with_df, cell.nao_nr(), f"outcore {label}")
        finally:
            handle.close()


def test_rsh_df(cell, base_df, omega):
    print(f"== range_coulomb omega={omega} ==")
    with base_df.range_coulomb(omega) as rsh_df:
        rsh_df.is_gamma_point = True
        rsh_df.build()
        describe_df(rsh_df, f"RSH DF omega={omega}")
        test_loop_gamma_point(rsh_df, blksize=8, max_blocks=2)
        test_cached_pbc_gamma_df(cell, rsh_df, "rsh_omega_minus_0p11")


def main() -> int:
    print_environment()
    try:
        import cupy as cp

        cell = make_diamond_carbon_cell()
        print("== Cell ==")
        print("nao", cell.nao_nr())
        print("nelectron", cell.nelectron)
        print("mesh", cell.mesh)
        print("volume", cell.vol)
        print()

        mf = cell.RKS(xc="pbe").density_fit().to_gpu()
        mf.with_df.is_gamma_point = True
        mf.with_df.build()
        cp.cuda.Stream.null.synchronize()

        describe_df(mf.with_df, "Base gamma GDF")
        test_loop_gamma_point(mf.with_df, blksize=8, max_blocks=4)
        test_get_jk(mf.with_df, cell.nao_nr())
        test_cached_pbc_gamma_df(cell, mf.with_df, "base_gamma")
        test_outcore_cached_pbc_gamma_df(cell, mf.with_df, "base_gamma")
        test_rsh_df(cell, mf.with_df, omega=-0.11)
        print("SUCCESS")
        return 0
    except Exception:
        traceback.print_exc()
        print("FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
