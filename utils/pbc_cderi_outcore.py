"""Experimental out-of-core PBC gamma CDERI builder.

This module is intentionally separate from ``df_cderi_cache`` because it uses
GPU4PySCF private builder APIs.  The stable part is the HDF5 writer: it accepts
AO-pair CDERI blocks and writes them directly to disk without ever assembling
the full three-center tensor in host memory.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from .backend import _asnumpy
from .df_cderi_cache import CderiCacheInfo


def _as_path(path) -> Path:
    return Path(path).expanduser()


def _create_cderi_dataset(h5g, name: str, shape, dtype):
    naux, nao_pair = map(int, shape)
    if naux <= 0 or nao_pair <= 0:
        raise ValueError("CDERI dataset shape must be positive")
    # GPU4PySCF consumes AO-pair column blocks, so keep chunks column-oriented.
    chunk_cols = min(nao_pair, 1024)
    chunk_rows = min(naux, max(1, 1024 * 1024 // max(chunk_cols, 1)))
    return h5g.create_dataset(
        name,
        shape=(naux, nao_pair),
        dtype=np.dtype(dtype),
        chunks=(chunk_rows, chunk_cols),
    )


def write_pbc_gamma_cderi_cache_from_blocks(
    gdf,
    path,
    *,
    cderi_blocks: Iterable[tuple[int, int, object]],
    cderi_shape: tuple[int, int],
    cderi_idx,
    group: str = "pbc_gamma",
    overwrite: bool = True,
    omega=None,
    auxbasis_repr: str | None = None,
) -> CderiCacheInfo:
    """Write gamma-point CDERI AO-pair blocks to an attachable HDF5 cache.

    ``cderi_blocks`` yields ``(p0, p1, block)`` where ``block`` has shape
    ``(naux, p1-p0)``.  Blocks are written immediately and then discarded by the
    caller, which is the key property needed to remove the full-CDERI memory
    peak during cache generation.
    """

    cache_path = _as_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        if not overwrite:
            from .df_cderi_cache import pbc_gamma_cderi_cache_info

            return pbc_gamma_cderi_cache_info(cache_path, group=group)
        cache_path.unlink()

    pair_idx, diag_idx = cderi_idx
    pair_idx = _asnumpy(pair_idx).astype(np.int64, copy=False)
    diag_idx = _asnumpy(diag_idx).astype(np.int64, copy=False)
    expected_cols = int(cderi_shape[1])
    written = np.zeros(expected_cols, dtype=bool)
    dtype = None

    with h5py.File(cache_path, "w") as handle:
        h5g = handle.create_group(group)
        h5d = None
        for p0, p1, block in cderi_blocks:
            p0, p1 = int(p0), int(p1)
            if p0 < 0 or p1 <= p0 or p1 > expected_cols:
                raise ValueError(f"invalid AO-pair block range ({p0}, {p1})")
            block = _asnumpy(block)
            if block.shape != (int(cderi_shape[0]), p1 - p0):
                raise ValueError(
                    f"CDERI block shape {block.shape} does not match "
                    f"expected {(int(cderi_shape[0]), p1 - p0)}"
                )
            if h5d is None:
                dtype = block.dtype
                h5d = _create_cderi_dataset(h5g, "cderi_0", cderi_shape, dtype)
            h5d[:, p0:p1] = block
            written[p0:p1] = True

        if h5d is None:
            raise ValueError("no CDERI blocks were provided")
        if not bool(np.all(written)):
            missing = np.flatnonzero(~written)
            raise ValueError(f"CDERI blocks did not cover all AO-pair columns; first missing column {missing[0]}")

        h5g.create_dataset("cderi_idx_pair", data=pair_idx)
        h5g.create_dataset("cderi_idx_diag", data=diag_idx)
        h5g.attrs["format"] = "gpu4pyscf_pbc_gamma_cderi"
        h5g.attrs["builder"] = "xtddft_outcore_column_blocks"
        h5g.attrs["is_gamma_point"] = True
        h5g.attrs["nao"] = int(getattr(gdf, "nao", getattr(getattr(gdf, "cell", None), "nao", 0)))
        h5g.attrs["omega"] = float(0.0 if omega is None else omega)
        h5g.attrs["auxbasis"] = auxbasis_repr if auxbasis_repr is not None else repr(getattr(gdf, "auxbasis", None))

    from .df_cderi_cache import pbc_gamma_cderi_cache_info

    return pbc_gamma_cderi_cache_info(cache_path, group=group)


def _require_gpu4pyscf_rsdf_builder():
    try:
        return importlib.import_module("gpu4pyscf.pbc.df.rsdf_builder")
    except Exception as err:
        raise RuntimeError(
            "GPU4PySCF PBC RSDF builder cannot be imported in this environment. "
            "Run this out-of-core builder on the GPU4PySCF server environment."
        ) from err


def _call_gpu4pyscf_outcore_hook(rsdf_builder, gdf, path, group, overwrite):
    hook = getattr(rsdf_builder, "compressed_cderi_j_only_outcore", None)
    if hook is None:
        hook = getattr(rsdf_builder, "build_cderi_outcore", None)
    if hook is None:
        return None
    return hook(gdf, path, group=group, overwrite=overwrite)


def _make_auxcell_for_gdf(gdf):
    auxcell = getattr(gdf, "auxcell", None)
    if auxcell is not None:
        return auxcell
    from pyscf.pbc.df import df as df_cpu

    return df_cpu.make_auxcell(gdf.cell, gdf.auxbasis, getattr(gdf, "exp_to_discard", None))


def _iter_gpu4pyscf_gamma_cderi_blocks(gdf, rsdf_builder):
    """Yield GPU4PySCF gamma CDERI AO-pair blocks.

    This is adapted from GPU4PySCF 1.6.1 ``compressed_cderi_j_only``.  The
    material change is that each ``j3c`` AO-pair block is yielded immediately
    instead of being stored into one full ``empty_mapped((naux, nao_pairs))``.
    """

    cell = gdf.cell
    base_auxcell = _make_auxcell_for_gdf(gdf)
    auxcell = base_auxcell
    kmesh = np.array([1, 1, 1])
    omega = abs(float(getattr(gdf, "_omega", getattr(cell, "omega", 0.0)) or 0.0))
    linear_dep_threshold = getattr(gdf, "linear_dep_threshold", rsdf_builder.LINEAR_DEP_THR)

    log = rsdf_builder.logger.new_logger(cell)
    t1 = log.init_timer()
    rsdf_omega = max(omega, rsdf_builder._guess_omega(cell))
    log.debug("omega = %g, rsdf_builder omega = %g", omega, rsdf_omega)

    int3c2e_opt = rsdf_builder.SRInt3c2eOpt(
        cell, auxcell, omega=-rsdf_omega, bvk_kmesh=kmesh
    ).build()
    cell = int3c2e_opt.cell
    auxcell = int3c2e_opt.auxcell
    bvk_ncells = len(int3c2e_opt.bvkmesh_Ls)

    log.debug("Generate auxcell 2c2e integrals")
    cd_j2c_cache, negative_metric_size = rsdf_builder._precontract_j2c_aux_coeff(
        auxcell, None, omega, rsdf_omega, linear_dep_threshold
    )
    if negative_metric_size:
        raise NotImplementedError(
            "Out-of-core PBC gamma CDERI generation does not yet support negative metric blocks."
        )
    naux_cart, naux = cd_j2c_cache[0].shape

    cderi_idx = int3c2e_opt.pair_and_diag_indices()
    nao_pairs = len(cderi_idx[0])

    with_long_range = omega < rsdf_omega
    if with_long_range:
        mesh = int3c2e_opt.mesh
    else:
        if cell.dimension != 3:
            raise NotImplementedError("Short-range-only out-of-core CDERI requires a 3D cell.")
        mesh = [1] * 3
    coulG = rsdf_builder._weighted_coulG_kpts(auxcell, mesh, omega, rsdf_omega)
    Gv = cell.get_Gv(mesh)
    ngrids = len(Gv)

    mem_free = rsdf_builder.get_avail_mem(exclude_memory_pool=True)
    mem_free -= cd_j2c_cache[0].nbytes
    mem_free -= ngrids * naux * 16
    batch_size = int(min(nao_pairs, mem_free // (naux_cart * bvk_ncells * 16 * 4)))
    if batch_size <= 0:
        raise RuntimeError("Insufficient GPU memory for one CDERI AO-pair batch")
    log.debug("Avail GPU mem = %s GB. batch_size = %d", mem_free * 1e-9, batch_size)
    log.debug(
        "Out-of-core CDERI target size %.6g GB on host HDF5",
        naux * nao_pairs * 8e-9,
    )

    nsp_per_block = rsdf_builder.ft_ao.ft_ao_scheme()[0]
    bas_ij_aggregated = cell.aggregate_shl_pairs(int3c2e_opt.bas_ij_cache, nsp_per_block)

    eval_j3c, aux_sorting, ao_pair_offsets = int3c2e_opt.int3c2e_evaluator(
        ao_pair_batch_size=batch_size, bas_ij_aggregated=bas_ij_aggregated
    )[:3]
    del aux_sorting
    shl_pair_batches = len(ao_pair_offsets) - 1
    aux_coeff = rsdf_builder.cp.asarray(cd_j2c_cache[0])

    ft_opt = rsdf_builder.ft_ao.FTOpt.from_intopt(int3c2e_opt)
    eval_ft, ft_ao_pair_offsets = ft_opt.ft_evaluator(
        batch_size, bas_ij_aggregated=bas_ij_aggregated
    )
    if not np.array_equal(ao_pair_offsets, ft_ao_pair_offsets):
        raise RuntimeError("GPU4PySCF AO-pair offsets mismatch between SR and FT evaluators")

    auxG_conj = rsdf_builder.ft_ao.ft_ao(auxcell, Gv).T.conj()
    auxG_conj = aux_coeff.T.dot(auxG_conj)
    auxG_conj *= rsdf_builder.cp.asarray(coulG)

    avail_mem = mem_free - naux_cart * batch_size * 16 * 2
    Gblksize = int(avail_mem // (16 * (batch_size + naux * 2))) // 32 * 32
    if Gblksize == 0:
        raise RuntimeError("Insufficient GPU memory for one reciprocal-space CDERI block")
    Gblksize = min(Gblksize, ngrids)
    log.debug1(
        "ngrids = %d Gblksize = %d naux=%d max_pair_size=%d",
        ngrids,
        Gblksize,
        naux,
        batch_size,
    )

    cp = rsdf_builder.cp
    buf2 = cp.empty(batch_size * Gblksize, dtype=np.complex128)
    buf0 = cp.empty(naux * batch_size)
    buf1 = cp.empty(batch_size * naux_cart * bvk_ncells, dtype=np.complex128)

    metadata = {
        "cderi_shape": (int(naux), int(nao_pairs)),
        "cderi_idx": cderi_idx,
        "omega": float(getattr(gdf, "_omega", getattr(gdf.cell, "omega", 0.0)) or 0.0),
        "auxcell": base_auxcell,
    }

    def blocks():
        nonlocal t1
        for batch_id in range(shl_pair_batches):
            log.debug1("batch %d/%d", batch_id, shl_pair_batches)
            j3c = eval_j3c(shl_pair_batch_id=batch_id, out=buf1)
            if j3c.size == 0:
                continue

            pair_size = j3c.shape[0]
            j3c_buf = rsdf_builder.ndarray((naux, pair_size), buffer=buf0)
            j3c = aux_coeff.T.dot(j3c.sum(axis=1).T, out=j3c_buf)
            t1 = log.timer_debug1("sr int3c2e", *t1)

            j3c_buf = rsdf_builder.ndarray(j3c.shape, dtype=np.complex128, buffer=buf1)
            for p0, p1 in rsdf_builder.lib.prange(0, ngrids, Gblksize):
                auxG_c = rsdf_builder.asarray(auxG_conj[:, p0:p1])
                pqG = eval_ft(Gv[p0:p1], batch_id, out=buf2)
                j3c += auxG_c.dot(pqG.T, out=j3c_buf).real
            t1 = log.timer_debug1("ft_ao and lr int3c2e", *t1)

            p0 = int(ao_pair_offsets[batch_id])
            p1 = int(ao_pair_offsets[batch_id + 1])
            yield p0, p1, j3c.get()
            t1 = log.timer_debug1("write int3c2e hdf5 block", *t1)

    metadata["blocks"] = blocks()
    return metadata


def build_pbc_gamma_cderi_cache_outcore(
    gdf,
    path,
    *,
    group: str = "pbc_gamma",
    overwrite: bool = True,
) -> CderiCacheInfo:
    """Build a PBC gamma CDERI HDF5 cache without materializing full ``_cderi``.

    This is a GPU4PySCF-version-sensitive entry point.  If the installed
    GPU4PySCF exposes an out-of-core hook, this function uses it.  Otherwise it
    runs an XTDDFT-local gamma builder adapted from GPU4PySCF 1.6.1.
    """

    if not getattr(gdf, "is_gamma_point", False):
        raise NotImplementedError("Out-of-core CDERI generation currently supports only gamma-point PBC GDF.")

    rsdf_builder = _require_gpu4pyscf_rsdf_builder()
    result = _call_gpu4pyscf_outcore_hook(rsdf_builder, gdf, _as_path(path), group, overwrite)
    if isinstance(result, CderiCacheInfo):
        return result
    if result is not None:
        from .df_cderi_cache import pbc_gamma_cderi_cache_info

        return pbc_gamma_cderi_cache_info(path, group=group)

    metadata = _iter_gpu4pyscf_gamma_cderi_blocks(gdf, rsdf_builder)
    gdf.auxcell = metadata["auxcell"]
    gdf.nao = gdf.cell.nao
    gdf._omega = metadata["omega"]
    gdf.kmesh = [1, 1, 1]
    info = write_pbc_gamma_cderi_cache_from_blocks(
        gdf,
        path,
        group=group,
        overwrite=overwrite,
        cderi_blocks=metadata["blocks"],
        cderi_shape=metadata["cderi_shape"],
        cderi_idx=metadata["cderi_idx"],
        omega=metadata["omega"],
    )
    return info
