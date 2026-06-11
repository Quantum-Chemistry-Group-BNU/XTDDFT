"""Streaming gamma-point PBC DF J/K from HDF5 CDERI caches.

This module is intentionally narrow: it targets gamma-point GPU4PySCF PBC GDF
objects whose compressed CDERI tensor has been written by
XTDDFT_dev.XTDDFT.df_cderi_cache.  The J/K contraction reads auxiliary-function
blocks from HDF5 and never materializes the full CDERI tensor in host memory.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType
import warnings

import h5py
import numpy as np


def _omega_key(omega) -> float:
    value = 0.0 if omega is None else float(omega)
    return 0.0 if abs(value) < 1e-14 else value


def _array_module(use_gpu: bool):
    if not use_gpu:
        return np
    try:
        import cupy as cp
    except ModuleNotFoundError as err:
        raise RuntimeError("use_gpu=True requires cupy") from err
    return cp


def _normalize_dm(dm, xp):
    arr = xp.asarray(dm)
    if arr.ndim == 2:
        return arr.reshape(1, arr.shape[0], arr.shape[1]), True
    if arr.ndim == 3:
        return arr, False
    raise NotImplementedError(
        "streaming PBC DF J/K supports 2D DM or stacked 3D DM only"
    )


def _restore_shape(value, single_dm):
    if value is None:
        return None
    return value[0] if single_dm else value


def _expand_cderi_block(packed, rows, cols, nao, xp):
    block = xp.zeros((packed.shape[0], nao, nao), dtype=packed.dtype)
    block[:, rows, cols] = packed
    block[:, cols, rows] = packed
    return block


def streaming_jk_from_h5(
    cache_path,
    dm,
    *,
    group="pbc_gamma",
    aux_block_size=1,
    with_j=True,
    with_k=True,
    hermi=1,
    use_gpu=True,
):
    """Contract J/K by streaming CDERI auxiliary blocks from an HDF5 cache."""

    if aux_block_size <= 0:
        raise ValueError("aux_block_size must be positive")
    if not (with_j or with_k):
        return None, None

    xp = _array_module(use_gpu)
    dm_work, single_dm = _normalize_dm(dm, xp)
    nset, nao, nao2 = dm_work.shape
    if nao != nao2:
        raise ValueError("density matrix must be square")

    vj = xp.zeros((nset, nao, nao), dtype=dm_work.dtype) if with_j else None
    vk = xp.zeros((nset, nao, nao), dtype=dm_work.dtype) if with_k else None

    with h5py.File(Path(cache_path), "r") as handle:
        h5g = handle[group]
        h5d = h5g["cderi_0"]
        file_nao = int(h5g.attrs["nao"])
        if file_nao != nao:
            raise ValueError(f"cache nao {file_nao} does not match DM nao {nao}")

        pair_idx = np.asarray(h5g["cderi_idx_pair"], dtype=np.int64)
        rows_np, cols_np = np.divmod(pair_idx, nao)
        rows = xp.asarray(rows_np)
        cols = xp.asarray(cols_np)

        naux = int(h5d.shape[0])
        for p0 in range(0, naux, aux_block_size):
            p1 = min(p0 + aux_block_size, naux)
            packed = xp.asarray(np.asarray(h5d[p0:p1]))
            cderi = _expand_cderi_block(packed, rows, cols, nao, xp)

            if with_j:
                rho = xp.einsum("Lpq,xpq->xL", cderi, dm_work, optimize=True)
                vj += xp.einsum("xL,Lpq->xpq", rho, cderi, optimize=True)

            if with_k:
                for lidx in range(cderi.shape[0]):
                    lmat = cderi[lidx]
                    for xidx in range(nset):
                        vk[xidx] += lmat @ dm_work[xidx] @ lmat.T

            del cderi, packed

    return _restore_shape(vj, single_dm), _restore_shape(vk, single_dm)


def _load_xtddft_cache_helpers():
    from XTDDFT_dev.XTDDFT.df_cderi_cache import (
        DFCderiCacheConfig,
        pbc_gamma_cderi_cache_path,
        prepare_pbc_gamma_df_cderi_cache,
    )

    return DFCderiCacheConfig, pbc_gamma_cderi_cache_path, prepare_pbc_gamma_df_cderi_cache


def _prepare_cache_for_omega(with_df, config, prepare_pbc_gamma_df_cderi_cache, omega):
    omega = _omega_key(omega)
    if omega == 0.0:
        with_df.is_gamma_point = True
        return prepare_pbc_gamma_df_cderi_cache(with_df, config, omega=0.0)

    with with_df.range_coulomb(omega) as rsh_df:
        rsh_df.is_gamma_point = True
        return prepare_pbc_gamma_df_cderi_cache(rsh_df, config, omega=omega)


def install_streaming_df_jk(
    mf,
    *,
    cache_dir,
    tag="uks_hse06_ccpvdz",
    omegas=(0.0, -0.11),
    aux_block_size=1,
    build_cache=True,
    use_gpu=True,
):
    """Build CDERI caches and patch ``mf.with_df.get_jk`` to stream them."""

    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        raise ValueError("mf.with_df is required")
    if not getattr(with_df, "is_gamma_point", False):
        raise NotImplementedError("streaming DF J/K currently supports gamma point only")

    DFCderiCacheConfig, cache_path_for, prepare_cache = _load_xtddft_cache_helpers()
    config = DFCderiCacheConfig(
        mode="auto",
        cache_dir=Path(cache_dir),
        tag=tag,
        build_backend="outcore",
    )

    cache_paths = {}
    for omega in omegas:
        omega = _omega_key(omega)
        cache_path = cache_path_for(config, omega=omega)
        cache_paths[omega] = cache_path
        if build_cache:
            print(f"Preparing streaming DF CDERI cache omega={omega}: {cache_path}")
            _prepare_cache_for_omega(with_df, config, prepare_cache, omega)

    original_get_jk = getattr(with_df, "get_jk")
    setattr(with_df, "_xtddft_streaming_original_get_jk", original_get_jk)
    setattr(with_df, "_xtddft_streaming_cache_paths", cache_paths)
    setattr(with_df, "_xtddft_streaming_aux_block_size", int(aux_block_size))

    def streaming_get_jk(
        self,
        dm,
        hermi=1,
        kpts=None,
        kpts_band=None,
        with_j=True,
        with_k=True,
        omega=None,
        exxdiv=None,
    ):
        if kpts is not None or kpts_band is not None:
            return original_get_jk(
                dm, hermi, kpts, kpts_band, with_j, with_k, omega=omega, exxdiv=exxdiv
            )

        actual_omega = _omega_key(omega)
        cache_path = cache_paths.get(actual_omega)
        if cache_path is None:
            return original_get_jk(
                dm, hermi, kpts, kpts_band, with_j, with_k, omega=omega, exxdiv=exxdiv
            )

        if exxdiv is not None and with_k and actual_omega == 0.0:
            warnings.warn(
                "streaming DF J/K does not apply the full-range PBC exxdiv correction; "
                "use this path for short-range HSE exchange or validate energies carefully.",
                RuntimeWarning,
                stacklevel=2,
            )

        return streaming_jk_from_h5(
            cache_path,
            dm,
            aux_block_size=aux_block_size,
            with_j=with_j,
            with_k=with_k,
            hermi=hermi,
            use_gpu=use_gpu,
        )

    with_df.get_jk = MethodType(streaming_get_jk, with_df)
    print("Installed streaming HDF5 DF J/K")
    for omega, path in cache_paths.items():
        print(f"  omega={omega}: {path}")
    print(f"  aux_block_size={aux_block_size}")
    return cache_paths


def _self_test():
    import tempfile

    nao = 4
    pair_idx = np.array([0, 1, 4, 5, 8, 10], dtype=np.int64)
    cderi = np.arange(30, dtype=float).reshape(5, 6) / 10.0
    dm = np.arange(32, dtype=float).reshape(2, nao, nao) / 17.0
    dm = dm + dm.transpose(0, 2, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cache.h5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group("pbc_gamma")
            group.create_dataset("cderi_0", data=cderi)
            group.create_dataset("cderi_idx_pair", data=pair_idx)
            group.create_dataset("cderi_idx_diag", data=np.array([], dtype=np.int64))
            group.attrs["nao"] = nao
            group.attrs["omega"] = 0.0
        streaming_jk_from_h5(path, dm, aux_block_size=2, use_gpu=False)
    print("streaming_pbc_df_jk self-test ok")


if __name__ == "__main__":
    _self_test()
