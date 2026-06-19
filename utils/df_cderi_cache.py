"""Out-of-core density-fitting CDERI cache helpers.

The molecular helpers in this module are a small correctness prototype for the
larger PBC/GPU4PySCF out-of-core path.  They keep the three-center DF tensor on
disk and read it over the auxiliary basis dimension when contracting to MO ERIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from pyscf import df, lib

from .backend import _asnumpy, contract


@dataclass(frozen=True)
class CderiCacheInfo:
    path: Path
    dataset: str
    naux: int
    nao_pair: int
    aosym: str


@dataclass(frozen=True)
class DFCderiCacheConfig:
    """Configuration for optional CDERI HDF5 reuse."""

    mode: str = "off"
    cache_dir: Path | str | None = None
    tag: str = "df_cderi"
    build_backend: str = "gpu4pyscf"

    def __post_init__(self):
        mode = self.mode.lower()
        if mode not in ("off", "load", "build", "auto"):
            raise ValueError("df cderi cache mode must be 'off', 'load', 'build', or 'auto'")
        object.__setattr__(self, "mode", mode)
        build_backend = self.build_backend.lower()
        if build_backend not in ("gpu4pyscf", "outcore"):
            raise ValueError("df cderi cache build_backend must be 'gpu4pyscf' or 'outcore'")
        object.__setattr__(self, "build_backend", build_backend)
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", _as_path(self.cache_dir))


class PbcGammaCderiCacheHandle:
    """Keep an attached PBC gamma CDERI HDF5 cache alive for a GDF object."""

    def __init__(self, gdf, h5file):
        self.gdf = gdf
        self.h5file = h5file

    def close(self) -> None:
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _as_path(path) -> Path:
    return Path(path).expanduser()


def normalize_df_cderi_cache_config(config) -> DFCderiCacheConfig:
    if config is None:
        return DFCderiCacheConfig()
    if isinstance(config, DFCderiCacheConfig):
        return config
    if isinstance(config, (str, Path)):
        return DFCderiCacheConfig(mode="auto", cache_dir=config)
    if isinstance(config, dict):
        return DFCderiCacheConfig(**config)
    raise TypeError("df_cache must be None, a path, a dict, or DFCderiCacheConfig")


def _omega_label(omega) -> str:
    value = 0.0 if omega is None else float(omega)
    text = f"{abs(value):.12g}".replace(".", "p").replace("+", "")
    return f"minus_{text}" if value < 0 else text


def pbc_gamma_cderi_cache_path(config: DFCderiCacheConfig, omega=None) -> Path:
    if config.cache_dir is None:
        raise ValueError("df cderi cache requires cache_dir when mode is not 'off'")
    return _as_path(config.cache_dir) / f"{config.tag}_omega_{_omega_label(omega)}.h5"


def build_molecular_cderi_cache(
    mol,
    path,
    auxbasis="weigend+etb",
    dataset="j3c",
    max_memory=2000,
    overwrite=True,
) -> CderiCacheInfo:
    """Build a molecular DF CDERI HDF5 cache with PySCF outcore integrals."""

    cache_path = _as_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        if not overwrite:
            return molecular_cderi_cache_info(cache_path, dataset=dataset)
        cache_path.unlink()

    df.outcore.cholesky_eri(
        mol,
        str(cache_path),
        auxbasis=auxbasis,
        dataname=dataset,
        aosym="s2ij",
        max_memory=max_memory,
        verbose=getattr(mol, "verbose", 0),
    )
    with h5py.File(cache_path, "a") as handle:
        handle[dataset].attrs["aosym"] = "s2ij"
        handle[dataset].attrs["auxbasis"] = str(auxbasis)
    return molecular_cderi_cache_info(cache_path, dataset=dataset)


def molecular_cderi_cache_info(path, dataset="j3c") -> CderiCacheInfo:
    cache_path = _as_path(path)
    with h5py.File(cache_path, "r") as handle:
        h5d = handle[dataset]
        naux, nao_pair = h5d.shape
        aosym = h5d.attrs.get("aosym", "s2ij")
        if isinstance(aosym, bytes):
            aosym = aosym.decode()
    return CderiCacheInfo(
        path=cache_path,
        dataset=dataset,
        naux=int(naux),
        nao_pair=int(nao_pair),
        aosym=str(aosym),
    )


def iter_molecular_cderi_blocks(path, dataset="j3c", block_size=256) -> Iterator[np.ndarray]:
    """Yield packed CDERI blocks from HDF5 over the auxiliary basis dimension."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    cache_path = _as_path(path)
    with h5py.File(cache_path, "r") as handle:
        h5d = handle[dataset]
        naux = h5d.shape[0]
        for p0 in range(0, naux, block_size):
            p1 = min(p0 + block_size, naux)
            yield np.asarray(h5d[p0:p1])


def _normalize_mo_coeffs(mo_coeffs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(mo_coeffs, np.ndarray) and mo_coeffs.ndim == 2:
        coeffs = (mo_coeffs,) * 4
    else:
        coeffs = tuple(mo_coeffs)
    if len(coeffs) != 4:
        raise ValueError("mo_coeffs must be a 2D array or a sequence of four 2D arrays")
    return tuple(np.asarray(coeff) for coeff in coeffs)


def ao2mo_from_molecular_cderi_cache(
    path,
    mo_coeffs,
    dataset="j3c",
    block_size=256,
    compact=False,
) -> np.ndarray:
    """Contract cached molecular CDERI blocks to MO ERIs.

    The returned layout matches PySCF ``df.DF(...).ao2mo(..., compact=False)``:
    ``(nmo_i * nmo_j, nmo_k * nmo_l)``.
    """

    if compact:
        raise NotImplementedError("compact=True is not implemented for cached molecular CDERI ao2mo.")
    ci, cj, ck, cl = _normalize_mo_coeffs(mo_coeffs)
    ni, nj = ci.shape[1], cj.shape[1]
    nk, nl = ck.shape[1], cl.shape[1]
    eri = np.zeros((ni * nj, nk * nl), dtype=np.result_type(ci, cj, ck, cl, np.float64))

    for packed in iter_molecular_cderi_blocks(path, dataset=dataset, block_size=block_size):
        cderi = lib.unpack_tril(packed)
        lij = contract("Lpq,pi,qj->Lij", cderi, ci.conj(), cj).reshape(packed.shape[0], -1)
        lkl = contract("Lpq,pk,ql->Lkl", cderi, ck.conj(), cl).reshape(packed.shape[0], -1)
        eri += contract("Lp,Lq->pq", lij, lkl)
    return eri


def _require_pbc_gamma_gdf(gdf) -> None:
    if not getattr(gdf, "is_gamma_point", False):
        raise NotImplementedError("Only gamma-point PBC GDF objects are supported.")
    cderi = getattr(gdf, "_cderi", None)
    if cderi is None or 0 not in cderi:
        raise ValueError("gdf._cderi[0] is not available. Build the gamma-point GDF first.")
    if getattr(gdf, "_cderi_idx", None) is None:
        raise ValueError("gdf._cderi_idx is not available. Build the gamma-point GDF first.")


def build_pbc_gamma_cderi_cache(
    gdf,
    path,
    group="pbc_gamma",
    overwrite=True,
) -> CderiCacheInfo:
    """Write a built GPU4PySCF/PBC gamma CDERI object to HDF5."""

    _require_pbc_gamma_gdf(gdf)
    cache_path = _as_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        if not overwrite:
            return pbc_gamma_cderi_cache_info(cache_path, group=group)
        cache_path.unlink()

    cderi0 = _asnumpy(gdf._cderi[0])
    pair_idx, diag_idx = gdf._cderi_idx
    with h5py.File(cache_path, "w") as handle:
        h5g = handle.create_group(group)
        h5g.create_dataset("cderi_0", data=cderi0)
        cderip = getattr(gdf, "_cderip", None)
        if cderip is not None and 0 in cderip:
            h5g.create_dataset("cderip_0", data=_asnumpy(cderip[0]))
        h5g.create_dataset("cderi_idx_pair", data=_asnumpy(pair_idx).astype(np.int64, copy=False))
        h5g.create_dataset("cderi_idx_diag", data=_asnumpy(diag_idx).astype(np.int64, copy=False))
        h5g.attrs["format"] = "gpu4pyscf_pbc_gamma_cderi"
        h5g.attrs["is_gamma_point"] = True
        h5g.attrs["nao"] = int(getattr(gdf, "nao", getattr(getattr(gdf, "cell", None), "nao", 0)))
        h5g.attrs["omega"] = float(getattr(gdf, "_omega", 0.0))
        h5g.attrs["auxbasis"] = repr(getattr(gdf, "auxbasis", None))
    return pbc_gamma_cderi_cache_info(cache_path, group=group)


def pbc_gamma_cderi_cache_info(path, group="pbc_gamma") -> CderiCacheInfo:
    cache_path = _as_path(path)
    with h5py.File(cache_path, "r") as handle:
        h5g = handle[group]
        h5d = h5g["cderi_0"]
        naux, nao_pair = h5d.shape
    return CderiCacheInfo(
        path=cache_path,
        dataset=f"{group}/cderi_0",
        naux=int(naux),
        nao_pair=int(nao_pair),
        aosym="gpu4pyscf_pbc_gamma_compressed",
    )


def attach_pbc_gamma_cderi_cache(gdf, path, group="pbc_gamma") -> PbcGammaCderiCacheHandle:
    """Attach an HDF5-backed gamma CDERI cache to a GPU4PySCF PBC GDF object."""

    cache_path = _as_path(path)
    handle = h5py.File(cache_path, "r")
    try:
        h5g = handle[group]
        gdf.is_gamma_point = bool(h5g.attrs.get("is_gamma_point", True))
        gdf.nao = int(h5g.attrs["nao"])
        gdf._omega = float(h5g.attrs.get("omega", 0.0))
        gdf._cderi = {0: h5g["cderi_0"]}
        if "cderip_0" in h5g:
            gdf._cderip = {0: h5g["cderip_0"]}
        else:
            gdf._cderip = {}
        pair_idx = np.asarray(h5g["cderi_idx_pair"])
        diag_idx = np.asarray(h5g["cderi_idx_diag"])
        try:
            import cupy as cp

            cp.cuda.runtime.getDeviceCount()
            pair_idx = cp.asarray(pair_idx)
            diag_idx = cp.asarray(diag_idx)
        except Exception:
            pass
        gdf._cderi_idx = (pair_idx, diag_idx)
        return PbcGammaCderiCacheHandle(gdf, handle)
    except Exception:
        handle.close()
        raise


def _get_gdf_cache_registry(gdf) -> dict:
    registry = getattr(gdf, "_xtddft_cderi_cache_handles", None)
    if registry is None:
        registry = {}
        setattr(gdf, "_xtddft_cderi_cache_handles", registry)
    return registry


def prepare_pbc_gamma_df_cderi_cache(gdf, config=None, omega=None, group="pbc_gamma"):
    """Build/load/attach a gamma-point PBC CDERI cache for one GDF object."""

    cfg = normalize_df_cderi_cache_config(config)
    if cfg.mode == "off":
        return None
    if not getattr(gdf, "is_gamma_point", False):
        return None

    cache_path = pbc_gamma_cderi_cache_path(cfg, omega=omega)
    key = (str(cache_path), group)
    registry = _get_gdf_cache_registry(gdf)
    existing = registry.get(key)
    if existing is not None and existing.h5file is not None:
        return existing

    exists = cache_path.exists()
    if cfg.mode == "load" and not exists:
        raise FileNotFoundError(f"CDERI cache does not exist: {cache_path}")

    if cfg.mode == "build" or (cfg.mode == "auto" and not exists):
        if cfg.build_backend == "outcore":
            from .pbc_cderi_outcore import build_pbc_gamma_cderi_cache_outcore

            build_pbc_gamma_cderi_cache_outcore(gdf, cache_path, group=group, overwrite=True)
        else:
            if getattr(gdf, "_cderi", None) is None:
                gdf.build()
            build_pbc_gamma_cderi_cache(gdf, cache_path, group=group, overwrite=True)
    elif cfg.mode == "auto" and exists:
        pass
    elif cfg.mode == "load":
        pass

    handle = attach_pbc_gamma_cderi_cache(gdf, cache_path, group=group)
    registry[key] = handle
    return handle
