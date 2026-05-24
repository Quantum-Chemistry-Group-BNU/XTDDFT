"""Array backend compatibility helpers.

This module only abstracts the small NumPy/CuPy surface needed to build and
move arrays.  PySCF/GPU4PySCF-specific functions should stay in their owning
modules until there is a concrete reason to centralize them here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
from opt_einsum import contract as _opt_einsum_contract


def _optional_import(name: str) -> tuple[ModuleType | None, Exception | None]:
    try:
        return import_module(name), None
    except Exception as err:  # pragma: no cover - depends on local env/CUDA
        return None, err


_cupy, _cupy_error = _optional_import("cupy")


def _cupy_runtime_ready() -> bool:
    if _cupy is None:
        return False
    try:
        _cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


HAS_CUPY = _cupy is not None
HAS_CUDA = _cupy_runtime_ready()
GPU_ENABLED = HAS_CUPY and HAS_CUDA


@dataclass
class Backend:
    """Switch between CPU NumPy/PySCF and GPU CuPy/GPU4PySCF execution.

    The class intentionally owns only backend selection and array movement.
    Domain-specific PySCF/GPU4PySCF functions should remain in their calling
    modules until they need shared handling.
    """

    mode: str = "auto"

    def __post_init__(self) -> None:
        self.set(self.mode)

    def set(self, mode: str = "auto") -> None:
        mode = mode.lower()
        if mode not in ("auto", "cpu", "gpu"):
            raise ValueError("backend mode must be 'auto', 'cpu', or 'gpu'")
        if mode == "gpu":
            ensure_gpu_ready()
        self.mode = mode

    def use_cpu(self) -> None:
        self.set("cpu")

    def use_gpu(self) -> None:
        self.set("gpu")

    @property
    def is_gpu(self) -> bool:
        return self.mode == "gpu" or (self.mode == "auto" and GPU_ENABLED)

    @property
    def xp(self) -> ModuleType:
        return require_cupy() if self.is_gpu else np

    @property
    def cp(self) -> ModuleType:
        return self.xp

    @property
    def pyscf_backend(self) -> str:
        return "gpu4pyscf" if self.is_gpu else "pyscf"

    def asarray(
        self,
        array: Any,
        dtype: Any | None = None,
        order: str | None = None,
        gpu: bool | None = None,
    ) -> Any:
        if gpu is True:
            module = require_cupy()
            if not HAS_CUDA:
                raise RuntimeError(f"CUDA runtime is not ready. Backend info: {backend_info()}")
        elif gpu is False:
            module = np
        else:
            module = self.xp
        return module.asarray(array, dtype=dtype, order=order)

    def asnumpy(self, array: Any, dtype: Any | None = None) -> np.ndarray:
        return asnumpy(array, dtype=dtype)

    def to_cpu(self, obj: Any) -> Any:
        return to_cpu(obj)

    def to_gpu(self, obj: Any) -> Any:
        return to_gpu(obj)

    def cast(self, obj: Any) -> Any:
        """Move arrays or PySCF-like objects to the selected backend."""

        if self.is_gpu:
            if hasattr(obj, "to_gpu"):
                return obj.to_gpu()
            if is_cupy_array(obj) or isinstance(obj, np.ndarray):
                return self.asarray(obj, gpu=True)
            return obj
        return self.to_cpu(obj)

    def info(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "array_module": getattr(self.xp, "__name__", type(self.xp).__name__),
            "pyscf_backend": self.pyscf_backend,
            "cupy": HAS_CUPY,
            "cuda": HAS_CUDA,
            "cupy_error": None if _cupy_error is None else repr(_cupy_error),
        }


class _ArrayModuleProxy:
    """Proxy NumPy/CuPy module access through the active Backend."""

    @property
    def __name__(self) -> str:
        return backend.xp.__name__

    def __getattr__(self, name: str) -> Any:
        return getattr(backend.xp, name)

    def __repr__(self) -> str:
        return repr(backend.xp)


backend = Backend()
cp = _ArrayModuleProxy()
xp = cp


def backend_info() -> dict[str, Any]:
    """Return a compact snapshot of the active backend."""

    return backend.info()


def set_backend(mode: str = "auto") -> None:
    """Set the global backend mode: ``auto``, ``cpu`` or ``gpu``."""

    backend.set(mode)


def require_cupy() -> ModuleType:
    """Return CuPy or raise an actionable import error."""

    if _cupy is None:
        raise ImportError("CuPy is required for the GPU backend") from _cupy_error
    return _cupy


def is_cupy_array(obj: Any) -> bool:
    """True when *obj* is a CuPy ndarray."""

    return _cupy is not None and isinstance(obj, _cupy.ndarray)


def get_array_module(*arrays: Any) -> ModuleType:
    """Return CuPy when any argument is a CuPy array, otherwise NumPy."""

    if _cupy is not None:
        for array in arrays:
            if is_cupy_array(array):
                return _cupy
    return np


def asnumpy(array: Any, dtype: Any | None = None) -> np.ndarray:
    """Convert an array-like object to a NumPy array."""

    if is_cupy_array(array):
        out = _cupy.asnumpy(array)
    elif hasattr(array, "get"):
        out = array.get()
    else:
        out = np.asarray(array)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def asarray(array: Any, dtype: Any | None = None, order: str | None = None, gpu: bool | None = None) -> Any:
    """Convert to the active array module, or force CPU/GPU with *gpu*."""

    return backend.asarray(array, dtype=dtype, order=order, gpu=gpu)


def contract(*args: Any, optimize: Any = True, **kwargs: Any) -> Any:
    """opt_einsum.contract with optimized contraction paths enabled by default."""

    kwargs["optimize"] = optimize
    return _opt_einsum_contract(*args, **kwargs)


def to_cpu(obj: Any) -> Any:
    """Move PySCF/GPU4PySCF objects or arrays to CPU when supported."""

    if hasattr(obj, "to_cpu"):
        return obj.to_cpu()
    if is_cupy_array(obj) or hasattr(obj, "get"):
        return asnumpy(obj)
    return obj


def to_gpu(obj: Any) -> Any:
    """Move PySCF objects or arrays to GPU when supported."""

    require_cupy()
    if not HAS_CUDA:
        raise RuntimeError(f"CUDA runtime is not ready. Backend info: {backend_info()}")
    if hasattr(obj, "to_gpu"):
        return obj.to_gpu()
    return _cupy.asarray(obj)


def ensure_gpu_ready() -> None:
    """Validate that CuPy and the CUDA runtime are available."""

    missing = []
    if not HAS_CUPY:
        missing.append("cupy")
    if HAS_CUPY and not HAS_CUDA:
        missing.append("cuda-runtime")
    if missing:
        info = backend_info()
        raise RuntimeError(f"GPU backend is not ready; missing {', '.join(missing)}. Backend info: {info}")


_asnumpy = asnumpy
_asarray = asarray
