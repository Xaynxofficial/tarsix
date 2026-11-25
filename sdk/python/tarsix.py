"""Zero-dependency Python ctypes bindings for the native Rust Tarsix AI engine."""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path
from typing import Sequence


def _default_lib_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    sys_name = platform.system()
    if sys_name == "Darwin":
        name = "libas_wasm.dylib"
    elif sys_name == "Windows":
        name = "as_wasm.dll"
    else:
        name = "libas_wasm.so"
    return root / "target" / "release" / name


def _load_library(lib_path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
    path = Path(lib_path) if lib_path is not None else _default_lib_path()
    lib = ctypes.CDLL(str(path))

    u32_p = ctypes.POINTER(ctypes.c_uint32)
    f32_p = ctypes.POINTER(ctypes.c_float)

    lib.as_tensor_zeros.argtypes = [u32_p, ctypes.c_size_t]
    lib.as_tensor_zeros.restype = ctypes.c_uint32

    lib.as_tensor_ones.argtypes = [u32_p, ctypes.c_size_t]
    lib.as_tensor_ones.restype = ctypes.c_uint32

    lib.as_tensor_from_f32.argtypes = [f32_p, ctypes.c_size_t, u32_p, ctypes.c_size_t]
    lib.as_tensor_from_f32.restype = ctypes.c_uint32

    for fn in ("as_tensor_add", "as_tensor_sub", "as_tensor_mul", "as_tensor_div", "as_tensor_matmul"):
        getattr(lib, fn).argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        getattr(lib, fn).restype = ctypes.c_uint32

    lib.as_tensor_transpose.argtypes = [ctypes.c_uint32, ctypes.c_size_t, ctypes.c_size_t]
    lib.as_tensor_transpose.restype = ctypes.c_uint32

    lib.as_tensor_reshape.argtypes = [ctypes.c_uint32, u32_p, ctypes.c_size_t]
    lib.as_tensor_reshape.restype = ctypes.c_uint32

    for fn in ("as_tensor_sum", "as_tensor_mean"):
        getattr(lib, fn).argtypes = [ctypes.c_uint32]
        getattr(lib, fn).restype = ctypes.c_uint32

    lib.as_tensor_numel.argtypes = [ctypes.c_uint32]
    lib.as_tensor_numel.restype = ctypes.c_size_t

    lib.as_tensor_read_f32.argtypes = [ctypes.c_uint32, f32_p, ctypes.c_size_t]
    lib.as_tensor_read_f32.restype = ctypes.c_int32

    lib.as_tensor_dispose.argtypes = [ctypes.c_uint32]
    lib.as_tensor_dispose.restype = None

    return lib


class Tensor:
    """Native multi-core Rust Tensor accessible directly from Python via C ABI."""

    _lib: ctypes.CDLL | None = None

    def __init__(self, handle: int, lib: ctypes.CDLL) -> None:
        if handle == 0:
            raise RuntimeError("Tarsix native operation failed (null handle)")
        self._handle = handle
        self._lib = lib

    @classmethod
    def _get_lib(cls, lib_path: str | os.PathLike[str] | None = None) -> ctypes.CDLL:
        if cls._lib is None or lib_path is not None:
            cls._lib = _load_library(lib_path)
        return cls._lib

    @classmethod
    def from_list(
        cls,
        data: Sequence[float],
        shape: Sequence[int],
        lib_path: str | os.PathLike[str] | None = None,
    ) -> Tensor:
        lib = cls._get_lib(lib_path)
        data_arr = (ctypes.c_float * len(data))(*data)
        shape_arr = (ctypes.c_uint32 * len(shape))(*shape)
        h = lib.as_tensor_from_f32(data_arr, len(data), shape_arr, len(shape))
        return cls(h, lib)

    @classmethod
    def zeros(cls, shape: Sequence[int]) -> Tensor:
        lib = cls._get_lib()
        shape_arr = (ctypes.c_uint32 * len(shape))(*shape)
        return cls(lib.as_tensor_zeros(shape_arr, len(shape)), lib)

    @classmethod
    def ones(cls, shape: Sequence[int]) -> Tensor:
        lib = cls._get_lib()
        shape_arr = (ctypes.c_uint32 * len(shape))(*shape)
        return cls(lib.as_tensor_ones(shape_arr, len(shape)), lib)

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor(self._lib.as_tensor_add(self._handle, other._handle), self._lib)

    def __sub__(self, other: Tensor) -> Tensor:
        return Tensor(self._lib.as_tensor_sub(self._handle, other._handle), self._lib)

    def __mul__(self, other: Tensor) -> Tensor:
        return Tensor(self._lib.as_tensor_mul(self._handle, other._handle), self._lib)

    def __truediv__(self, other: Tensor) -> Tensor:
        return Tensor(self._lib.as_tensor_div(self._handle, other._handle), self._lib)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Tensor(self._lib.as_tensor_matmul(self._handle, other._handle), self._lib)

    def transpose(self, axis_a: int = 0, axis_b: int = 1) -> Tensor:
        return Tensor(self._lib.as_tensor_transpose(self._handle, axis_a, axis_b), self._lib)

    def reshape(self, new_shape: Sequence[int]) -> Tensor:
        shape_arr = (ctypes.c_uint32 * len(new_shape))(*new_shape)
        return Tensor(self._lib.as_tensor_reshape(self._handle, shape_arr, len(new_shape)), self._lib)

    def sum(self) -> Tensor:
        return Tensor(self._lib.as_tensor_sum(self._handle), self._lib)

    def mean(self) -> Tensor:
        return Tensor(self._lib.as_tensor_mean(self._handle), self._lib)

    def to_list(self) -> list[float]:
        n = self._lib.as_tensor_numel(self._handle)
        if n == 0:
            return []
        buf = (ctypes.c_float * n)()
        rc = self._lib.as_tensor_read_f32(self._handle, buf, n)
        if rc != 0:
            raise RuntimeError(f"as_tensor_read_f32 failed with status {rc}")
        return [float(buf[i]) for i in range(n)]

    def dispose(self) -> None:
        if self._handle != 0 and self._lib is not None:
            self._lib.as_tensor_dispose(self._handle)
            self._handle = 0

    def __del__(self) -> None:
        self.dispose()


if __name__ == "__main__":
    a = Tensor.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = Tensor.from_list([5.0, 6.0, 7.0, 8.0], [2, 2])
    bias = Tensor.from_list([1.0, -2.0], [1, 2])

    c = (a @ b) + bias
    assert c.to_list() == [20.0, 20.0, 44.0, 48.0], f"unexpected {c.to_list()}"
    assert c.mean().to_list() == [33.0]
    print("Python -> Native Rust C-ABI test PASSED:", c.to_list())
