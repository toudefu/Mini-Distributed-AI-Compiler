"""Core tensor types for Mini Distributed AI Compiler."""

import math
from dataclasses import dataclass
from enum import Enum


class DType(str, Enum):
    """Supported scalar data types."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"

    INT8 = "int8"
    INT32 = "int32"
    INT64 = "int64"

    BOOL = "bool"

    @property
    def itemsize(self) -> int:
        """Number of bytes per element."""
        return _DTYPE_ITEMSIZE[self]

    @property
    def ir_name(self) -> str:
        """Textual representation in IR."""
        return _DTYPE_IR_NAME[self]


_DTYPE_ITEMSIZE = {
    DType.FLOAT16: 2,
    DType.FLOAT32: 4,
    DType.FLOAT64: 8,
    DType.BFLOAT16: 2,
    DType.INT8: 1,
    DType.INT32: 4,
    DType.INT64: 8,
    DType.BOOL: 1,
}


_DTYPE_IR_NAME = {
    DType.FLOAT16: "f16",
    DType.FLOAT32: "f32",
    DType.FLOAT64: "f64",
    DType.BFLOAT16: "bf16",
    DType.INT8: "i8",
    DType.INT32: "i32",
    DType.INT64: "i64",
    DType.BOOL: "bool",
}


@dataclass(frozen=True)
class TensorType:
    """A statically shaped tensor type."""

    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            raise TypeError("shape must be a tuple")

        for dim in self.shape:
            if type(dim) is not int:
                raise TypeError("shape dimensions must be integers")

            if dim < 0:
                raise ValueError("shape dimensions must be non-negative")

        if not isinstance(self.dtype, DType):
            raise TypeError("dtype must be an instance of DType")

    @property
    def rank(self) -> int:
        """Number of tensor dimensions."""
        return len(self.shape)

    @property
    def numel(self) -> int:
        """Total number of tensor elements."""
        return math.prod(self.shape)

    @property
    def itemsize(self) -> int:
        """Number of bytes per element."""
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        """Logical tensor size in bytes."""
        return self.numel * self.itemsize

    def __str__(self) -> str:
        parts = [str(dim) for dim in self.shape]
        parts.append(self.dtype.ir_name)

        return f"tensor<{'x'.join(parts)}>"
