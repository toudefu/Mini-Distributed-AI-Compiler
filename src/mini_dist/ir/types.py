"""TensorType、dtype、shape。

TODO: define interfaces, implement and add tests.
"""
from enum import Enum
from dataclasses import dataclass
import math

class DType(str, Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"

    INT8 = "int8"
    INT32 = "int32"
    INT64 = "int64"

    BOOL = "bool"


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: DType

    def __post_init(self):
        if not (self.shape is not None
            and all(i > 0 for i in self.shape)
            and self.dtype is not None
        ):
            raise NotImplementedError

    @property
    def rank(self) -> int:
        if self.shape is not None:
            return len(self.shape)
        raise NotImplementedError

    @property
    def numel(self) -> int:
        if self.shape is not None:
            return math.prod(self.shape)
        raise NotImplementedError
    
    @property
    def itemsize(self) -> int:
        if self.dtype is not None:
            return self.__sizeof__(self.dtype)
        raise NotImplementedError

    @property
    def nbytes(self) -> int:
        return self.numel() * self.itemsize()

    @property
    def __str__(self) -> str:
        if (
            self.shape is not None
            and self.dtype is not None
            and len(self.shape) > 0
        ):
            tensor_str = "tensor<"
            for dim in self.shape:
                tensor_str += str(dim)
            tensor_str += "x"
            if self.dtype == "float32":
                tensor_str += "f32"
            elif self.dtype == "float16":
                tensor_str += "f16"
            elif self.dtype == "bfloat16":
                tensor_str += "bf16"
            elif self.dtype == "int8":
                tensor_str += "i8"
            elif self.dtype == "int32":
                tensor_str += "i32"
            elif self.dtype == "int64":
                tensor_str += "i64"
            elif self.dtype == "bool":
                tensor_str += "bool"
            tensor_str += ">"
            return tensor_str
        raise NotImplementedError
