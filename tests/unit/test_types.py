from dataclasses import FrozenInstanceError

import pytest

from mini_dist.ir.types import DType, TensorType


@pytest.mark.parametrize(
    "shape, dtype, rank, numel, itemsize, nbytes, ir",
    [
        (
            (128, 256),
            DType.FLOAT32,
            2,
            32768,
            4,
            131072,
            "tensor<128x256xf32>",
        ),
        (
            (2, 3, 4),
            DType.FLOAT16,
            3,
            24,
            2,
            48,
            "tensor<2x3x4xf16>",
        ),
        (
            (1024,),
            DType.INT8,
            1,
            1024,
            1,
            1024,
            "tensor<1024xi8>",
        ),
        (
            (),
            DType.FLOAT32,
            0,
            1,
            4,
            4,
            "tensor<f32>",
        ),
        (
            (0, 128),
            DType.FLOAT32,
            2,
            0,
            4,
            0,
            "tensor<0x128xf32>",
        ),
    ],
)
def test_tensor_type(
    shape,
    dtype,
    rank,
    numel,
    itemsize,
    nbytes,
    ir,
):
    t = TensorType(shape, dtype)

    assert t.shape == shape
    assert t.dtype == dtype
    assert t.rank == rank
    assert t.numel == numel
    assert t.itemsize == itemsize
    assert t.nbytes == nbytes
    assert str(t) == ir


@pytest.mark.parametrize(
    "shape",
    [
        None,
        [128, 256],
        (1.5, 128),
        ("128", 256),
        (True, 128),
    ],
)
def test_invalid_shape_type(shape):
    with pytest.raises(TypeError):
        TensorType(shape, DType.FLOAT32)


@pytest.mark.parametrize(
    "shape",
    [
        (-1,),
        (128, -1),
        (-2, 256),
    ],
)
def test_invalid_shape_value(shape):
    with pytest.raises(ValueError):
        TensorType(shape, DType.FLOAT32)


@pytest.mark.parametrize(
    "dtype",
    [
        None,
        "float32",
        "invalid_dtype",
        123,
    ],
)
def test_invalid_dtype(dtype):
    with pytest.raises(TypeError):
        TensorType((128, 256), dtype)


@pytest.mark.parametrize(
    "dtype, itemsize, ir_name",
    [
        (DType.FLOAT16, 2, "f16"),
        (DType.FLOAT32, 4, "f32"),
        (DType.FLOAT64, 8, "f64"),
        (DType.BFLOAT16, 2, "bf16"),
        (DType.INT8, 1, "i8"),
        (DType.INT32, 4, "i32"),
        (DType.INT64, 8, "i64"),
        (DType.BOOL, 1, "bool"),
    ],
)
def test_dtype(dtype, itemsize, ir_name):
    assert dtype.itemsize == itemsize
    assert dtype.ir_name == ir_name


@pytest.mark.parametrize("dtype", list(DType))
def test_dtype_metadata_complete(dtype):
    assert dtype.itemsize > 0
    assert dtype.ir_name


def test_tensor_type_immutable():
    t = TensorType((128, 256), DType.FLOAT32)

    with pytest.raises(FrozenInstanceError):
        t.shape = (256, 256)


def test_tensor_type_equality():
    a = TensorType((128, 256), DType.FLOAT32)
    b = TensorType((128, 256), DType.FLOAT32)
    c = TensorType((128, 256), DType.FLOAT16)

    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_tensor_type_hashable():
    a = TensorType((128, 256), DType.FLOAT32)

    mapping = {a: "tensor"}

    b = TensorType((128, 256), DType.FLOAT32)

    assert mapping[b] == "tensor"
