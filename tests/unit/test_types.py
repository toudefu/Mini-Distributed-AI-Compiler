from mini_dist.ir.types import DType, TensorType
import pytest

@pytest
def test_tensor_type():
    t = TensorType(
        shape=(128, 256),
        dtype=DType.FLOAT32,
    )

    assert t.shape == (128, 256)
    assert t.rank == 2
    assert t.numel == 32768
    assert t.itemsize == 4
    assert t.nbytes == 131072
    assert str(t) == "tensor<128x256xf32>"

