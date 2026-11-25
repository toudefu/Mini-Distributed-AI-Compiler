import importlib
import pkgutil
import mini_dist
from mini_dist.compiler import CompileOptions, compile_model
import pytest

def test_imports():
    for mod in pkgutil.walk_packages(mini_dist.__path__, mini_dist.__name__ + "."):
        importlib.import_module(mod.name)

def test_unimplemented():
    with pytest.raises(NotImplementedError):
        compile_model(None, CompileOptions())
