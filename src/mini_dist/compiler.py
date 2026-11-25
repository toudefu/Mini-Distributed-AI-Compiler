"""Public compiler entry point; implementation intentionally deferred."""
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class CompileOptions:
    backend: str = "numpy"
    world_size: int = 1
    optimize: bool = False

def compile_model(graph: Any, options: CompileOptions) -> Any:
    raise NotImplementedError("TODO: verify, partition, schedule and lower graph")
