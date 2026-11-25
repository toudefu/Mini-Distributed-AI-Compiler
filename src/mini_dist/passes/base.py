"""Compiler pass contract (provisional)."""
from typing import Protocol, Any

class Pass(Protocol):
    name: str
    def run(self, module: Any) -> Any: ...
