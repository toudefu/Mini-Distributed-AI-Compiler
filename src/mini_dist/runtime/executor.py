"""Runtime contract (provisional)."""
from typing import Protocol, Any

class Executor(Protocol):
    def run(self, plan: Any, inputs: dict[str, Any]) -> Any: ...
