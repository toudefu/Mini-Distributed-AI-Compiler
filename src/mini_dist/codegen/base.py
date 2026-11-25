"""Code generation contract (provisional)."""
from typing import Protocol, Any

class Backend(Protocol):
    def lower(self, plan: Any) -> Any: ...
