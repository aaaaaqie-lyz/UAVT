"""Definitions of blocks representing partitioned Transformer components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(eq=True, frozen=True)
class Block:
    layer: int
    kind: str  # "head", "proj", "ffn"
    head_index: Optional[int] = None

    def identifier(self) -> str:
        if self.kind == "head" and self.head_index is not None:
            return f"layer{self.layer}_head{self.head_index}"
        return f"layer{self.layer}_{self.kind}"

    def __str__(self) -> str:  # pragma: no cover - readability helper
        return self.identifier()

