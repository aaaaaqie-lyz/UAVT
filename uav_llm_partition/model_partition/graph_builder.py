"""Block dependency graph builder."""
from __future__ import annotations

from typing import Dict, List, Tuple
from .blocks import Block


def build_dependencies(blocks: List[Block], num_heads: int) -> List[Tuple[Block, Block]]:
    edges: List[Tuple[Block, Block]] = []
    # For each layer, connect heads -> proj -> ffn -> next layer heads
    by_layer = {}
    for blk in blocks:
        by_layer.setdefault(blk.layer, []).append(blk)
    max_layer = max(by_layer.keys())
    for layer, layer_blocks in by_layer.items():
        head_blocks = [b for b in layer_blocks if b.kind == "head"]
        proj = next(b for b in layer_blocks if b.kind == "proj")
        ffn = next(b for b in layer_blocks if b.kind == "ffn")
        for h in head_blocks:
            edges.append((h, proj))
        edges.append((proj, ffn))
        if layer < max_layer:
            next_heads = [b for b in by_layer[layer + 1] if b.kind == "head"]
            for nh in next_heads:
                edges.append((ffn, nh))
    return edges

