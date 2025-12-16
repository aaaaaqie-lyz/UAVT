"""Demand model estimating compute and memory per block including KV cache growth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .blocks import Block


@dataclass
class BlockDemand:
    memory: float
    compute: float
    kv_cache: float


class DemandModel:
    def __init__(self, num_layers: int, num_heads: int, hidden_size: int, head_dim: int, interval_tokens: int = 4) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.interval_tokens = interval_tokens
        self.kv_cache_per_token = head_dim * 2.0  # key + value
        self.static_head_mem = head_dim * 0.1
        self.proj_mem = hidden_size * 0.05
        self.ffn_mem = hidden_size * 0.08
        self.compute_scaler = hidden_size * 2.5
        self.token_count = 0
        self.demands: Dict[Block, BlockDemand] = {}
        self._init_blocks()

    def _init_blocks(self) -> None:
        for l in range(self.num_layers):
            for h in range(self.num_heads):
                blk = Block(layer=l, kind="head", head_index=h)
                self.demands[blk] = BlockDemand(memory=self.static_head_mem, compute=self.compute_scaler, kv_cache=0.0)
            for kind, mem_factor in [("proj", self.proj_mem), ("ffn", self.ffn_mem)]:
                blk = Block(layer=l, kind=kind)
                self.demands[blk] = BlockDemand(memory=mem_factor, compute=self.compute_scaler * 1.2, kv_cache=0.0)

    def blocks(self) -> List[Block]:
        return list(self.demands.keys())

    def update_interval(self) -> Dict[Block, BlockDemand]:
        self.token_count += self.interval_tokens
        for block, demand in self.demands.items():
            if block.kind == "head":
                kv_growth = self.interval_tokens * self.kv_cache_per_token
                demand.kv_cache += kv_growth
                demand.memory = self.static_head_mem + demand.kv_cache
            demand.compute = self.compute_scaler * (1.5 if block.kind == "ffn" else 1.0)
        return self.demands

    def activation_size(self, upstream: Block, downstream: Block) -> float:
        # Simplified activation estimate proportional to hidden size
        return self.hidden_size * (0.5 if upstream.kind == "head" else 1.0)

