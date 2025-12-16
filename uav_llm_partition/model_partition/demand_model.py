"""Demand model estimating compute, memory, and activation sizes with KV cache growth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .blocks import Block


GB = 1024.0**3


@dataclass
class BlockDemand:
    memory: float  # GB
    compute: float  # FLOPs per interval
    kv_cache: float  # GB


class DemandModel:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        head_dim: int,
        interval_tokens: int = 4,
        precision_bytes: float = 2.0,
        initial_seq_len: int = 32,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.interval_tokens = interval_tokens
        self.precision_bytes = precision_bytes
        self.seq_len = initial_seq_len
        self.initial_seq_len = initial_seq_len
        self.demands: Dict[Block, BlockDemand] = {}
        self._init_blocks()

    def _init_blocks(self) -> None:
        for l in range(self.num_layers):
            for h in range(self.num_heads):
                blk = Block(layer=l, kind="head", head_index=h)
                self.demands[blk] = BlockDemand(memory=0.0, compute=0.0, kv_cache=0.0)
            for kind in ("proj", "ffn"):
                blk = Block(layer=l, kind=kind)
                self.demands[blk] = BlockDemand(memory=0.0, compute=0.0, kv_cache=0.0)

    def blocks(self) -> List[Block]:
        return list(self.demands.keys())

    def _head_memory(self, L_s: int) -> float:
        qkv = 3 * L_s * self.head_dim * self.precision_bytes
        weights = 3 * self.hidden_size * self.head_dim * self.precision_bytes
        kv_cache = max(L_s - self.initial_seq_len, 0) * self.hidden_size * self.precision_bytes
        return (qkv + weights) / GB, kv_cache / GB

    def _head_flops(self, L_s: int) -> float:
        return 3 * L_s * self.hidden_size * self.head_dim + (L_s**2) * self.head_dim

    def _proj_memory(self, L_s: int) -> float:
        return (L_s * self.hidden_size * self.precision_bytes) / GB

    def _proj_flops(self, L_s: int) -> float:
        return L_s * (self.hidden_size**2)

    def _ffn_memory(self, L_s: int) -> float:
        return (4 * L_s * self.hidden_size * self.precision_bytes) / GB

    def _ffn_flops(self, L_s: int) -> float:
        return 0.8 * L_s * (self.hidden_size**2)

    def update_interval(self) -> Dict[Block, BlockDemand]:
        self.seq_len += self.interval_tokens
        for block, demand in self.demands.items():
            if block.kind == "head":
                mem_static, kv_cache = self._head_memory(self.seq_len)
                demand.kv_cache = kv_cache
                demand.memory = mem_static + kv_cache
                demand.compute = self._head_flops(self.seq_len)
            elif block.kind == "proj":
                demand.kv_cache = 0.0
                demand.memory = self._proj_memory(self.seq_len)
                demand.compute = self._proj_flops(self.seq_len)
            else:  # ffn
                demand.kv_cache = 0.0
                demand.memory = self._ffn_memory(self.seq_len)
                demand.compute = self._ffn_flops(self.seq_len)
        return self.demands

    def activation_size(self, upstream: Block, downstream: Block) -> float:
        if upstream.kind == "head" and downstream.kind == "proj":
            return (self.seq_len * self.head_dim * self.precision_bytes) / GB
        if upstream.kind == "proj" and downstream.kind == "ffn":
            return (self.seq_len * self.hidden_size * self.precision_bytes) / GB
        return (self.seq_len * self.hidden_size * self.precision_bytes) / GB


