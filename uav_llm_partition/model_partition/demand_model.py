"""Demand model built on top of PartitionedTransformer.

This re-implements the original UAVT ``DemandModel`` interface, but the
underlying formulas and structure are taken directly from the DTIS-style
``PartitionedTransformer`` in ``transformer_partition.py``.

It exposes:
  - ``blocks()``: logical blocks (layer/head/proj/ffn) as in the original UAVT code.
  - ``update_interval()``: return a ``Dict[Block, BlockDemand]`` for the next interval.
  - ``activation_size()``: activation transfer size between two blocks (GB).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .blocks import Block
from .transformer_partition import (
    AttentionHeadComponent,
    FFNComponent,
    PartitionedTransformer,
    TransformerBlockComponent,
    TransformerPartitionConfig,
)


GB = 1024.0**3


@dataclass
class BlockDemand:
    """Per-block demand snapshot used by the scheduler.

    - ``memory``: total memory footprint in GB (activations + weights + KV cache)
    - ``compute``: FLOPs required in the current interval
    - ``kv_cache``: KV cache portion only (GB)
    """

    memory: float  # GB
    compute: float  # FLOPs per interval
    kv_cache: float  # GB


class DemandModel:
    """High-level demand model wrapping a DTIS-style PartitionedTransformer.

    The public API mirrors the original UAVT DemandModel, but all per-block
    numbers are computed by calling ``compute_memory_requirements``,
    ``compute_cache_memory`` and ``compute_flops`` on the underlying
    PartitionedTransformer components.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        head_dim: Optional[int] = None,
        interval_tokens: int = 4,
        precision_bytes: float = 2.0,
        initial_seq_len: int = 32,
    ) -> None:
        # Build DTIS-style config and transformer
        cfg = TransformerPartitionConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            initial_sequence_length=initial_seq_len,
            interval_tokens=interval_tokens,
            precision_bytes=int(precision_bytes),
            head_dim_override=head_dim,
        )
        self.transformer = PartitionedTransformer(cfg)

        # Convenience aliases
        self.interval_tokens = interval_tokens
        self.initial_seq_len = initial_seq_len

        # Mapping: Block -> component_id and back
        self._block_to_comp: Dict[Block, str] = {}
        self._comp_to_block: Dict[str, Block] = {}
        self._init_block_mapping()

        # Storage for last computed demands
        self.demands: Dict[Block, BlockDemand] = {
            blk: BlockDemand(memory=0.0, compute=0.0, kv_cache=0.0)
            for blk in self._block_to_comp
        }

    # ------------------------------------------------------------------
    # Structure and mapping helpers
    # ------------------------------------------------------------------
    def _init_block_mapping(self) -> None:
        """Create Block <-> component_id mapping matching the original UAVT Blocks."""

        for layer in range(self.transformer.cfg.num_layers):
            # Heads
            for h in range(self.transformer.cfg.num_heads):
                blk = Block(layer=layer, kind="head", head_index=h)
                comp_id = f"layer{layer}_head{h}"
                self._block_to_comp[blk] = comp_id
                self._comp_to_block[comp_id] = blk

            # Projection
            proj_blk = Block(layer=layer, kind="proj", head_index=None)
            proj_id = f"layer{layer}_proj"
            self._block_to_comp[proj_blk] = proj_id
            self._comp_to_block[proj_id] = proj_blk

            # FFN
            ffn_blk = Block(layer=layer, kind="ffn", head_index=None)
            ffn_id = f"layer{layer}_ffn"
            self._block_to_comp[ffn_blk] = ffn_id
            self._comp_to_block[ffn_id] = ffn_blk

    def blocks(self) -> List[Block]:
        """Return all logical blocks (layer/head/proj/ffn)."""

        return list(self._block_to_comp.keys())

    # ------------------------------------------------------------------
    # Per-component demand computation
    # ------------------------------------------------------------------
    def _component_for_block(self, block: Block) -> TransformerBlockComponent:
        comp_id = self._block_to_comp[block]
        return self.transformer.get_component(comp_id)

    def _kv_generation_tokens(self) -> int:
        """Number of generated tokens so far (beyond the initial prompt)."""

        L = self.transformer.current_sequence_length
        return max(L - self.initial_seq_len, 0)

    def _compute_block_demand(self, block: Block) -> BlockDemand:
        comp = self._component_for_block(block)
        L = self.transformer.current_sequence_length
        generated = self._kv_generation_tokens()

        # Static + activation memory
        mem = comp.compute_memory_requirements(L)

        # KV cache only applies to attention heads
        if isinstance(comp, AttentionHeadComponent):
            kv = comp.compute_cache_memory(generated)
        else:
            kv = 0.0

        # FLOPs
        flops = comp.compute_flops(L)
        return BlockDemand(memory=mem + kv, compute=flops, kv_cache=kv)

    # ------------------------------------------------------------------
    # Public API: interval update & activation sizes
    # ------------------------------------------------------------------
    def update_interval(self) -> Dict[Block, BlockDemand]:
        """Advance the model by one UAVT interval and recompute all block demands."""

        # Advance sequence length by interval size
        self.transformer.advance_interval()

        # Recompute per-block demands
        for blk in self._block_to_comp:
            self.demands[blk] = self._compute_block_demand(blk)

        return self.demands

    def activation_size(self, upstream: Block, downstream: Block) -> float:
        """Approximate activation transfer size (GB) between two blocks.

        This mirrors DTIS's _estimate_transfer_size_global logic:
          - head -> proj:  L * head_dim * precision_bytes
          - proj -> ffn:   L * hidden_size * precision_bytes
          - other edges:   treat as passing hidden state: L * hidden_size * precision_bytes
        """

        L = self.transformer.current_sequence_length
        cfg = self.transformer.cfg

        if upstream.kind == "head" and downstream.kind == "proj":
            bytes_ = L * cfg.head_dim * cfg.precision_bytes
        elif upstream.kind == "proj" and downstream.kind == "ffn":
            bytes_ = L * cfg.hidden_size * cfg.precision_bytes
        else:
            bytes_ = L * cfg.hidden_size * cfg.precision_bytes
        return bytes_ / GB
