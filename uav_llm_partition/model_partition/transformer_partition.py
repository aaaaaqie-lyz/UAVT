"""Detailed Transformer partitioning model, inspired by DTIS core.transformer.

This module mirrors the level of detail used in the
`Distributed-Transformer-Inference-Simulator` project for modeling
Transformer components, but adapted to the UAVT head-level, per-layer
partitioning setting.

It does **not** perform any scheduling or resource allocation – it only
provides a rich, DTIS-style description of:

- model configuration (dimensions, precision, sequence length),
- per-component memory / KV-cache / FLOPs,
- per-layer, per-head components and their identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

GB = 1024.0**3


@dataclass
class TransformerPartitionConfig:
    """Configuration for the partitioned Transformer model.

    This is conceptually similar to DTIS ``TransformerConfig``, but
    extended with:
    - ``num_layers``: number of decoder layers,
    - ``num_heads``: number of attention heads per layer,
    - ``hidden_size``: embedding / hidden dimension,
    - ``initial_sequence_length``: prompt length at t=0,
    - ``interval_tokens``: how many tokens are generated per interval
      in the UAVT simulator.
    """

    num_layers: int
    num_heads: int
    hidden_size: int
    initial_sequence_length: int
    interval_tokens: int
    precision_bytes: int = 2  # e.g. 2 for fp16, 4 for fp32
    head_dim_override: Optional[int] = None

    @property
    def head_dim(self) -> int:
        """Dimension per attention head (like DTIS ``head_dim``)."""

        if self.head_dim_override is not None and self.head_dim_override > 0:
            return int(self.head_dim_override)
        return max(int(self.hidden_size / max(self.num_heads, 1)), 1)


class TransformerBlockComponent:
    """Base class for a *single* logical component that can be partitioned.

    This is the DTIS-style abstraction; in UAVT we attach an explicit
    ``layer`` index and, for heads, a ``head_index``.
    """

    def __init__(
        self,
        cfg: TransformerPartitionConfig,
        layer: int,
        kind: str,
        head_index: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.layer = layer
        self.kind = kind  # "head", "proj", "ffn"
        self.head_index = head_index

    # -----------------------------
    # Identification helpers
    # -----------------------------
    @property
    def component_id(self) -> str:
        """String identifier similar to DTIS ``component_id``."""

        if self.kind == "head" and self.head_index is not None:
            return f"layer{self.layer}_head{self.head_index}"
        return f"layer{self.layer}_{self.kind}"

    # -----------------------------
    # DTIS-style API (must override)
    # -----------------------------
    def compute_memory_requirements(self, seq_len: int) -> float:
        """Activation + parameter memory (GB) at given sequence length."""

        raise NotImplementedError

    def compute_cache_memory(self, generated_tokens: int) -> float:
        """KV cache memory (GB). Default for non-heads is zero."""

        return 0.0

    def compute_flops(self, seq_len: int) -> float:
        """FLOPs needed at given sequence length."""

        raise NotImplementedError


class AttentionHeadComponent(TransformerBlockComponent):
    """Per-layer, per-head attention component (with KV cache)."""

    def __init__(self, cfg: TransformerPartitionConfig, layer: int, head_index: int):
        super().__init__(cfg, layer=layer, kind="head", head_index=head_index)

    def compute_memory_requirements(self, seq_len: int) -> float:
        """QKV activations + QKV weights, in GB.

        This mirrors DTIS ``AttentionHead.compute_memory_requirements``:
        - QKV activations: 3 * L * head_dim * precision_bytes
        - QKV weights:     3 * hidden_size * head_dim * precision_bytes
        """

        qkv = 3 * seq_len * self.cfg.head_dim * self.cfg.precision_bytes
        weights = 3 * self.cfg.hidden_size * self.cfg.head_dim * self.cfg.precision_bytes
        return (qkv + weights) / GB

    def compute_cache_memory(self, generated_tokens: int) -> float:
        """K/V cache for this head, GB.

        As in DTIS: cache ∝ generated_tokens * hidden_size.
        """

        if generated_tokens <= 0:
            return 0.0
        cache_bytes = generated_tokens * self.cfg.hidden_size * self.cfg.precision_bytes
        return cache_bytes / GB

    def compute_flops(self, seq_len: int) -> float:
        """QKV transform FLOPs + attention score FLOPs."""

        flops_qkv = 3 * seq_len * self.cfg.hidden_size * self.cfg.head_dim
        flops_attn = (seq_len**2) * self.cfg.head_dim
        return flops_qkv + flops_attn


class ProjectionComponent(TransformerBlockComponent):
    """Per-layer projection component."""

    def __init__(self, cfg: TransformerPartitionConfig, layer: int) -> None:
        super().__init__(cfg, layer=layer, kind="proj")

    def compute_memory_requirements(self, seq_len: int) -> float:
        bytes_ = seq_len * self.cfg.hidden_size * self.cfg.precision_bytes
        return bytes_ / GB

    def compute_flops(self, seq_len: int) -> float:
        return seq_len * (self.cfg.hidden_size**2)


class FFNComponent(TransformerBlockComponent):
    """Per-layer feed-forward network component."""

    def __init__(self, cfg: TransformerPartitionConfig, layer: int) -> None:
        super().__init__(cfg, layer=layer, kind="ffn")

    def compute_memory_requirements(self, seq_len: int) -> float:
        bytes_ = 4 * seq_len * self.cfg.hidden_size * self.cfg.precision_bytes
        return bytes_ / GB

    def compute_flops(self, seq_len: int) -> float:
        return 0.8 * seq_len * (self.cfg.hidden_size**2)


class PartitionedTransformer:
    """DTIS-style Transformer wrapper for UAVT partitioning.

    This class provides:
    - a complete list of per-layer components (heads, proj, ffn),
    - helpers to iterate over them,
    - utilities to compute total memory/FLOPs and per-component stats.

    It is **read-only** from the scheduler's perspective: no resource
    allocation is done here.
    """

    def __init__(self, cfg: TransformerPartitionConfig) -> None:
        self.cfg = cfg
        self.current_sequence_length = cfg.initial_sequence_length

        # Build all components: layer by layer
        self.components: Dict[str, TransformerBlockComponent] = {}
        for layer in range(cfg.num_layers):
            # Heads
            for h in range(cfg.num_heads):
                head = AttentionHeadComponent(cfg, layer=layer, head_index=h)
                self.components[head.component_id] = head
            # Projection + FFN
            proj = ProjectionComponent(cfg, layer=layer)
            ffn = FFNComponent(cfg, layer=layer)
            self.components[proj.component_id] = proj
            self.components[ffn.component_id] = ffn

    # --------------------------------------------------------------
    # Sequence length helpers
    # --------------------------------------------------------------
    def step_sequence(self) -> None:
        """Advance sequence length by one token (DTIS-style step)."""

        self.current_sequence_length += 1

    def advance_interval(self) -> None:
        """Advance sequence length by UAVT interval size."""

        self.current_sequence_length += self.cfg.interval_tokens

    # --------------------------------------------------------------
    # Component access
    # --------------------------------------------------------------
    def get_component(self, component_id: str) -> TransformerBlockComponent:
        return self.components[component_id]

    def get_all_components(self) -> List[TransformerBlockComponent]:
        return list(self.components.values())

    # --------------------------------------------------------------
    # Global stats
    # --------------------------------------------------------------
    def total_memory_gb(self, include_kv: bool = True) -> float:
        """Total memory across all components at current sequence length."""

        total = 0.0
        L = self.current_sequence_length
        generated = max(L - self.cfg.initial_sequence_length, 0)
        for comp in self.get_all_components():
            total += comp.compute_memory_requirements(L)
            if include_kv and isinstance(comp, AttentionHeadComponent):
                total += comp.compute_cache_memory(generated)
        return total

    def total_flops(self) -> float:
        """Total FLOPs across all components at current sequence length."""

        L = self.current_sequence_length
        return sum(comp.compute_flops(L) for comp in self.get_all_components())
