"""
ModelInfo class for scheduling.

Abstraction of transformer models containing key architectural configurations
used for FLOPs and I/O estimation. This information is later consumed by
nodes within the scheduling system to make informed layer allocation
and performance estimation decisions.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """
    Abstraction of transformer model architecture for scheduling decisions.
    Assuming all decoder layers have uniform computational and memory requirements
    """

    model_name: str
    head_size: int
    hidden_dim: int
    intermediate_dim: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    num_layers: int
    ffn_num_projections: int = 3
    num_local_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    # Default sequence lengths for IO calculations
    batch_size: int = 1
    target_seq_len: int = 1
    source_seq_len: int = 256
    # Default int8
    param_bytes_per_element: float = 1
    cache_bytes_per_element: int = 1
    embedding_bytes_per_element: int = 1

    @property
    def kv_dim(self) -> int:
        """Return key and value head dim."""
        return self.num_kv_heads * self.head_size

    @property
    def kv_cache_size(self) -> int:
        """Return size of KV cache in bytes."""
        return (
            2 * self.cache_bytes_per_element * self.kv_dim * self.batch_size * self.source_seq_len
        )

    @property
    def expected_num_activated_experts(self) -> Optional[int]:
        """Return expected number of activated experts."""
        num_tokens = self.batch_size * self.target_seq_len
        if self.num_local_experts is not None and self.num_experts_per_tok is not None:
            return int(
                self.num_local_experts
                * (1 - (1 - self.num_experts_per_tok / self.num_local_experts) ** num_tokens)
            )
        return None

    @property
    def decoder_layer_flops(self) -> int:
        """
        Estimate FLOPs per decoder layer including attention and FFN components.
        For GEMM: (M, K) @ (K, N) = (M, N), flops would be 2 * M * K * N;
        """
        # Attention
        # Q/O projections: (T, hidden_dim) @ (hidden_dim, hidden_dim)
        qo_flops = 2 * 2 * self.target_seq_len * self.hidden_dim * self.hidden_dim
        # K/V projections: (T, hidden_dim) @ (hidden_dim, kv_dim)
        kv_flops = 2 * 2 * self.target_seq_len * self.hidden_dim * self.kv_dim
        projection_flops = qo_flops + kv_flops

        # 'roof' estimation for GQA
        # score: (T, hidden_dim) @ (hidden_dim, S)
        score_flops = 2 * self.hidden_dim * self.target_seq_len * self.source_seq_len
        # weight: (T, S) @ (S, hidden_dim)
        attn_v_flops = 2 * self.hidden_dim * self.target_seq_len * self.source_seq_len

        attention_flops = projection_flops + score_flops + attn_v_flops

        # Dense FFN
        ffn_flops = (
            2
            * self.ffn_num_projections
            * self.target_seq_len
            * self.hidden_dim
            * self.intermediate_dim
        )
        # Sparse MoE FFN, if applicable
        if self.expected_num_activated_experts is not None:
            ffn_flops *= self.expected_num_activated_experts

        return self.batch_size * (attention_flops + ffn_flops)

    def decoder_layer_io_bytes(self, active: bool = True) -> int:
        """
        Estimate memory per decoder layer in bytes including params and kv cache.

        Args:
            active: Whether calculation is for roofline io latency or param size estimation.
                If True, MoE will based on expected number of activated experts,
                    and kv-cache will be included.
                If False, MoE will based on total number of experts and kv-cache will be excluded.
        """
        # Attention
        qo_params = self.param_bytes_per_element * self.hidden_dim * self.hidden_dim
        kv_params = self.param_bytes_per_element * self.hidden_dim * self.kv_dim
        attention_params = qo_params + kv_params

        # FFN
        ffn_params = (
            self.param_bytes_per_element
            * self.ffn_num_projections
            * self.hidden_dim
            * self.intermediate_dim
        )
        if active:
            if self.expected_num_activated_experts is not None:
                ffn_params *= self.expected_num_activated_experts
            kv_cache_size = self.kv_cache_size
        else:
            if self.num_local_experts is not None:
                ffn_params *= self.num_local_experts
            kv_cache_size = 0

        return round(ffn_params + kv_cache_size + attention_params)

    @property
    def embedding_io_bytes(self) -> int:
        """Estimate memory for embedding table (first layer)."""
        return self.embedding_bytes_per_element * self.vocab_size * self.hidden_dim

    @property
    def lm_head_flops(self) -> int:
        """Estimate FLOPs for lm_head (last layer GEMM)."""
        return 2 * self.target_seq_len * self.hidden_dim * self.vocab_size
