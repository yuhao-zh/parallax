from scheduling.model_info import ModelInfo

MODEL_INFO_MAP = {
    "Qwen/Qwen3-0.6B-MLX-bf16": ModelInfo(
        model_name="Qwen/Qwen3-0.6B-MLX-bf16",
        head_size=64,
        hidden_dim=2880,
        intermediate_dim=2880,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=201088,
        num_layers=28,
        ffn_num_projections=3,
        num_local_experts=128,
        num_experts_per_tok=4,
        param_bytes_per_element=1,
        cache_bytes_per_element=2,
        embedding_bytes_per_element=2,
    )
}


def get_model_info(model_name):
    return MODEL_INFO_MAP.get(model_name, None)
