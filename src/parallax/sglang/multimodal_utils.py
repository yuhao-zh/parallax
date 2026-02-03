"""
Multimodal utilities for SGLang backend.

This module provides utilities for processing multimodal inputs in the Parallax
framework when using SGLang as the GPU backend.
"""

from typing import Any, List, Optional, Tuple

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs, MultimodalDataItem


def find_token_offsets(input_ids: List[int], token_id: int) -> List[Tuple[int, int]]:
    offsets = []
    start = None
    for i, tok in enumerate(input_ids):
        if tok == token_id:
            if start is None:
                start = i
        elif start is not None:
            offsets.append((start, i - 1))
            start = None
    if start is not None:
        offsets.append((start, len(input_ids) - 1))
    return offsets


def find_token_offsets_by_pair(
    input_ids: List[int],
    start_token_id: int,
    end_token_id: int,
) -> List[Tuple[int, int]]:
    start_indices = [i for i, tok in enumerate(input_ids) if tok == start_token_id]
    end_indices = [i for i, tok in enumerate(input_ids) if tok == end_token_id]
    
    offsets = []
    for start, end in zip(start_indices, end_indices):
        if start < end:
            # Content is between start+1 and end-1
            offsets.append((start + 1, end - 1))
    
    return offsets


def compute_mrope_positions(
    input_ids: List[int],
    image_grid_thw: Optional[torch.Tensor],
    mm_config: dict,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    import logging
    logger = logging.getLogger(__name__)
    
    seq_len = len(input_ids)
    
    def get_default_positions():
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(3, 1)
        delta = torch.zeros(1, dtype=torch.long)
        return positions, delta
    
    if image_grid_thw is None:
        return get_default_positions()
    
    image_token_id = mm_config.get("image_token_id")
    vision_start_id = mm_config.get("vision_start_token_id")
    video_token_id = mm_config.get("video_token_id")
    model_type = mm_config.get("model_type")
    vision_config = mm_config.get("vision_config", {})
    
    spatial_merge_size = (
        vision_config.get("spatial_merge_size") if isinstance(vision_config, dict) else None
    )
    tokens_per_second = (
        vision_config.get("tokens_per_second") if isinstance(vision_config, dict) else None
    )

    if image_token_id is None or spatial_merge_size is None:
        logger.debug(
            f"Missing mrope config: image_token_id={image_token_id}, "
            f"spatial_merge_size={spatial_merge_size}. Using default positions."
        )
        return get_default_positions()
    
    try:
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
        
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=spatial_merge_size,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_id,
            model_type=model_type,
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            tokens_per_second=tokens_per_second,
        )
        
        mrope_positions = mrope_positions.squeeze(1) 
        return mrope_positions, mrope_position_delta
        
    except Exception as e:
        logger.warning(f"Failed to compute mrope_positions: {e}. Using default positions.")
        return get_default_positions()


def prepare_sglang_multimodal_inputs(
    mm_items: List[MultimodalDataItem],
    image_grid_thw: Optional[torch.Tensor] = None,
    mm_config: Optional[dict] = None,
    input_ids: Optional[List[int]] = None,
) -> MultimodalInputs:
    mm_config = mm_config or {}
    
    # Extract token IDs from config
    image_token_id = mm_config.get("image_token_id")
    vision_start_id = mm_config.get("vision_start_token_id")
    vision_end_id = mm_config.get("vision_end_token_id")
    video_token_id = mm_config.get("video_token_id")
    audio_token_id = mm_config.get("audio_token_id")
    

    for item in mm_items:
        if item.pad_value is None:
            item.set_pad_value()
    
    mrope_positions = None
    mrope_position_delta = None
    
    if input_ids is not None:
        mrope_positions, mrope_position_delta = compute_mrope_positions(
            input_ids, image_grid_thw, mm_config
        )
    
    return MultimodalInputs(
        mm_items=mm_items,
        im_token_id=image_token_id,
        im_start_id=vision_start_id,
        im_end_id=vision_end_id,
        video_token_id=video_token_id,
        audio_token_id=audio_token_id,
        mrope_positions=mrope_positions,
        mrope_position_delta=mrope_position_delta,
    )
