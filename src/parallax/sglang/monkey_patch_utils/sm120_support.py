"""
Monkey patch to support SM120 (RTX 5090 etc.) for GPT-OSS model with MXFP4 quantization.

This patch is based on PR #9992: https://github.com/sgl-project/sglang/pull/9992
It adds SM120 support for triton attention backend in GPT-OSS models.
"""

import logging

logger = logging.getLogger(__name__)


def apply_sm120_mxfp4_support_monkey_patch():
    """
    Apply monkey patch to support SM120 (RTX 50 series GPUs) for MXFP4 quantization.

    This patch modifies the mxfp4.py module to recognize SM120 GPUs and enable
    the same optimizations that are available for SM100 (Blackwell) GPUs.
    """
    try:
        import torch
        from sglang.srt.utils import is_cuda

        # Check if we're on CUDA
        if not is_cuda():
            logger.debug("Not on CUDA, skipping SM120 monkey patch")
            return

        # Check if SM120 is supported
        try:
            from sglang.srt.utils import is_sm120_supported

            _is_sm120_supported = is_sm120_supported()
        except ImportError:
            # is_sm120_supported might not exist in older versions of sglang
            # Define it ourselves based on PR #9992
            def is_sm120_supported(device=None) -> bool:
                if not is_cuda():
                    return False
                return (torch.cuda.get_device_capability(device)[0] == 12) and (
                    torch.version.cuda >= "12.8"
                )

            _is_sm120_supported = is_sm120_supported()

        if not _is_sm120_supported:
            logger.debug("SM120 not detected, skipping SM120 monkey patch")
            return

        logger.info("Detected SM120 GPU (RTX 50 series), applying MXFP4 support patch...")

        # Patch the mxfp4 module
        import sglang.srt.layers.quantization.mxfp4 as mxfp4_module

        # Check if already patched
        if getattr(mxfp4_module, "_is_sm120_supported", False):
            logger.debug("SM120 patch already applied")
            return

        # Add the _is_sm120_supported flag
        mxfp4_module._is_sm120_supported = True

        # Patch _swizzle_mxfp4 function to support SM120
        original_swizzle_mxfp4 = mxfp4_module._swizzle_mxfp4

        def patched_swizzle_mxfp4(quant_tensor, scale, num_warps):
            """Patched _swizzle_mxfp4 that supports SM120"""
            import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
            from triton_kernels.numerics import InFlexData
            from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
            from triton_kernels.tensor_details import layout

            value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
            scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
                mx_axis=1, num_warps=num_warps
            )

            # This is the key change from PR #9992:
            # Use constraints for both SM100 and SM120
            _is_sm100_supported = getattr(mxfp4_module, "_is_sm100_supported", False)
            _is_sm120_supported = getattr(mxfp4_module, "_is_sm120_supported", False)

            if _is_sm100_supported or _is_sm120_supported:
                constraints = {
                    "is_persistent": True,
                    "epilogue_subtile": 1,
                }
                opt_flags.update_opt_flags_constraints(constraints)

            # transpose the tensor so that the quantization axis is on dim1
            quant_tensor = quant_tensor.transpose(-2, -1)
            scale = scale.transpose(-2, -1)
            quant_tensor = convert_layout(
                wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
            )
            scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
            return quant_tensor, InFlexData(), scale

        mxfp4_module._swizzle_mxfp4 = patched_swizzle_mxfp4

        # Also need to patch Mxfp4MoEMethod.create_weights to recognize SM120
        # for the intermediate_size_per_partition padding logic
        original_create_weights = mxfp4_module.Mxfp4MoEMethod.create_weights

        def patched_create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            with_bias=False,
            **extra_weight_attrs,
        ):
            """Patched create_weights that supports SM120"""
            from sglang.srt.utils import round_up, set_weight_attrs

            self.num_experts = num_experts
            weight_dtype = torch.uint8
            scale_dtype = torch.uint8
            self.with_bias = with_bias
            mxfp4_block = 32

            # pad the intermediate size to be a multiple of 2 * mxfp4_block
            # for to hold non-uniform sharded tensor as well as swizzling
            intermediate_size_per_partition_after_pad = intermediate_size_per_partition

            _is_sm100_supported = getattr(mxfp4_module, "_is_sm100_supported", False)
            _is_sm120_supported = getattr(mxfp4_module, "_is_sm120_supported", False)
            has_triton_kernels = getattr(mxfp4_module, "has_triton_kernels", False)

            # Key change: check for both SM100 and SM120
            if _is_sm100_supported or _is_sm120_supported:
                if self.use_flashinfer:
                    intermediate_size_per_partition_after_pad = round_up(
                        intermediate_size_per_partition, 256
                    )
                    hidden_size = round_up(hidden_size, 256)
                else:
                    intermediate_size_per_partition_after_pad = round_up(
                        intermediate_size_per_partition, 64
                    )
            elif has_triton_kernels:
                intermediate_size_per_partition_after_pad = round_up(
                    intermediate_size_per_partition, mxfp4_block
                )

            self.intermediate_size_per_partition = intermediate_size_per_partition_after_pad
            self.hidden_size = hidden_size

            # Fused gate_up_proj (column parallel)
            w13_weight = torch.nn.Parameter(
                torch.zeros(
                    layer.num_local_experts,
                    2 * intermediate_size_per_partition_after_pad,
                    hidden_size // 2,
                    dtype=weight_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight", w13_weight)
            set_weight_attrs(w13_weight, extra_weight_attrs)

            w13_weight_scale = torch.nn.Parameter(
                torch.zeros(
                    layer.num_local_experts,
                    2 * intermediate_size_per_partition_after_pad,
                    hidden_size // mxfp4_block,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)

            w13_weight_bias = torch.nn.Parameter(
                torch.zeros(
                    layer.num_local_experts,
                    2 * intermediate_size_per_partition_after_pad,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

            # down_proj (row parallel)
            w2_weight = torch.nn.Parameter(
                torch.zeros(
                    layer.num_local_experts,
                    hidden_size,
                    intermediate_size_per_partition_after_pad // 2,
                    dtype=weight_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight", w2_weight)
            set_weight_attrs(w2_weight, extra_weight_attrs)

            w2_weight_scale = torch.nn.Parameter(
                torch.zeros(
                    layer.num_local_experts,
                    hidden_size,
                    intermediate_size_per_partition_after_pad // mxfp4_block,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

            w2_weight_bias = torch.nn.Parameter(
                torch.zeros(layer.num_local_experts, hidden_size, dtype=torch.bfloat16),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

        mxfp4_module.Mxfp4MoEMethod.create_weights = patched_create_weights

        logger.info("Successfully applied SM120 MXFP4 support patch for GPT-OSS model")

    except ImportError as e:
        logger.warning(f"Could not apply SM120 MXFP4 patch due to import error: {e}")
    except Exception as e:
        logger.warning(f"Could not apply SM120 MXFP4 patch: {e}")
