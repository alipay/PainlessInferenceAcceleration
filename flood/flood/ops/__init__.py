from typing import Optional, Tuple, Union

import flood_cuda
import torch


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d,))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    flood_cuda.silu_and_mul(out, x)
    return out


def update_cache(
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        indices: torch.Tensor
):
    flood_cuda.update_cache(k_out, v_out,
                            key_states, value_states,
                            indices, key_states.size(0),
                            key_states.size(1) * key_states.size(2),
                            key_states.stride(0) // 8)


def update_fusion_cache(
        kv_out: torch.Tensor,
        kv_states: torch.Tensor,
        indices: torch.Tensor
):
    flood_cuda.update_fusion_cache(kv_out, kv_states,
                                   indices, kv_states.size(0),
                                   kv_states.size(1) * kv_states.size(2),
                                   kv_states.stride(0) // 8)


def quant_and_update_cache(
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        indices: torch.Tensor
):
    n_token = key_states.size(0)
    kv_dim = key_states.size(1) * key_states.size(2)
    q_dim = query_states.size(1) * query_states.size(2)
    group = q_dim // kv_dim
    kv_stride = key_states.stride(0) // 8
    q_stride = query_states.stride(0) // 8
    flood_cuda.quant_to_fp8_and_update_cache(q_out, k_out, v_out,
                                             query_states, key_states,
                                             value_states,
                                             indices,
                                             n_token, group, kv_dim, q_stride,
                                             kv_stride)


class RMSNorm(torch.nn.Module):

    def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size),
                                         requires_grad=False)

    def forward(self, x: torch.Tensor):
        y = torch.empty_like(x)
        flood_cuda.rmsnorm(x, self.weight, y, self.variance_epsilon)
        return y


def scaled_fp8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        scale_ub: Optional[torch.Tensor] = None,
        use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    assert (input.ndim == 2)
    shape: Union[Tuple[int, int], torch.Size] = input.shape
    out_dtype: torch.dtype = torch.float8_e4m3fn
    output = torch.empty(shape, device=input.device, dtype=out_dtype)

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1),
                                device=input.device,
                                dtype=torch.float32)
            flood_cuda.dynamic_per_token_scaled_fp8_quant(output, input, scale,
                                                          scale_ub)
        else:
            raise ValueError(
                "NOT implement for use_per_token_if_dynamic=False!")
    else:
        assert scale.numel() == 1
        flood_cuda.static_scaled_fp8_quant(output, input, scale)

    return output, scale


