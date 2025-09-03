# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""

# The file is copied from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py


import functools
from types import NoneType
from typing import Any, Callable, Dict, Optional, Tuple, List, Union

import torch
import triton
import triton.language as tl

import flood_cuda
from flood.ops.quantization import scaled_fp8_quant, moe_kernel_quantize_input


@triton.jit
def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                          token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                          compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:,
                                                                        None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
      should be padded to a multiple of block_size,

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = ((max_num_tokens_padded + block_size - 1) // block_size) * block_size
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    expert_ids = torch.zeros((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)

    flood_cuda.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad)
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad



def invoke_fused_moe_kernel(A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: Optional[torch.Tensor],
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            config: dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            per_channel_quant: bool,
                            block_shape: Optional[list[int]] = None) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert (block_shape is None
                or triton.cdiv(B.size(-2), block_shape[0]) == B_scale.size(-2))
        assert (block_shape is None
                or triton.cdiv(B.size(-1), block_shape[1]) == B_scale.size(-1))

    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0),
                 A.size(0) * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.size(1), META['BLOCK_SIZE_N']), )

    config = config.copy()
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0],
                                                block_shape[1]))
    fused_moe_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0)
        if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1)
        if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0)
        if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2)
        if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1)
        if B_scale is not None and B_scale.ndim >= 2 else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        **config,
    )



# def get_config_file_name(E: int, N: int, dtype: Optional[str]) -> str:
#     device_name = current_platform.get_device_name().replace(" ", "_")
#     dtype_selector = "" if not dtype else f",dtype={dtype}"
#     return f"E={E},N={N},device_name={device_name}{dtype_selector}.json"


@functools.lru_cache
def get_moe_configs(E: int, N: int,
                    dtype: Optional[str]) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(E, N, dtype)

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info("Using configuration from %s for MoE layer.",
                        config_file_path)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    return None



def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[list[int]] = None,
) -> dict[str, int]:
    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        # num_stages=3 can cause triton.runtime.errors.OutOfResources
        # on ROCm, set it to 2 instead.
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3
        }
    elif M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    override_config: Optional[Dict[str, Any]] = None,
    is_marlin: bool = False,
    block_shape: Optional[list[int]] = None,
) -> dict[str, int]:
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        # block_n = block_shape[0] if block_shape else 0
        # block_k = block_shape[1] if block_shape else 0
        # configs = get_moe_configs(E, N, dtype, block_n, block_k)

        # if configs:
        #     # If an optimal configuration map has been found, look up the
        #     # optimal config
        #     config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        # else:
        #     # Else use the default config
        config = get_default_config(M, E, N, w1_shape[2], top_k, dtype,
                                    is_marlin, block_shape)
    return config


def fused_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device)

    flood_cuda.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# This is used by the Deepseek-V2 model
def grouped_topk(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool,
                 num_expert_group: int = 0,
                 topk_group: int = 0,
                 scoring_func: str = "softmax",
                 e_score_correction_bias: Optional[torch.Tensor] = None,
    ):
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(gating_output)
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group,
                                   -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)


    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def get_config_dtype_str(
        dtype: torch.dtype,
        use_int4_w4a16: Optional[bool] = False,
        use_int8_w8a16: Optional[bool] = False,
        use_fp8_w8a8: Optional[bool] = False,
        use_mxfp4_w4a4: Optional[bool] = False) -> Optional[str]:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_mxfp4_w4a4:
        return "mxfp4_w4a4"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None

# TODO (bnell): use scalar_type instead of bools?
def get_config_quant_dtype(
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    use_mxfp4_w4a4: bool,
) -> Union[None, torch.dtype, str]:
    if use_fp8_w8a8:
        return torch.float8_e4m3fn
    elif use_int8_w8a8:
        return torch.int8
    elif use_mxfp4_w4a4:
        return "mxfp4"
    return None

def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a4: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), (
            "Hidden size mismatch")
    elif use_mxfp4_w4a4:
        # 16bit activation and fp4x2 packed weight
        assert hidden_states.size(1) // 2 == w1.size(2), "hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}")

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
                                        use_int8_w8a16=use_int8_w8a16,
                                        use_int4_w4a16=use_int4_w4a16,
                                        use_mxfp4_w4a4=use_mxfp4_w4a4,
                                        dtype=hidden_states.dtype)

    qtype = get_config_quant_dtype(use_fp8_w8a8=use_fp8_w8a8,
                                   use_int8_w8a8=use_int8_w8a8,
                                   use_int8_w8a16=use_int8_w8a16,
                                   use_int4_w4a16=use_int4_w4a16,
                                   use_mxfp4_w4a4=use_mxfp4_w4a4)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        override_config=override_config,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, K),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.size(1)]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=qtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                 global_num_experts, expert_map))

        invoke_fused_moe_kernel(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                a1q_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                apply_router_weight_on_input,
                                top_k_num,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape)

        flood_cuda.silu_and_mul(intermediate_cache2,
                                    intermediate_cache1.view(-1, N))

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=qtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape)

        invoke_fused_moe_kernel(qintermediate_cache2,
                                w2,
                                intermediate_cache3,
                                a2q_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                not apply_router_weight_on_input,
                                1,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape)

        torch.sum(intermediate_cache3.view(*intermediate_cache3.size()),
                    dim=1,
                    out=out_hidden_states[begin_chunk_idx:end_chunk_idx])

    return out_hidden_states


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    activation: str = "silu",
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a4: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    scoring_func: str = 'softmax',
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - activation (str): The activation function to apply after the first
        MoE layer.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a8 (bool): If True, use int8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use matmul of int8 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - use_mxfp4_w4a4 (bool): If True, use matmul of OCP MXFP4 weight and
        OCP MXFP4 activation to compute the inner products for w1 and w2.
        Defaults to False.
    - global_num_experts (int): The total number of experts in the global
        expert space.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices 
        from the global expert space to the local expert space of the expert 
        parallel shard.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[list[int]]): Optional block size for block-wise
        quantization.
    - scoring_func (str): The scoring function used for routing. Typically 
        'softmax' or other alternatives like 'topk'. Defaults to 'softmax'.
    - e_score_correction_bias (Optional[torch.Tensor]): Optional bias tensor 
        applied to the expert scores before routing. 
    - routed_scaling_factor (Optional[float]): Scaling factor applied to the expert 
        scores before routing.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group, 
                                              scoring_func, e_score_correction_bias)
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)
    return fused_experts(hidden_states,
                         w1,
                         w2,
                         topk_weights,
                         topk_ids,
                         inplace=inplace,
                         override_config=override_config,
                         use_fp8_w8a8=use_fp8_w8a8,
                         use_int8_w8a16=use_int8_w8a16,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         a1_scale=a1_scale,
                         a2_scale=a2_scale,
                         block_shape=block_shape
                         )


class AutoExperts():
    @classmethod
    def from_pretrained(cls,
                        module_list,
                        hidden_size,
                        intermediate_size,
                        num_expert,
                        scoring_func='softmax',
                        num_expert_group=None,
                        topk_group=None,
                        config=None
                        ):
        dtype = None
        if config is not None and hasattr(config, 'quantization_config'):
            dtype = torch.float8_e4m3fn
        if isinstance(module_list, torch.nn.ModuleList):
            if dtype is None:
                return NativeExperts(module_list, scoring_func, num_expert_group, topk_group)
            elif dtype == torch.float8_e4m3fn:
                return Fp8Experts(module_list, scoring_func, num_expert_group, topk_group)
            else:
                raise ValueError(f"unknown dtype:{dtype}")
        else:
            if dtype is None:
                return StackExperts(hidden_size, intermediate_size, num_expert,
                                    dtype, config)
            elif dtype == torch.float8_e4m3fn:
                return StackFp8Experts(hidden_size, intermediate_size,
                                       num_expert, dtype, config)
            else:
                raise ValueError(f"unknown dtype:{dtype}")


class NativeExperts(torch.nn.ModuleList):
    def __init__(self, modules, scoring_func, num_expert_group, topk_group) -> None:
        super().__init__(modules)
        self.scoring_func = scoring_func
        self.use_grouped_topk = (num_expert_group is not None) and (topk_group is not None)
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group

    def _flood_patch_func(self):
        w1 = []
        w2 = []
        for expert in iter(self._modules.values()):
            w1.append(torch.cat(
                [expert.gate_proj.weight.data, expert.up_proj.weight.data],
                dim=0))
            w2.append(expert.down_proj.weight.data)
        self.w1 = torch.stack(w1, 0)
        self.w2 = torch.stack(w2, 0)
        self._modules.clear()

    def forward(self, hidden_states, router_logits, top_k, renormalize=True, e_score_correction_bias=None):
        return fused_moe(hidden_states,
                         self.w1,
                         self.w2,
                         router_logits,
                         top_k,
                         renormalize=renormalize,
                         inplace=True,
                         use_grouped_topk=self.use_grouped_topk,
                         num_expert_group=self.num_expert_group,
                         topk_group=self.topk_group,
                         use_fp8_w8a8=False,
                         w1_scale=None,
                         w2_scale=None,
                         a1_scale=None,
                         a2_scale=None,
                         scoring_func=self.scoring_func,
                         e_score_correction_bias=e_score_correction_bias,
                        )

class Fp8Experts(torch.nn.ModuleList):
    def __init__(self, modules, scoring_func, num_expert_group, topk_group) -> None:
        super().__init__(modules)
        self.scoring_func = scoring_func
        self.use_grouped_topk = (num_expert_group is not None) and (topk_group is not None)
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group

    def _flood_patch_func(self):
        w1 = []
        w2 = []
        w1_scale = []
        w2_scale = []
        a1_scale = []
        a2_scale = []

        for expert in iter(self._modules.values()):
            w1.append(torch.cat(
                [
                    expert.gate_proj.weight.data.view(torch.int8),
                    expert.up_proj.weight.data.view(torch.int8)
                ],
                dim=0
            ))
            w2.append(expert.down_proj.weight.data.view(torch.int8))

            if (hasattr(expert.gate_proj, 'weight_scale') and
                hasattr(expert.up_proj, 'weight_scale') and
                hasattr(expert.down_proj, 'weight_scale')
            ):
                w1_scale.append(torch.cat(
                    [
                        expert.gate_proj.weight_scale.data,
                        expert.up_proj.weight_scale.data
                    ],
                    dim=0
                ))
                w2_scale.append(expert.down_proj.weight_scale.data)
            elif (hasattr(expert.gate_proj, 'weight_scale_inv') and
                  hasattr(expert.up_proj, 'weight_scale_inv') and 
                  hasattr(expert.down_proj, 'weight_scale_inv')
                ):
                w1_scale.append(torch.cat(
                    [
                        expert.gate_proj.weight_scale_inv.data,
                        expert.up_proj.weight_scale_inv.data
                    ],
                    dim=0
                    ))
                w2_scale.append(expert.down_proj.weight_scale_inv.data)
            else:
                raise RuntimeError(
                    f"Expert {expert} does not have expected attributes: "
                    "'weight_scale' or 'weight_scale_inv' for all of "
                    "gate_proj, up_proj, and down_proj."
                    )
            if (expert.gate_proj.input_scale and
                expert.up_proj.input_scale and
                expert.down_proj.input_scale
                ):
                a1_scale.append(torch.max(
                    expert.gate_proj.input_scale.data,
                    expert.up_proj.input_scale.data
                ))
                a2_scale.append(expert.down_proj.input_scale.data)

        self.w1 = torch.stack(w1, 0).view(torch.float8_e4m3fn)
        self.w2 = torch.stack(w2, 0).view(torch.float8_e4m3fn)
        self.w1_scale = torch.stack(w1_scale, 0)
        self.w2_scale = torch.stack(w2_scale, 0)
        if a1_scale and a2_scale:
            self.a1_scale = torch.stack(a1_scale, 0)
            self.a2_scale = torch.stack(a2_scale, 0)
        else:
            self.a1_scale = None
            self.a2_scale = None

        self._modules.clear()

    def forward(self, hidden_states, router_logits, top_k, renormalize=True, e_score_correction_bias=None, routed_scaling_factor=None):
        return fused_moe(hidden_states,
                         self.w1,
                         self.w2,
                         router_logits,
                         top_k,
                         renormalize=renormalize,
                         inplace=True,
                         use_grouped_topk=self.use_grouped_topk,
                         num_expert_group=self.num_expert_group,
                         topk_group=self.topk_group,
                         use_fp8_w8a8=True,
                         w1_scale=self.w1_scale,
                         w2_scale=self.w2_scale,
                         a1_scale=self.a1_scale,
                         a2_scale=self.a2_scale,
                         block_shape=[128, 128],
                         scoring_func=self.scoring_func,
                         e_score_correction_bias=e_score_correction_bias                         
                        )

class StackExperts(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_expert, dtype,
                 config) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(
            torch.empty(num_expert, hidden_size, 2 * intermediate_size,
                        dtype=dtype))
        self.w2 = torch.nn.Parameter(
            torch.empty(num_expert, intermediate_size, hidden_size,
                        dtype=dtype))

    def _flood_patch_func(self):
        self.w1 = torch.nn.Parameter(self.w1.permute(0, 2, 1).contiguous())
        self.w2 = torch.nn.Parameter(self.w2.permute(0, 2, 1).contiguous())

    def forward(self, hidden_states, router_logits, top_k, renormalize=True):
        return fused_moe(hidden_states,
                         self.w1,
                         self.w2,
                         router_logits,
                         top_k,
                         renormalize=renormalize,
                         inplace=True,
                         use_fp8_w8a8=False,
                         w1_scale=None,
                         w2_scale=None,
                         a1_scale=None,
                         a2_scale=None)


class StackFp8Experts(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_expert, dtype,
                 config) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(
            torch.empty(num_expert, hidden_size, 2 * intermediate_size,
                        dtype=dtype))
        self.w2 = torch.nn.Parameter(
            torch.empty(num_expert, intermediate_size, hidden_size,
                        dtype=dtype))
        self.w1_scale = torch.nn.Parameter(
            torch.empty(num_expert, 2, dtype=torch.float32))
        self.w2_scale = torch.nn.Parameter(
            torch.empty(num_expert, 1, dtype=torch.float32))
        self.a1_scale = torch.nn.Parameter(
            torch.empty(num_expert, 1, dtype=torch.float32))
        self.a2_scale = torch.nn.Parameter(
            torch.empty(num_expert, 1, dtype=torch.float32))

    def _flood_patch_func(self):
        self.w1 = torch.nn.Parameter(self.w1.permute(0, 2, 1).contiguous())
        self.w2 = torch.nn.Parameter(self.w2.permute(0, 2, 1).contiguous())

    def forward(self, hidden_states, router_logits, top_k, renormalize=True):
        return fused_moe(hidden_states,
                         self.w1,
                         self.w2,
                         router_logits,
                         top_k,
                         renormalize=renormalize,
                         inplace=True,
                         use_fp8_w8a8=True,
                         w1_scale=self.w1_scale,
                         w2_scale=self.w2_scale,
                         a1_scale=self.a1_scale,
                         a2_scale=self.a2_scale)
