import torch
import torch.nn.functional as F
from flood.ops.quantization import per_token_group_quant_fp8
from flood.ops.gemm import w8a8_block_fp8_matmul
from flood.layers.moe import fused_topk, fused_experts

def native_per_token_group_quant_fp8(x,
                                     group_size,
                                     eps=1e-10,
                                     dtype=torch.float8_e4m3fn):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch."""
    assert x.shape[-1] % group_size == 0, ("the last dimension of `x` must "
                                           "be divisible by `group_size`")
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1,
                        keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size, ))

    return x_q, x_s

def native_w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
    compute_type: torch.dtype = torch.float32,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization using native torch.
    It is agnostic to the input data type and can be used for both int8 and
    fp8 data types.

    It takes two input tensors `A` and `B` (int8) with scales `As` and
    `Bs` (float32).
    The output is returned in the specified `output_dtype`.
    """
    A = A.to(compute_type)
    B = B.to(compute_type)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N, )
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0], f"{n_tiles} == {Bs.shape[0]}"
    assert k_tiles == Bs.shape[1], f"{k_tiles} == {Bs.shape[1]}"

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=compute_type, device=A.device)

    A_tiles = [
        A[:, i * block_k:min((i + 1) * block_k, K)] for i in range(k_tiles)
    ]
    B_tiles = [[
        B[
            j * block_n:min((j + 1) * block_n, N),
            i * block_k:min((i + 1) * block_k, K),
        ] for i in range(k_tiles)
    ] for j in range(n_tiles)]
    C_tiles = [
        C[:, j * block_n:min((j + 1) * block_n, N)] for j in range(n_tiles)
    ]
    As_tiles = [As[:, i:i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C

def test_per_token_group_quant_fp8(num_tokens, d, dtype, group_size, seed):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype, device='cuda')

    ref_out, ref_scale = native_per_token_group_quant_fp8(x, group_size)
    out, scale = per_token_group_quant_fp8(x, group_size)

    assert torch.allclose(out.to(torch.float32),
                          ref_out.to(torch.float32),
                          rtol=0.15)
    assert torch.allclose(scale, ref_scale)


def test_w8a8_block_fp8_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device='cuda') - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device='cuda') - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32, device='cuda') * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device='cuda') * factor_for_scale

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size,
                                       out_dtype)
    out = w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    rel_diff = (torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))) /
                torch.mean(torch.abs(ref_out.to(torch.float32))))
    assert rel_diff < 0.001


def torch_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids,
                             block_shape):
    """Fused moe with block-wise quantization using native torch."""
    B, D = a.shape
    topk = topk_ids.size(1)
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_fp8(a, block_k)
    a_q = a_q.to(torch.float32)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_matmul(a_q[mask],
                                                 w1[i],
                                                 a_s[mask],
                                                 w1_s[i],
                                                 block_shape,
                                                 output_dtype=a.dtype)
            dim = inter_out.size(-1) // 2
            act_out = F.silu(inter_out[..., :dim]) * inter_out[..., dim:]
            act_out_q, act_out_s = native_per_token_group_quant_fp8(
                act_out, block_k)
            out[mask] = native_w8a8_block_matmul(act_out_q,
                                                 w2[i],
                                                 act_out_s,
                                                 w2_s[i],
                                                 block_shape,
                                                 output_dtype=a.dtype)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)

def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y

def per_block_cast_to_fp8(
    x: torch.Tensor,
    block_shape: list[int] = [128, 128],
) -> tuple[torch.Tensor, torch.Tensor]:
    block_m, block_n = block_shape
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((round_up(m, block_m), round_up(n, block_n)),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales

def make_test_weight(
    e: int,
    rows: int,
    cols: int,
    block_shape = [128, 128],
):
    w_16 = torch.randn((e, rows, cols), device="cuda", dtype=torch.bfloat16) / 15
    w_l = [None] * e
    w_s_l = [None] * e
    for idx in range(e):
        w_l[idx], w_s_l[idx] = per_block_cast_to_fp8(
            w_16[idx], block_shape)
    w = torch.stack(w_l)
    w_s = torch.stack(w_s_l)
    return w_16, w, w_s

def test_w8a8_block_fp8_fused_moe(M, N, K, E, topk, block_size, dtype, seed):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=dtype, device="cuda") / 10
    score = torch.randn((M, E), dtype=dtype, device="cuda")

    _, w1, w1_s = make_test_weight(e=E, rows=2 * N, cols=K, block_shape=block_size)
    _, w2, w2_s = make_test_weight(e=E, rows=K, cols=N, block_shape=block_size)

    topk_weights, topk_ids = fused_topk(a, score.float(), topk, False)

    # Set the context to avoid lots of warning spam.
    ref_out = torch_w8a8_block_fp8_moe(
        a,
        w1,
        w2,
        w1_s,
        w2_s,
        topk_weights,
        topk_ids,
        block_size,
    )

    out = fused_experts(
        a,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_s,
        w2_scale=w2_s,
        block_shape=block_size,
    )

    # 0.039 only needed for [40000-4608-7168-2-1-block_size852-dtype852-0]
    tol = 0.035 if M < 40000 else 0.039
    torch.testing.assert_close(out, ref_out, atol=tol, rtol=tol)


if __name__ == "__main__":
    # test_per_token_group_quant_fp8(256, 4096, torch.bfloat16, 128, 42)
    # test_w8a8_block_fp8_matmul(256, 2048, 4096, [128, 128], torch.bfloat16, 42)
    test_w8a8_block_fp8_fused_moe(256, 2048, 4096, E=128, topk=6, block_size=[128, 128], dtype=torch.bfloat16, seed=42)