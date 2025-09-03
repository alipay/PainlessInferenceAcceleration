import torch 
import triton
import triton.language as tl

@triton.jit
def qk_norm_and_half_rope_forward_kernel(
    qkv_ptr,
    q_norm_weight_ptr, k_norm_weight_ptr, 
    freqs_ptr, 
    qo_ptr, ko_ptr, vo_ptr,
    indptr_ptr, offsets_ptr,
    stride,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    interleave: tl.constexpr,
    BLOCK: tl.constexpr
):
    batch_idx = tl.program_id(0)
    seq_block_idx = tl.program_id(1)

    seq_start = tl.load(indptr_ptr + batch_idx)
    seq_end = tl.load(indptr_ptr + batch_idx + 1)
    seq_len = seq_end - seq_start
    pos_offset = tl.load(offsets_ptr + batch_idx)

    seq_offsets = seq_block_idx * BLOCK + tl.arange(0, BLOCK)
    mask_seq = seq_offsets < seq_len
    pos_ids = pos_offset + seq_offsets  # [BLOCK]

    seq_offsets = seq_start + seq_offsets

    # DD is head_dim
    DD = D * 2

    freqs_offset = tl.arange(0, D) % d
    freqs = tl.load(
        freqs_ptr + pos_ids[:, None] * d + freqs_offset[None, :],
        mask=mask_seq[:, None],
        other=0.0
    )  # [BLOCK, D]

    cos = tl.cos(freqs)[:, None, :]
    sin = tl.sin(freqs)[:, None, :]
    signs = tl.arange(0,2).to(tl.float32)*2-1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D))
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D))
    q_ptr = qkv_ptr
    w = H//h
    
    if interleave:
        head_offs = tl.arange(0, H)+tl.arange(0, H)//w*2
    else:
        head_offs = tl.arange(0, H)

    # ------ q norm and rope ------
    q0 = tl.load(q_ptr + seq_offsets[:, None, None] * stride + DD * head_offs[None, :, None] + tl.arange(0, D)[None, None, :], mask=mask_seq[:, None, None], other=0.0)
    q1 = tl.load(q_ptr + seq_offsets[:, None, None] * stride + D + DD * head_offs[None, :, None] + tl.arange(0, D)[None, None, :], mask=mask_seq[:, None, None], other=0.0)
    rms = 1/tl.sqrt((tl.sum(q0*q0, 2) + tl.sum(q1*q1, 2)) / DD + eps)
    q1 *= rms[:, :, None]
    q1 *= q_weight_1[None, None, :]

    tl.store(qo_ptr + seq_offsets[:, None, None] * H * DD + D + DD * tl.arange(0, H)[None, :,None] + tl.arange(0, D)[None, None, :], q1, mask=mask_seq[:, None, None])


    q0 *= rms[:, :, None]
    q0 *= q_weight_0[None, None, :]
    qr = tl.reshape(tl.permute(tl.flip(tl.permute(tl.reshape(q0, (BLOCK, H, 2, d)), (0, 1, 3, 2)), dim=3) * signs, (0, 1, 3, 2)), (BLOCK, H, D))
    q0 = q0*cos + qr*sin
    tl.store(qo_ptr + seq_offsets[:, None, None] * H * DD + DD * tl.arange(0, H)[None, :,None] + tl.arange(0, D)[None, None, :], q0, mask=mask_seq[:, None, None])

    # ------ k norm and rope ------
    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D))
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D))
    if interleave:
        head_offs = tl.arange(0, h)*(w+2)
        k_ptr = qkv_ptr + DD * w
    else:
        head_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + DD * H


    k0 = tl.load(k_ptr + seq_offsets[:, None, None] * stride + DD * head_offs[None, :, None] + tl.arange(0, D)[None, None,:], mask=mask_seq[:, None, None], other=0.0)
    k1 = tl.load(k_ptr + seq_offsets[:, None, None] * stride + D + DD * head_offs[None, :, None] + tl.arange(0, D)[None, None,:], mask=mask_seq[:, None, None], other=0.0)

    rms = 1/tl.sqrt( (tl.sum(k0*k0, 2) + tl.sum(k1*k1, 2)) / DD + eps)
    k1 *= rms[:, :,None] 
    k1 *= k_weight_1[None, None, :]
    tl.store(ko_ptr + seq_offsets[:, None, None] * h * DD + D + DD * tl.arange(0, h)[None, : ,None] + tl.arange(0, D)[None, :], k1, mask=mask_seq[:, None, None])

    k0 *= rms[:, :,None]
    k0 *= k_weight_0[None, None, :]
    kr = tl.reshape(tl.permute(tl.flip(tl.permute(tl.reshape(k0, (BLOCK, h, 2, d)), (0, 1, 3, 2)), dim=3) * signs, (0,1, 3, 2)), (BLOCK, h, D))
    k0 = k0*cos + kr*sin
    tl.store(ko_ptr +  seq_offsets[:, None, None] * h * DD + DD * tl.arange(0, h)[None, :,None] + tl.arange(0, D)[None, None, :], k0, mask=mask_seq[:, None, None])

    if interleave:
        head_offs = tl.arange(0, h)*(w+2)
        v_ptr = qkv_ptr + DD * w + DD
    else:
        head_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + DD * H + DD * h

    v0 = tl.load(v_ptr + seq_offsets[:, None, None] * stride + DD * head_offs[None, :,None] + tl.arange(0, D)[None, None,:], mask=mask_seq[:, None, None], other=0.0)
    tl.store(vo_ptr + seq_offsets[:, None, None] * h * DD + DD * tl.arange(0, h)[None, :,None] + tl.arange(0, D)[None, None, :], v0, mask=mask_seq[:, None, None])

    v1 = tl.load(v_ptr + seq_offsets[:, None, None] * stride + D + DD * head_offs[None, :,None] + tl.arange(0, D)[None, None,:], mask=mask_seq[:, None, None], other=0.0)
    tl.store(vo_ptr + seq_offsets[:, None, None] * h * DD + D + DD * tl.arange(0, h)[None, :,None] + tl.arange(0, D)[None, None, :], v1, mask=mask_seq[:, None, None])



@triton.jit
def qk_norm_and_rope_forward_kernel(
    qkv_ptr,
    q_norm_weight_ptr, k_norm_weight_ptr, 
    freqs_ptr, 
    qo_ptr, ko_ptr, vo_ptr,
    indptr_ptr, offsets_ptr,
    stride,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr, # head_dim
    d: tl.constexpr, # half head_dim
    interleave: tl.constexpr,
    BLOCK: tl.constexpr
):
    batch_idx = tl.program_id(0)
    seq_block_idx = tl.program_id(1)

    seq_start = tl.load(indptr_ptr + batch_idx)
    seq_end = tl.load(indptr_ptr + batch_idx + 1)
    seq_len = seq_end - seq_start
    pos_offset = tl.load(offsets_ptr + batch_idx)

    seq_offsets = seq_block_idx * BLOCK + tl.arange(0, BLOCK)
    mask_seq = seq_offsets < seq_len
    pos_ids = pos_offset + seq_offsets  # [BLOCK]

    seq_offsets = seq_start + seq_offsets


    freqs_offset = tl.arange(0, D) % d
    freqs = tl.load(
        freqs_ptr + pos_ids[:, None] * d + freqs_offset[None, :],
        mask=mask_seq[:, None],
        other=0.0
    )  # [BLOCK, D]

    cos = tl.cos(freqs)[:, None, :]
    sin = tl.sin(freqs)[:, None, :]
    signs = tl.arange(0,2).to(tl.float32)*2-1

    q_weight = tl.load(q_norm_weight_ptr + tl.arange(0, D))
    q_ptr = qkv_ptr
    w = H//h
    
    if interleave:
        head_offs = tl.arange(0, H)+tl.arange(0, H)//w*2
    else:
        head_offs = tl.arange(0, H)

    # ------ q norm and rope ------
    q = tl.load(q_ptr + seq_offsets[:, None, None] * stride + D * head_offs[None, :, None] + tl.arange(0, D)[None, None, :], mask=mask_seq[:, None, None], other=0.0)
    rms = 1/tl.sqrt(tl.sum(q*q, 2) / D + eps)
    q *= rms[:, :, None]
    q *= q_weight[None, None, :]

    q_r = tl.reshape(tl.permute(tl.flip(tl.permute(tl.reshape(q, (BLOCK, H, 2, d)), (0, 1, 3, 2)), dim=3) * signs, (0, 1, 3, 2)), (BLOCK, H, D))
    q = q*cos + q_r*sin
    tl.store(qo_ptr + seq_offsets[:, None, None] * H * D + D * tl.arange(0, H)[None, :,None] + tl.arange(0, D)[None, None, :], q, mask=mask_seq[:, None, None])

    # ------ k norm and rope ------
    k_weight = tl.load(k_norm_weight_ptr + tl.arange(0, D))
    if interleave:
        head_offs = tl.arange(0, h)*(w+2)
        k_ptr = qkv_ptr + D * w
    else:
        head_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + D * H


    k = tl.load(k_ptr + seq_offsets[:, None, None] * stride + D * head_offs[None, :, None] + tl.arange(0, D)[None, None,:], mask=mask_seq[:, None, None], other=0.0)

    rms = 1/tl.sqrt(tl.sum(k*k, 2) / D + eps)
    k *= rms[:, :,None] 
    k *= k_weight[None, None, :]
    k_r = tl.reshape(tl.permute(tl.flip(tl.permute(tl.reshape(k, (BLOCK, h, 2, d)), (0, 1, 3, 2)), dim=3) * signs, (0, 1, 3, 2)), (BLOCK, h, D))
    k = k*cos + k_r*sin
    tl.store(ko_ptr + seq_offsets[:, None, None] * h * D + D * tl.arange(0, h)[None, : ,None] + tl.arange(0, D)[None, :], k, mask=mask_seq[:, None, None])

    if interleave:
        head_offs = tl.arange(0, h)*(w+2)
        v_ptr = qkv_ptr + D * w + D
    else:
        head_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + D * H + D * h

    v = tl.load(v_ptr + seq_offsets[:, None, None] * stride + D * head_offs[None, :,None] + tl.arange(0, D)[None, None,:], mask=mask_seq[:, None, None], other=0.0)
    tl.store(vo_ptr + seq_offsets[:, None, None] * h * D + D * tl.arange(0, h)[None, :,None] + tl.arange(0, D)[None, None, :], v, mask=mask_seq[:, None, None])


"""
split qkv, apply norm to qk, apply rope to qk
qkv: [totol_tokens, kv_head*(q_head//kv_head + 2 ) * head_dim)]
indptr: [bs+1]
offsets: [bs]
"""
def triton_qk_norm_and_rope_forward(
    qkv, q_norm_weight, k_norm_weight, freqs, indptr, offsets, max_seq_len,
    q_head=32, kv_head=4, rotary_dim=None, eps=1e-6, interleave=False
):
    num_tokens, Dim = qkv.shape
    stride = qkv.stride(0)
    head_dim = Dim // (q_head + 2 * kv_head)
    dtype = qkv.dtype
    device = qkv.device

    qo = torch.empty((num_tokens, q_head, head_dim), dtype=dtype, device=device)
    ko = torch.empty((num_tokens, kv_head, head_dim), dtype=dtype, device=device)
    vo = torch.empty((num_tokens, kv_head, head_dim), dtype=dtype, device=device)

    bs = offsets.shape[0]

    num_stages = 3
    num_warps = 4
    block_size = 1 if max_seq_len < 4096 else 32

    grid = (bs, triton.cdiv(max_seq_len, block_size))

    if rotary_dim == head_dim // 2:
        qk_norm_and_half_rope_forward_kernel[grid](
            qkv,
            q_norm_weight, k_norm_weight,
            freqs,
            qo, ko, vo,
            indptr, offsets,
            stride,
            eps,
            q_head,
            kv_head,
            rotary_dim,
            rotary_dim // 2,
            interleave,
            BLOCK=block_size,
            num_stages=num_stages,
            num_warps=num_warps
        )
    elif rotary_dim == head_dim:
        qk_norm_and_rope_forward_kernel[grid](
            qkv,
            q_norm_weight, k_norm_weight,
            freqs,
            qo, ko, vo,
            indptr, offsets,
            stride,
            eps,
            q_head,
            kv_head,
            rotary_dim,
            rotary_dim // 2,
            interleave,
            BLOCK=block_size,
            num_stages=num_stages,
            num_warps=num_warps
        )
    else:
        raise ValueError(f"rotary_dim must be head_dim or head_dim // 2, but got {rotary_dim}")
    return qo, ko, vo


@triton.jit
def q_k_norm_and_half_rope_forward_kernel(
    q_ptr, k_ptr,
    q_norm_weight_ptr, k_norm_weight_ptr, 
    freqs_ptr, 
    qo_ptr, ko_ptr,
    indptr_ptr, offsets_ptr,
    q_stride_n,
    q_stride_h,
    k_stride_n,
    k_stride_h,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    group: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    interleave: tl.constexpr,
    BLOCK: tl.constexpr
):
    batch_idx = tl.program_id(0)
    seq_block_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    seq_start = tl.load(indptr_ptr + batch_idx)
    seq_end = tl.load(indptr_ptr + batch_idx + 1)
    seq_len = seq_end - seq_start
    pos_offset = tl.load(offsets_ptr + batch_idx)

    seq_offsets = seq_block_idx * BLOCK + tl.arange(0, BLOCK)
    mask_seq = seq_offsets < seq_len
    pos_ids = pos_offset + seq_offsets  # [BLOCK]

    seq_offsets = seq_start + seq_offsets

    # DD is head_dim
    DD = D * 2

    freqs_offset = tl.arange(0, D) % d
    freqs = tl.load(
        freqs_ptr + pos_ids[:, None] * d + freqs_offset[None, :],
        mask=mask_seq[:, None],
        other=0.0
    )  # [BLOCK, D]

    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = tl.arange(0,2).to(tl.float32)*2-1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D))
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D))
    
    # if interleave:
    #     head_offs = tl.arange(0, H)+tl.arange(0, H)//w*2
    # else:
    #     head_offs = tl.arange(0, H)

    # ------ q norm and rope ------
    head_offs = head_idx * group + tl.arange(0, group)
    mask_head = head_offs < H
    mask = mask_seq[:, None, None] & mask_head[None, :, None]
    q0 = tl.load(q_ptr + seq_offsets[:, None, None] * q_stride_n + q_stride_h * head_offs[None, :, None] + tl.arange(0, D)[None, None, :], mask=mask, other=0.0)
    q1 = tl.load(q_ptr + seq_offsets[:, None, None] * q_stride_n + D + q_stride_h * head_offs[None, :, None] + tl.arange(0, D)[None, None, :], mask=mask, other=0.0)
    rms = 1/tl.sqrt((tl.sum(q0*q0, 2) + tl.sum(q1*q1, 2)) / DD + eps)
    q1 *= rms[:, :, None]
    q1 *= q_weight_1[None, None, :]

    tl.store(qo_ptr + seq_offsets[:, None, None] * q_stride_n + D + q_stride_h * head_offs[None, :, None] + tl.arange(0, D)[None, None, :], q1, mask=mask)

    q0 *= rms[:, :, None]
    q0 *= q_weight_0[None, None, :]
    qr = tl.reshape(tl.permute(tl.flip(tl.permute(tl.reshape(q0, (BLOCK, group, 2, d)), (0, 1, 3, 2)), dim=3) * signs, (0, 1, 3, 2)), (BLOCK, group, D))
    q0 = q0*cos[:, None, :] + qr*sin[:, None, :]
    tl.store(qo_ptr + seq_offsets[:, None, None] * q_stride_n + q_stride_h * head_offs[None, :, None] + tl.arange(0, D)[None, None, :], q0, mask=mask)

    # ------ k norm and rope ------
    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D))
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D))


    if head_idx < h:
        k0 = tl.load(k_ptr + seq_offsets[:, None] * k_stride_n + k_stride_h * head_idx + tl.arange(0, D)[None,:], mask=mask_seq[:, None], other=0.0)
        k1 = tl.load(k_ptr + seq_offsets[:, None] * k_stride_n + D + k_stride_h * head_idx + tl.arange(0, D)[None,:], mask=mask_seq[:, None], other=0.0)

        rms = 1/tl.sqrt((tl.sum(k0*k0, 1) + tl.sum(k1*k1, 1)) / DD + eps)
        rms = rms[:, None]
        k1 *= rms
        k1 *= k_weight_1[None, :]
        tl.store(ko_ptr + seq_offsets[:, None] * k_stride_n + D + k_stride_h * head_idx + tl.arange(0, D)[None, :], k1, mask=mask_seq[:, None])

        k0 *= rms
        k0 *= k_weight_0[None, :]
        kr = tl.reshape(tl.permute(tl.flip(tl.permute(tl.reshape(k0, (BLOCK, 2, d)), (0, 2, 1)), dim=2) * signs, (0, 2, 1)), (BLOCK, D))
        k0 = k0*cos + kr*sin
        tl.store(ko_ptr +  seq_offsets[:, None] * k_stride_n + k_stride_h * head_idx + tl.arange(0, D)[None, :], k0, mask=mask_seq[:, None])


@triton.jit
def q_k_norm_and_rope_forward_kernel(
    q_ptr, k_ptr,
    q_norm_weight_ptr, k_norm_weight_ptr, 
    freqs_ptr, 
    qo_ptr, ko_ptr,
    indptr_ptr, offsets_ptr,
    q_stride_n,
    q_stride_h,
    k_stride_n,
    k_stride_h,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    group: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    interleave: tl.constexpr,
    BLOCK: tl.constexpr
):
    batch_idx = tl.program_id(0)
    seq_block_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    seq_start = tl.load(indptr_ptr + batch_idx)
    seq_end = tl.load(indptr_ptr + batch_idx + 1)
    seq_len = seq_end - seq_start
    pos_offset = tl.load(offsets_ptr + batch_idx)

    seq_offsets = seq_block_idx * BLOCK + tl.arange(0, BLOCK)
    mask_seq = seq_offsets < seq_len
    pos_ids = pos_offset + seq_offsets  # [BLOCK]

    seq_offsets = seq_start + seq_offsets

    freqs_offset = tl.arange(0, D) % d
    freqs = tl.load(
        freqs_ptr + pos_ids[:, None] * d + freqs_offset[None, :],
        mask=mask_seq[:, None],
        other=0.0
    )  # [BLOCK, D]

    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = tl.arange(0,2).to(tl.float32)*2-1

    q_weight = tl.load(q_norm_weight_ptr + tl.arange(0, D))

    # ------ q norm and rope ------
    head_offs = head_idx * group + tl.arange(0, group)
    mask_head = head_offs < H
    mask = mask_head[None, :, None] & mask_seq[:, None, None]
    q = tl.load(q_ptr + seq_offsets[:, None, None] * q_stride_n + head_offs[None, :, None] * q_stride_h + tl.arange(0, D)[None, None, :], mask=mask, other=0.0)
    rms = 1/tl.sqrt(tl.sum(q*q, 2) / D + eps)
    q *= rms[:, :, None]
    q *= q_weight[None, None, :]

    q_r = tl.reshape(tl.permute(tl.flip(tl.permute(tl.reshape(q, (BLOCK, group, 2, d)), (0, 1, 3, 2)), dim=3) * signs, (0, 1, 3, 2)), (BLOCK, group, D))
    q = q*cos[:, None, :] + q_r*sin[:, None, :]
    tl.store(qo_ptr + seq_offsets[:, None, None] * q_stride_n + head_offs[None, :, None] * q_stride_h + tl.arange(0, D)[None, None, :], q, mask=mask)

    # ------ k norm and rope ------
    k_weight = tl.load(k_norm_weight_ptr + tl.arange(0, D))

    if head_idx < h:
        k = tl.load(k_ptr + seq_offsets[:, None] * k_stride_n + k_stride_h * head_idx + tl.arange(0, D)[None, :], mask=mask_seq[:, None], other=0.0)

        rms = 1/tl.sqrt(tl.sum(k*k, 1) / D + eps)
        rms = rms[:, None]
        k *= rms
        k *= k_weight[None, :]
        k_r = tl.reshape(tl.permute(tl.flip(tl.permute(tl.reshape(k, (BLOCK, 2, d)), (0, 2, 1)), dim=2) * signs, (0, 2, 1)), (BLOCK, D))
        k = k*cos + k_r*sin
        tl.store(ko_ptr + seq_offsets[:, None] * k_stride_n + k_stride_h * head_idx + tl.arange(0, D)[None, :], k, mask=mask_seq[:, None])


def triton_q_k_norm_and_rope_forward(
    q, k, q_norm_weight, k_norm_weight, freqs, indptr, offsets, max_seq_len,
    q_head=32, kv_head=4, rotary_dim=None, eps=1e-6, interleave=False
):
    _, q_head, head_dim = q.shape
    kv_head = k.shape[1]
    q_stride_n = q.stride(0)
    q_stride_h = q.stride(1)
    k_stride_n = k.stride(0)
    k_stride_h = k.stride(1)
    group = q_head // kv_head

    bs = offsets.shape[0]

    num_stages = 3
    num_warps = 4
    block_size = 1 if max_seq_len == 1 else 32

    grid = (bs, triton.cdiv(max_seq_len, block_size), kv_head)

    if rotary_dim == head_dim // 2:
        q_k_norm_and_half_rope_forward_kernel[grid](
            q, k,
            q_norm_weight, k_norm_weight,
            freqs,
            q, k,
            indptr, offsets,
            q_stride_n,
            q_stride_h,
            k_stride_n,
            k_stride_h,
            eps,
            q_head,
            kv_head,
            group,
            rotary_dim,
            rotary_dim // 2,
            interleave,
            BLOCK=block_size,
            num_stages=num_stages,
            num_warps=num_warps
        )
    elif rotary_dim == head_dim:
        q_k_norm_and_rope_forward_kernel[grid](
            q, k,
            q_norm_weight, k_norm_weight,
            freqs,
            q, k,
            indptr, offsets,
            q_stride_n,
            q_stride_h,
            k_stride_n,
            k_stride_h,
            eps,
            q_head,
            kv_head,
            group,
            rotary_dim,
            rotary_dim // 2,
            interleave,
            BLOCK=block_size,
            num_stages=num_stages,
            num_warps=num_warps
        )
    else:
        raise ValueError(f"rotary_dim must be head_dim or head_dim // 2, but got {rotary_dim}")
    return q, k