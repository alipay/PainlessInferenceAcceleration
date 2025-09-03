
import random
import math 
import torch 
from flood.utils.benchmark import benchmark_func
from flood.utils.benchmark import *

from flood.ops.seg_la import seg_la_fwd

from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla


class Meta:
    def __init__(self) -> None:
        pass

def torch_linear_attn(q, k, v, s, s_scales, decay_scales, causal=True, mask=None):
    q = q.float()
    k = k.float()
    v = v.float()
    s = s.float()
    bs, q_len, q_head, head_dim = q.shape
    k_head = k.shape[2]
    k_len = k.shape[1]
    assert q_len == k_len
    if mask is None:
        if causal:
            mask = torch.tril(
                torch.ones((q_len, k_len), dtype=q.dtype, device='cuda:0'),
                k_len - q_len)
        else:
            mask = torch.ones((q_len, k_len), dtype=q.dtype, device='cuda:0')

    softmax_scale = 1.0/math.sqrt(head_dim)
    query = q.transpose(1, 2) * softmax_scale  # [bs, head, len, dim]
    key = torch.permute(k, (0, 2, 3, 1))  # [bs, head, dim, len]
    value = v.transpose(1, 2)  # [bs, head, len, dim]
    if k_head != q_head:
        g = q_head // k_head
        key = torch.repeat_interleave(key, g, dim=1)
        value = torch.repeat_interleave(value, g, dim=1)

    arr = torch.arange(q_len, dtype=torch.float32, device=q.device)
    decay_matrix = arr.view(-1,1) - arr.view(1,-1)
    decay_matrix = torch.exp(decay_scales[:,None,None] * decay_matrix[None])
    decay_matrix = torch.tril(decay_matrix, 0)

    score = torch.matmul(query, key)
    score *= mask
    score *= decay_matrix[None]
    att = torch.matmul(score, value)

    decay_arr = torch.exp(decay_scales[:,None,None]*(arr[:,None]+1))
    att += torch.matmul(query*decay_arr, s*s_scales[:,None,None,None]) 

    att = torch.reshape(att.transpose(1, 2),
                        [bs, q_len, q_head, head_dim]).contiguous()

    decay_key = key*torch.exp(decay_scales[:,None,None]*(q_len-1-arr))
    state = decay_key@value

    return att, state

import torch.nn.functional as F
from einops import rearrange
def chunk_simple_gla_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    scale = None,
):
    q, k, v = map(lambda x: rearrange(x, 'b t h ... -> b h t ...'), [q, k, v])
    if g is not None:
        g = rearrange(g, 'b t h ... -> b h t ...')
    if scale is None:
        scale = 1.0 / q.shape[-1] ** 0.5

    T = q.shape[-2]
    BT = chunk_size
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, g = map(lambda x: x.to(torch.float32), [q, k, v, g])
    decay = g
    b, h, t, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    q, k, v, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, decay.unsqueeze(-1)])
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state.float()
    o = torch.zeros_like(v)
    for i in range(0, t // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i])
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_i
        S = S * decay[:, :, i, -1, None, None].exp() + \
            (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_i
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S

def get_seg_attn_meta(qls, kls, mask=None):
    device = 'cuda:0'
    bs = len(qls)
    q_offsets = [0]  # [bs+1]
    k_offsets = [0]  # [bs+1]
    q_lengths = []  # [bs]
    k_lengths = []  # [bs]
    max_q_length = max(qls)
    max_k_length = max(kls)
    for i, ql in enumerate(qls):
        kl = kls[i]
        q_offsets.append(q_offsets[-1] + ql)
        q_lengths.append(ql)
        k_offsets.append(k_offsets[-1] + kl)
        k_lengths.append(kl)

    q_offsets = torch.tensor(q_offsets, device=device, dtype=torch.int32)
    q_lengths = torch.tensor(q_lengths, device=device, dtype=torch.int32)
    s_offsets = torch.arange(bs, device=device, dtype=torch.int32)

    meta = Meta()
    meta.batch_size = bs
    meta.q_offsets = q_offsets
    meta.q_lengths = q_lengths
    meta.s_offsets = s_offsets
    meta.max_q_length = max_q_length
    meta.max_k_length = max_k_length
    meta.mask = mask
    meta.qls = qls
    meta.kls = kls
    return meta



def test_seg_attn(mode='prefill', even=True, decouple=False):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    qo_head = 16
    kv_head = 16
    dim = 128
    masks = None
    mask_size = 16
    torch_mask = None
    if mode == 'prefill':
        bs = 1
        if even:
            qls = [128]*bs
        else:
            qls = [1024+63]*bs
        kls = qls
        kls_fa = qls

    elif mode == 'decode':
        bs = 64
        qls = [1] * bs
        kls = qls
        kls_fa = [2048] * bs
    elif mode == 'mix':
        bs = 40
        qls = [(1024 - bs +1)] + [1] * (bs-1)
        kls = qls
    elif mode == 'spec':
        bs = 16
        qls = [mask_size] * bs
        kls = qls

        assert all([x==mask_size for x in qls])

        masks = torch.tril(torch.ones((bs, mask_size, mask_size), 
                                dtype=torch.int8,
                                device=device), 0)
        for i in range(mask_size):
            for j in range(mask_size):
                if j < i:
                    masks[:, i, j] = random.randint(0, 1)
        # print(f'{masks}')

        full_mask = torch.ones((bs, qls[0], kls[0] - qls[0]),
                                dtype=torch.float32,
                                device=device)
        torch_mask = -10000 * (1 - torch.cat([full_mask,masks.float()], dim=2))
        torch_mask = torch_mask[:,None]
    else:
        raise ValueError(f'unknown mode:{mode}')

    assert all([x<=kls[i] for i,x in enumerate(qls)])
    flops = 0


    qs = []
    ks = []
    vs = []
    ks_fa = []
    vs_fa = []
    flops = 0.0  # act as softmax attn
    for i, ql in enumerate(qls):
        kvl = kls[i]
        q = torch.randn(ql, qo_head, dim, dtype=dtype, device=device)
        k = torch.randn(kvl, kv_head, dim, dtype=dtype, device=device)
        v = torch.randn(kvl, kv_head, dim, dtype=dtype, device=device)
        qs.append(q)
        ks.append(k)
        vs.append(v)
        kvl_fa = kls_fa[i]
        k_fa = torch.randn(kvl_fa, kv_head, dim, dtype=dtype, device=device)
        v_fa = torch.randn(kvl_fa, kv_head, dim, dtype=dtype, device=device)
        ks_fa.append(k_fa)
        vs_fa.append(v_fa)

        flops += (ql * ql + ql * (
                    kvl - ql) * 2) * qo_head * dim * 2

    q = torch.cat(qs, 0)
    k = torch.cat(ks, 0)
    v = torch.cat(vs, 0)
    k_fa = torch.cat(ks_fa, 0)
    v_fa = torch.cat(vs_fa, 0)
    s = torch.randn(bs, kv_head, dim, dim, dtype=dtype, device=device)
    s_scales = torch.ones((bs,), device=device, dtype=dtype)

    decay_scales = -2**(-0.5  * torch.arange(1, qo_head+1, dtype=torch.float32, device=device))

    desc = f'mode:{mode} decouple:{decouple} bs:{len(qls)} q:{qls[0]} k:{kls[0]} qo_head:{qo_head} kv_head:{kv_head} dim:{dim}'

    seg_attn_meta = get_seg_attn_meta(qls, kls, mask=masks)
    seg_attn_meta.s_scales = s_scales

    flash_attn_meta = get_flash_attn_meta(qls, kls_fa, mask=masks)
    flash_attn = flash_attn_2


    if mode == 'prefill':
        org_output, state_ref = chunk_simple_gla(q.view(bs, qls[0], qo_head, dim), 
                                        k.view(bs, kls[0], kv_head, dim), 
                                        v.view(bs, kls[0], kv_head, dim),
                                        decay_scales.view(1,1,-1).expand(bs, qls[0], qo_head),
                                        initial_state=s,
                                        output_final_state=True)
        org_output = torch.reshape(org_output, (bs*qls[0], qo_head, dim))
    elif mode == 'decode':
        org_output, state_ref = fused_recurrent_simple_gla(q.view(bs, qls[0], qo_head, dim), 
                                k.view(bs, kls[0], kv_head, dim), 
                                v.view(bs, kls[0], kv_head, dim),
                                decay_scales.view(1,1,-1).expand(bs, qls[0], qo_head),
                                initial_state=s,
                                output_final_state=True)
        org_output = torch.reshape(org_output, (bs*qls[0], qo_head, dim))
    print(f'{state_ref.shape=}')
    torch.cuda.synchronize()

    opt_output = seg_la_fwd(q, k, v, s, decay_scales, seg_attn_meta, decouple=decouple)
    torch.cuda.synchronize()

    # print(org_output.shape, opt_output.shape)


    # print("org",org_output.dtype,org_output.shape)
    # print("opt",opt_output.dtype,opt_output.shape)
    errs = (opt_output.float() - org_output.float()).abs()
    err = errs.mean().item()
    amp = org_output.float().abs().mean().item()
    rate = err / amp

    state_errs = (state_ref.float() - s.float()).abs()
    state_err = state_errs.mean().item()
    state_amp = state_ref.float().abs().mean().item()
    state_rate = state_err / state_amp

    print(f"\n{desc} err:{err:.4f} output_rate:{rate:.4f} state_rate:{state_rate:.4f} flops:{flops:.4f}")
    n_repeat = 1000
    benchmark_func(flash_attn, q, k_fa, v_fa, flash_attn_meta, ref_flops=flops, n_repeat=n_repeat)

    if mode == 'prefill':
        benchmark_func(chunk_simple_gla, q.view(bs, qls[0], qo_head, dim),  k.view(bs, kls[0], kv_head, dim), v.view(bs, kls[0], kv_head, dim), decay_scales.view(1,1,-1).expand(bs, qls[0], qo_head), initial_state=s, output_final_state=True, ref_flops=flops, n_repeat=n_repeat)
    elif mode == 'decode':
        benchmark_func(fused_recurrent_simple_gla, q.view(bs, qls[0], qo_head, dim),  k.view(bs, kls[0], kv_head, dim), v.view(bs, kls[0], kv_head, dim), decay_scales.view(1,1,-1).expand(bs, qls[0], qo_head), initial_state=s, output_final_state=True, ref_flops=flops, n_repeat=n_repeat)

    benchmark_func(seg_la_fwd, q, k, v, s, decay_scales, seg_attn_meta, decouple=decouple, ref_flops=flops, n_repeat=n_repeat)
    

if __name__ == '__main__':
    for mode in ['prefill', 'decode']:
        for even in [True]:
            for decouple in [True, False]:
                test_seg_attn(mode=mode, even=even, decouple=decouple)