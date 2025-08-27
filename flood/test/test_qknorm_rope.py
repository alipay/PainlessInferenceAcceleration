import random
import torch
from flood.ops.rope import triton_qk_norm_and_rope_forward
from flood.utils.benchmark import benchmark_func

import flood_cuda

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    position_ids = position_ids.flatten()
    dim = cos.shape[-1]

    cos = cos.index_select(0, position_ids).reshape(q.shape[0], q.shape[1], 1, dim)
    sin = sin.index_select(0, position_ids).reshape(q.shape[0], q.shape[1], 1, dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rope_freqs(length, dim, rope_theta=10000.0):
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device='cuda:0').float() / dim))
    t = torch.arange(length, device='cuda:0', dtype=torch.int64).float()
    freqs = torch.outer(t, inv_freq)
    return freqs 

def torch_rope(q,k,rope_theta=10000.0, position_ids=None, rotary_dim=None, max_position_embeddings=4096):
    freqs = rope_freqs(max_position_embeddings, rotary_dim, rope_theta=rope_theta)
    freqs = torch.cat([freqs,freqs], -1)
    cos = freqs.cos().to(q.dtype)
    sin = freqs.sin().to(q.dtype)
    # position_ids = torch.arange(L, device='cuda:0')[:,None].expand(-1,B)
    qr, kr = apply_rotary_pos_emb(q[:,:,:,:rotary_dim], k[:,:,:,:rotary_dim], cos, sin, position_ids)
    qo = torch.cat([qr,q[:,:,:,rotary_dim:]],dim=-1)
    ko = torch.cat([kr,k[:,:,:,rotary_dim:]],dim=-1)
    return qo,ko

def torch_qk_norm(q,k, qw, kw, eps=1e-6):
    dtype = q.dtype
    L, B, H, D = q.shape 
    rms = torch.sqrt(q.float().square().mean(-1)+eps)
    q = q/rms[:,:,:,None]
    q = q*qw
    rms = torch.sqrt(k.float().square().mean(-1)+eps)
    k = k/rms[:,:,:,None]
    k = k*kw
    return q.to(dtype),k.to(dtype)

def torch_qk_norm_and_rope(qkv,qw,kw, position_ids, rope_theta=10000.0,H=32,h=4, rotary_dim=128, max_position_embeddings=4096, eps=1e-6,interleave=True):
    bs,length,dim = qkv.shape
    D = dim//(H+2*h)
    if interleave:
        qkv = qkv.view(bs, length, h, (2+H//h)*D)
        q,k,v = torch.split(qkv, [H//h*D, D, D], 3) 
        q = torch.reshape(q, (bs, length, H, D))
    else:
        qkv = qkv.view(bs, length, H+2*h, D)
        q,k,v = torch.split(qkv, [H,h,h], dim=2)
    q, k = torch_qk_norm(q, k, qw, kw, eps=eps)
    q, k = torch_rope(q, k, rope_theta=rope_theta, position_ids=position_ids, rotary_dim=rotary_dim, max_position_embeddings=max_position_embeddings)
    return q,k,v

def cuda_qk_norm_and_rope(qkv,qw,kw, indptr, offsets, q_head=32,kv_head=8, rotary_dim=128,rope_theta=10000.0, eps=1e-6,interleave=False):
    head_dim = qkv.shape[-1]//(q_head+2*kv_head)
    qkv = qkv.view(-1, q_head+2*kv_head, head_dim)
    q, k, v = qkv.split([q_head, kv_head, kv_head], dim=-2)
    q_y = torch.empty_like(q)
    flood_cuda.rmsnorm(q, qw, q_y, eps)
    k_y = torch.empty_like(k)
    flood_cuda.rmsnorm(k, kw, k_y, eps)

    flood_cuda.apply_rope_inplace(
        q_y, k_y, indptr, offsets, rotary_dim, False, 1.0, rope_theta)
    return q_y, k_y, v


def test_qk_norm_and_rope(B=2,ql=1024, max_position_embeddings=4096,H=32,h=8,D=128, rotary_dim=128,rope_theta=10000.0,interleave=False, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    kvl = 0
    qkv = torch.randn(B, ql, (H+2*h)*D,dtype=dtype,device=device)
    qw = torch.randn(D,dtype=dtype,device=device)
    kw = torch.randn(D,dtype=dtype,device=device)
    position_ids = torch.stack([torch.arange(0, kvl + ql)] * B, 0).to(device)
    indptr = torch.tensor(
        [i * ql for i in range(B + 1)], dtype=torch.int32, device=device
    )
    offsets = torch.full((B,), kvl, dtype=torch.int32, device=device)

    freqs = rope_freqs(max_position_embeddings, rotary_dim, rope_theta=rope_theta)

    q_ref,k_ref,v_ref = torch_qk_norm_and_rope(qkv,qw,kw, position_ids,rope_theta, H=H,h=h, rotary_dim=rotary_dim, max_position_embeddings=4096, eps=1e-6,interleave=interleave)
    
    qkv_clone = qkv.clone().reshape(-1, (H+2*h)*D).contiguous()
    qo,ko,vo = triton_qk_norm_and_rope_forward(qkv_clone,qw,kw,freqs, indptr, offsets, q_head=H,kv_head=h, rotary_dim=rotary_dim,eps=1e-6,interleave=interleave)
    
    qo = qo.view(B,ql,H,D)
    ko = ko.view(B,ql,h,D)
    vo = vo.view(B,ql,h,D)

    torch.testing.assert_close(vo, v_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(qo, q_ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(ko, k_ref, atol=1e-1, rtol=1e-2)

    qkv_clone = qkv.clone().reshape(-1, (H+2*h)*D).contiguous()
    qo, ko, vo = cuda_qk_norm_and_rope(qkv_clone,qw,kw,indptr, offsets, q_head=H,kv_head=h, rotary_dim=rotary_dim,rope_theta=rope_theta, eps=1e-6,interleave=interleave)
    qo = qo.view(B,ql,H,D)
    ko = ko.view(B,ql,h,D)
    vo = vo.view(B,ql,h,D)

    torch.testing.assert_close(vo, v_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(qo, q_ref, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(ko, k_ref, atol=1e-1, rtol=1e-2)

    if bench:
        print(f"{rotary_dim=}, head_dim={D}, {ql=}, {B=}")
        benchmark_func(triton_qk_norm_and_rope_forward, qkv_clone,qw,kw,freqs, indptr, offsets,q_head=H,kv_head=h, rotary_dim=rotary_dim, eps=1e-6, n_repeat=1000, ref_bytes=ql*B*(H+2*h)*D*4, n_profile=0, trace_dir='./triton_qknorm_rope.json')
        benchmark_func(cuda_qk_norm_and_rope, qkv_clone,qw,kw,indptr, offsets,q_head=H,kv_head=h, rotary_dim=rotary_dim, rope_theta=rope_theta, eps=1e-6, n_repeat=1000, ref_bytes=ql*B*(H+2*h)*D*4, n_profile=0, trace_dir='./cuda_qknorm_rope.json')


if __name__ == '__main__':

    for ql in [128, 256, 512, 1024, 2048, 4096]:
    # for ql in [256]:
        # partial rope
        test_qk_norm_and_rope(B=ql,ql=1, max_position_embeddings=4096,H=16,h=16,D=128, rotary_dim=64, bench=True,interleave=False)
        # full rope
        test_qk_norm_and_rope(B=ql,ql=1, max_position_embeddings=4096,H=16,h=16,D=128, rotary_dim=128, bench=True,interleave=False)

        print('-'*30)