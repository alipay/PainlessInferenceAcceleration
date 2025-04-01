# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math
import torch
import triton
import triton.language as tl



@triton.jit
def update_draft_table_kernel(
        tokens,
        freq_table,
        draft_table,
        token_count,
        stride,
        BRANCH_LENGTH: tl.constexpr,
        BRANCH_COUNT: tl.constexpr,
        BLOCK: tl.constexpr,
        SIZE: tl.constexpr
):
    bid = tl.program_id(0)
    indices = tl.arange(0, BRANCH_LENGTH)
    for i in range(BLOCK):
        if bid * BLOCK + i + 4 <= token_count:
            p0 = tl.load(tokens + bid * BLOCK + i, mask = bid * BLOCK + i < token_count, other=0)
            p1 = tl.load(tokens + bid * BLOCK + i + 1, mask = bid * BLOCK + i + 1 < token_count, other=0)
            uid = p0 * 8191 + p1
            bucket_idx = uid % SIZE
            branch = tl.load(tokens + bid * BLOCK + 2 + i + indices, 
                             mask=bid * BLOCK + 2 + i + indices < token_count, 
                             other=0)
            branch_uid = tl.sum(branch)
            hit = False

            # if bid==0:
            #     if i==0:
            #         tl.device_print('uid',uid)
            #         tl.device_print('SIZE',SIZE)
            #         tl.device_print('bucket_idx',bucket_idx)
            #         # tl.device_print('branch',branch)

            for j in range(BRANCH_COUNT):
                if not hit:
                    draft_branch = tl.load(
                        draft_table + (bucket_idx + j) * stride + indices)
                    draft_branch_uid = tl.sum(draft_branch)
                    freq = tl.load(freq_table + bucket_idx + j)

                    # if bid==0:
                    #     if i==0:
                    #         if j == 0:
                    #             tl.device_print('freq',freq)
                    #             # tl.device_print('draft_branch_uid',draft_branch_uid)

                    if branch_uid == draft_branch_uid:
                        tl.store(freq_table + bucket_idx + j, freq + 1.0)
                        hit = True
                    elif freq == 0:
                        tl.store(
                            draft_table + (bucket_idx + j) * stride + indices,
                            branch)
                        tl.store(freq_table + bucket_idx + j, 1.0)
                        hit = True

            if not hit:
                for j in range(BRANCH_COUNT):
                    freq = tl.load(freq_table + bucket_idx + j)
                    freq = freq / 2

                    if freq < 1:
                        if not hit:
                            tl.store(draft_table + (
                                        bucket_idx + j) * stride + indices,
                                     branch)
                            tl.store(freq_table + bucket_idx + j, 1.0)
                            hit = True
                    else:
                        tl.store(freq_table + bucket_idx + j, freq)


def update_draft_table(tokens, 
                       freq_table, 
                       draft_table, 
                       table_size=2 ** 16, 
                       branch_length=8,
                       branch_count=8, 
                       vocab=100000, 
                       eos=0):
    # tokens: list of token_id, 
    # table: [sum(sizes),(1+length)], table_meta: [32(frq),32*length]
    token_count = len(tokens)
    # min seg: 2 prefix + 2 draft
    if token_count <= 3:
        return
    tokens = torch.tensor(tokens + [eos] * (branch_length - 2),
                          device=draft_table.device, 
                          dtype=draft_table.dtype)
    block = 32
    grid = lambda META: ((token_count - 4) // block + 1,)
    update_draft_table_kernel[grid](
        tokens,
        freq_table,
        draft_table,
        token_count,
        draft_table.stride(0),
        BRANCH_LENGTH=branch_length,
        BRANCH_COUNT=branch_count,
        BLOCK=block,
        SIZE=table_size,
        num_warps=1,
        num_stages=1
    )


@triton.jit
def retrieve_draft_table_kernel(
        output_tokens,
        tokens,
        freq_table,
        draft_table,
        stride,
        BRANCH_COUNT: tl.constexpr,
        BRANCH_LENGTH: tl.constexpr,
        RETRIEVE_COUNT: tl.constexpr,
        SIZE: tl.constexpr
):
    bid = tl.program_id(0)
    indices = tl.arange(0, BRANCH_LENGTH)

    p0 = tl.load(tokens + bid * 2)
    p1 = tl.load(tokens + bid * 2 + 1)
    uid = p0 * 8191 + p1
    bucket_idx = uid % (SIZE - BRANCH_COUNT)

    hit = False
    max_retry = 8
    hit_count = 0
    l = RETRIEVE_COUNT * BRANCH_LENGTH
    min_freq = 1024.0
    for i in range(max_retry):
        if not hit:
            min_freq = tl.exp2(1.0 * max_retry - 1.0 * i - 2)  # [64, 0.5]
            hit_count = 0
            for j in range(BRANCH_COUNT):
                if hit_count < RETRIEVE_COUNT:
                    freq = tl.load(freq_table + bucket_idx + j)
                    if freq >= min_freq:
                        hit_count += 1

            if hit_count >= RETRIEVE_COUNT:
                idx = 0
                for j in range(BRANCH_COUNT):
                    freq = tl.load(freq_table + bucket_idx + j)
                    if freq >= min_freq:
                        draft_branch = tl.load(
                            draft_table + (bucket_idx + j) * stride + indices)
                        tl.store(
                            output_tokens + bid * (l + 1) + 1 + idx * BRANCH_LENGTH + indices,
                            draft_branch)
                        idx += 1
                hit = True

    if hit_count < RETRIEVE_COUNT:
        idx = 0
        for j in range(BRANCH_COUNT):
            freq = tl.load(freq_table + bucket_idx + j)
            if freq >= 0.5:
                draft_branch = tl.load(
                    draft_table + (bucket_idx + j) * stride + indices)
                tl.store(
                    output_tokens + bid * (l + 1) + 1 + idx * BRANCH_LENGTH + indices,
                    draft_branch)
                idx += 1


def retrieve_draft_table(tokens, 
                         freq_table, 
                         draft_table, 
                         table_size=2 ** 16,
                         branch_length=8, 
                         branch_count=8, 
                         retrieve_count=8):
    # tokens: list of [token_id_0,token_id_1], freq_table:[size]  draft_table: [size, length]
    batch_size = len(tokens)
    device = draft_table.device
    tokens = torch.tensor(tokens, 
                          device=device,
                          dtype=draft_table.dtype)
    assert retrieve_count <= branch_count and branch_length <= branch_length
    # output mask contains current token 
    l = retrieve_count * branch_length
    output_tokens = torch.zeros((batch_size, l + 1), 
                                device=device,
                                dtype=draft_table.dtype)
    output_tokens[:, 0] = tokens[:, 1]

    output_masks = torch.triu(
        torch.ones((batch_size, l, l), 
                    device=device, 
                    dtype=torch.int8),
        diagonal=1)
    for i in range(batch_size):
        for j in range(1, retrieve_count):
            output_masks[i,j * branch_length + 1:(j + 1) * branch_length + 1,
            1:j * branch_length + 1] = 1
    output_masks = output_masks.view(batch_size * l, l)

    grid = lambda META: (batch_size,)
    retrieve_draft_table_kernel[grid](
        output_tokens,
        tokens,
        freq_table,
        draft_table,
        draft_table.stride(0),
        BRANCH_COUNT=branch_count,
        BRANCH_LENGTH=branch_length,
        RETRIEVE_COUNT=retrieve_count,
        SIZE=table_size,
        num_warps=1,
        num_stages=1
    )
    return output_tokens[:,:-1].contiguous(), output_masks


@triton.jit
def verify_draft_kernel(tile_input_ids,
                        tile_next_ids,
                        output_ids,
                        cache_offsets,
                        cache_src,
                        cache_dst,
                        BRANCH_COUNT: tl.constexpr,
                        BRANCH_LEGNTH: tl.constexpr,
                        ):
    bid = tl.program_id(0)
    tile_stride = (BRANCH_LEGNTH + 1) * BRANCH_COUNT

    max_accept_count = 0
    max_accept_idx = 0
    for i in range(BRANCH_COUNT):
        accept = 0
        stop = False
        for j in range(BRANCH_LEGNTH):
            if not stop:
                input_id = tl.load(tile_input_ids + bid * tile_stride + i * (BRANCH_LEGNTH+1) + j + 1)
                next_id = tl.load(tile_next_ids + bid * tile_stride+ i * (BRANCH_LEGNTH+1) + j)
                if input_id == next_id:
                    accept += 1
                    if accept > max_accept_count:
                        max_accept_idx = i
                        max_accept_count = accept
                        # tl.debug_barrier()
                        # if bid == 0:
                        #     tl.device_print("max_accept_count",max_accept_count)
                        #     tl.device_print("accept",accept)
                        #     tl.device_print("max_accept_idx",max_accept_idx)

                else:
                    stop = True

    # if bid == 0:
    #     tl.device_print("max_accept_count",max_accept_count)
    #     tl.device_print("max_accept_idx",max_accept_idx)

    stop = False
    for j in range(BRANCH_LEGNTH):
        if not stop:
            input_id = tl.load(
                tile_input_ids + bid * tile_stride + max_accept_idx * (
                            BRANCH_LEGNTH + 1) + j + 1)
            next_id = tl.load(
                tile_next_ids + bid * tile_stride + max_accept_idx * (
                            BRANCH_LEGNTH + 1) + j)
            if j == 0:
                tl.store(output_ids + bid * (BRANCH_LEGNTH + 1),
                         next_id)

            if input_id == next_id:
                accept_id = tl.load(
                    tile_next_ids + bid * tile_stride + max_accept_idx * (
                            BRANCH_LEGNTH + 1) + j + 1)
                tl.store(output_ids + bid * (BRANCH_LEGNTH + 1) + j + 1,
                            accept_id)
                # should move caches of the accepted tokens
                if max_accept_idx != 0:
                    offset = tl.load(cache_offsets+bid)
                    tl.store(cache_src + bid * BRANCH_LEGNTH + j,
                    offset + BRANCH_LEGNTH * max_accept_idx + 1 + j)
                    tl.store(cache_dst + bid * BRANCH_LEGNTH + j,
                    offset + 1 + j)
            else:
                stop = True


def verify_draft(input_ids, next_ids, cache_offsets, masks, batch_size,
                 branch_count, branch_length):
    # tokens: list of [token_id,token_id], freq_table:[size]  draft_table: [size,length]
    assert input_ids.size(0) == batch_size * branch_count * branch_length
    device = input_ids.device
    output_ids = -1 * torch.ones((batch_size, branch_length + 1),
                                 device=device, 
                                 dtype=input_ids.dtype)
    cache_src_indices = -1*torch.ones((batch_size, branch_length),
                                     device=device,
                                     dtype=input_ids.dtype)
    cache_dst_indices = -1*torch.ones((batch_size, branch_length),
                                     device=device,
                                     dtype=input_ids.dtype)
    batch_input_ids = torch.reshape(input_ids, (batch_size, branch_count, branch_length))
    tile_input_ids = -1 * torch.ones(
        (batch_size, branch_count, branch_length + 1), 
        device=device,
        dtype=input_ids.dtype)
    tile_input_ids[:, :, 0] = batch_input_ids[:, 0:1, 0]
    tile_input_ids[:, :, 1:-1] = batch_input_ids[:, :, 1:]
    tile_input_ids[:, :-1, -1] = batch_input_ids[:, 1:, 0]

    batch_next_ids = torch.reshape(next_ids, (batch_size, branch_count, branch_length))
    tile_next_ids = -1 * torch.ones(
        (batch_size, branch_count, branch_length + 1),
        device=device,
        dtype=input_ids.dtype)
    tile_next_ids[:, :, 0] = batch_next_ids[:, 0:1, 0]
    tile_next_ids[:, :, 1:-1] = batch_next_ids[:, :, 1:]
    tile_next_ids[:, :-1, -1] = batch_next_ids[:, 1:, 0]

    # print(f'{tile_input_ids=} \n{tile_next_ids=}')

    # print(tile_input_ids)
    grid = lambda META: (batch_size,)
    verify_draft_kernel[grid](
        tile_input_ids,
        tile_next_ids,
        output_ids,
        cache_offsets,
        cache_src_indices,
        cache_dst_indices,
        branch_count,
        branch_length,
        num_warps=1,
        num_stages=1
    )
    # output_ids = output_ids.tolist()
    # return [[y for y in x if y != -1] for x in output_ids]
    return output_ids, cache_src_indices, cache_dst_indices


@triton.jit
def update_draft_cache_kernel(k_cache,
                              v_cache,
                              src_indices,
                              dst_indices,
                              dim,
                              token_count,
                              DIM: tl.constexpr
                              ):
    bid = tl.program_id(0)
    sm = tl.num_programs(0)
    token_per_sm = (token_count - 1) // sm + 1
    indices = tl.arange(0, DIM)
    for i in range(token_per_sm):
        if i + token_per_sm * bid < token_count:
            src_idx = tl.load(src_indices + token_per_sm * bid + i)
            dst_idx = tl.load(dst_indices + token_per_sm * bid + i)
            if src_idx >= 0:
                if dst_idx >= 0:
                    k_vals = tl.load(k_cache + src_idx * dim + indices,
                                    mask=indices < dim, other=0.0)
                    tl.store(k_cache + dst_idx * dim + indices, k_vals,
                            mask=indices < dim)
                    v_vals = tl.load(v_cache + src_idx * dim + indices,
                                    mask=indices < dim, other=0.0)
                    tl.store(v_cache + dst_idx * dim + indices, v_vals,
                            mask=indices < dim)


def update_draft_cache(k_cache, v_cache, src_indices, dst_indices, sm=78):
    num_head, head_dim = k_cache.size(1), k_cache.size(2)
    token_count = src_indices.size(0)
    dim = num_head * head_dim
    round_dim = 2 ** (int(math.log2(dim - 1) + 1))

    grid = lambda META: (sm,)
    update_draft_cache_kernel[grid](k_cache, 
                              v_cache,
                              src_indices, 
                              dst_indices,
                              dim,
                              token_count,
                              DIM=round_dim,
                              num_warps=4,
                              num_stages=3
                              )
