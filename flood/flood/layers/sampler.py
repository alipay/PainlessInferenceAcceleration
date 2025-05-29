# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from flood.ops.sample import sample_from_logit


class Sampler(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits, batch_meta_info):
        reqs = batch_meta_info.reqs
        logit_counts = batch_meta_info.logit_counts

        # no sampling & no lookhead & no targeting
        if batch_meta_info.samplings is None and batch_meta_info.mode != 2:
            next_ids = logits.argmax(dim=-1).tolist()
            index = 0
            for i, req in enumerate(reqs):
                if logit_counts is not None and logit_counts[i] == 0:
                    continue
                req.output_ids.append(next_ids[index])
                index += 1
            return 

        # lookahead
        if batch_meta_info.mode == 2:
            next_ids = logits.argmax(dim=-1)
            output_ids, cache_src_indices, cache_dst_indices = \
                batch_meta_info.spec.verify_draft(batch_meta_info.input_ids, 
                                                  next_ids,
                                                  batch_meta_info=batch_meta_info)
            output_ids = [[y for y in x if y!=-1] for x in output_ids.tolist()]
            
            if False:
                accept_counts = [len(x) for x in output_ids]
                texts = batch_meta_info.spec.tokenizer.batch_decode(output_ids)
                print(f'{accept_counts=} {output_ids=} {texts=}')
                if tuple(output_ids[0]) == tuple([4891, 236, 228, 99497]):
                    print(f'{cache_src_indices=} {cache_dst_indices=}')

            for i, req in enumerate(reqs):
                req.output_ids.extend(output_ids[i])
            batch_meta_info.cache_src_indices = cache_src_indices
            batch_meta_info.cache_dst_indices = cache_dst_indices
            return 

        # NOTE: sampling with prob can not work together with targeting
        temperature = []
        top_k = []
        top_p = []
        min_p = []
        for i, sampling in enumerate(batch_meta_info.samplings):
            if sampling is None or temperature is None and top_k is None and top_p is None and min_p is None:
                temperature.append(1.0)
                top_k.append(1)
                top_p.append(1.0)
                min_p.append(0.0)
            else:
                temperature.append(sampling.temperature or 1.0)
                top_k.append(sampling.top_k or 1)
                top_p.append(sampling.top_p or 1.0)
                min_p.append(sampling.min_p or 0.0)
        max_top_k = max(top_k)
        if max_top_k > 1:
            temperature = torch.tensor(temperature, dtype=logits.dtype,
                                       device=logits.device)
            top_k = torch.tensor(top_k, dtype=torch.int32, device=logits.device)
            top_p = torch.tensor(top_p, dtype=logits.dtype,
                                 device=logits.device)
            min_p = torch.tensor(min_p, dtype=logits.dtype,
                                 device=logits.device)
            next_token_id_list = sample_from_logit(logits, temperature, top_k,
                                                   top_p, min_p,
                                                   max_top_k).tolist()
            for i, req in enumerate(reqs):
                if logit_counts is not None and logit_counts[i] == 0:
                    continue
                req.output_ids.append(next_token_id_list[i])
            return 


        # targeting
        argmax_ids = logits.argmax(dim=-1).tolist()
        next_token_id_list = [0] * batch_meta_info.batch_size

        voc = logits.size(-1)
        batch_indices = []
        if any([x is not None for x in batch_meta_info.samplings]):
            ppls = -torch.log_softmax(logits, dim=-1)
            index = 0
            for i, sampling in enumerate(batch_meta_info.samplings):
                if logit_counts is not None and logit_counts[i] == 0:
                    continue 

                req = batch_meta_info.reqs[i]
                ql = batch_meta_info.qls[i]

                if sampling is None or sampling.target_ids is None:
                    req.output_ids.append(argmax_ids[index])
                    if logit_counts is None or logit_counts[i] > 0:
                        index += 1
                    continue
                batch_indices.append(i)

                if isinstance(sampling.target_ids[0], list):
                    if req.done >= req.input_length:  # options
                        idx, _, target_ids = req.iterate_target()
                        target_ids = target_ids[:ql]
                        indices = [(j + index) * voc + target_ids[j] for j in
                                range(1, ql)]
                        ppl = sum(ppls.view(-1)[indices].tolist())
                        vs = req.output_ids[0]
                        vs[idx] += ppl
                        vs[idx] /= len(target_ids)
                        index += len(target_ids)
                    else:  # prefill
                        vs = [0.0]*len(req.target_ids)
                        first_token_ids = [x[0] for x in req.target_ids]
                        ppl = ppls[index, first_token_ids].tolist()  # i->index
                        for j in range(len(req.target_ids)):
                            vs[j] += ppl[j]
                        req.output_ids.append(vs)
                        index += 1
                elif isinstance(sampling.target_ids[0], int):
                    ppl = ppls[index, sampling.target_ids].tolist()  # i->index
                    req.output_ids.append(ppl)
                    index += 1

