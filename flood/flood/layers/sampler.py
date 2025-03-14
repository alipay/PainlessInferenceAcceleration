# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from flood.ops.sample import sample_from_logit


class Sampler(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits, batch_meta_info=None):
        if batch_meta_info is None or batch_meta_info.samplings is None:
            return logits.argmax(dim=-1).tolist()

        temperature = []
        top_k = []
        top_p = []
        min_p = []
        for i, sampling in enumerate(batch_meta_info.samplings):
            if sampling is None:
                temperature.append(1.0)
                top_k.append(1)
                top_p.append(1.0)
                min_p.append(0.0)
            else:
                temperature.append(sampling.temperature)
                top_k.append(sampling.top_k)
                top_p.append(sampling.top_p)
                min_p.append(sampling.min_p)
        
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
        else:
            next_token_id_list = [0] * batch_meta_info.batch_size

        voc = logits.size(-1)
        batch_indices = []
        if any([x.target_ids is not None for x in batch_meta_info.samplings]):
            ppls = -torch.log_softmax(logits, dim=-1)
            index = 0
            for i, sampling in enumerate(batch_meta_info.samplings):
                if sampling.target_ids is None:
                    index += 1
                    continue
                batch_indices.append(i)
                req = batch_meta_info.reqs[i]
                ql = batch_meta_info.qls[
                    i]  # options must be processed in decoding stage,[last_token]+options
                if isinstance(sampling.target_ids[0], list):
                    full_target_ids = sum(sampling.target_ids, [])
                    target_ids = (req.input_ids + full_target_ids)[
                                 req.done - ql:req.done]
                    indices = [(j + index) * voc + target_ids[j] for j in
                               range(ql)]
                    ppl = ppls.view(-1)[indices].tolist()
                    next_token_id_list[i] = ppl
                    index += len(indices)
                else:
                    ppl = ppls[i, sampling.target_ids].tolist()
                    next_token_id_list[i] = ppl
                    index += 1

        return next_token_id_list
