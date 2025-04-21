

import torch
from flood.ops.draft import *


class Spec():
    def __init__(*args, **kwargs):
        pass

    def proposal_draft(self, input_ids, **kwargs):
        raise NotImplementedError

    def update_state(self, input_ids, **kwargs):
        raise NotImplementedError

    def verify_draft(self, input_ids, next_ids, **kwargs):
        raise NotImplementedError

    def update_cache(self, src_idx, dst_idx, caches, **kwargs):
        raise NotImplementedError


class Lookahead(Spec):
    def __init__(self, 
                 table_size=2 ** 20, 
                 branch_length=8, 
                 branch_count=32,
                 vocab_size=128256,
                 device=torch.device('cuda:0'),
                 tokenizer=None):
        
        assert 2 ** (int(round(math.log2(branch_length)))) == branch_length
        assert 2 ** (int(round(math.log2(branch_count)))) == branch_count

        self.table_size = table_size
        self.branch_length = branch_length
        self.branch_count = branch_count
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer  # used for debug

        self.freq_table = torch.zeros((table_size,), 
                                      dtype=torch.float32,
                                      device=device)
        self.draft_table = torch.zeros((table_size, branch_length), 
                                       dtype=torch.int32,
                                       device=device)
        torch._dynamo.mark_static_address(self.freq_table)
        torch._dynamo.mark_static_address(self.draft_table)

    def proposal_draft(self, input_ids, retrieve_count=4, **kwargs):
        output_tokens, output_masks = retrieve_draft_table(input_ids, 
                                                           self.freq_table,
                                                           self.draft_table, 
                                                           table_size=self.table_size,
                                                           vocab=self.vocab_size,
                                                           branch_length=self.branch_length,
                                                           branch_count=self.branch_count,
                                                           retrieve_count=retrieve_count)
        return output_tokens, output_masks

    def update_state(self, input_ids, **kwargs):
        update_draft_table(input_ids, 
                               self.freq_table, 
                               self.draft_table, 
                               table_size=self.table_size,
                               vocab=self.vocab_size,
                               branch_length=self.branch_length, 
                               branch_count=self.branch_count)


    def verify_draft(self, input_ids, next_ids, **kwargs):
        meta = kwargs['batch_meta_info']
        bs = meta.batch_size
        cache_offsets = meta.cache_indices.view(bs,-1)[:,0].contiguous()
        masks = None
        output_ids, cache_src_indices, cache_dst_indces = verify_draft(input_ids, 
                                                        next_ids, 
                                                        cache_offsets, 
                                                        masks, 
                                                        bs, 
                                                        meta.retrieve_count, 
                                                        self.branch_length)
        return output_ids, cache_src_indices, cache_dst_indces

    
    def update_cache(self, src_idx, dst_idx, caches, **kwargs):
        for i in range(caches.num_layers):
            cache = caches.caches[i]
            if src_idx.device != cache.device:
                src_idx = src_idx.to(cache.device)
            update_draft_cache(cache, src_idx, dst_idx)
