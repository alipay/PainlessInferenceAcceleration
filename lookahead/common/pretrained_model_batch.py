# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from __future__ import print_function

import time
from operator import itemgetter
import inspect

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from collections import defaultdict
from functools import reduce
import pickle
import json
import copy
import warnings

import numpy as np

import torch
import torch.distributed as dist
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from transformers.generation import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.utils import (
    GenerateOutput,
    GreedySearchOutput,
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput)
# from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.utils import (
    SampleEncoderDecoderOutput,
    GreedySearchOutput,
    SampleOutput,
    BeamSearchOutput,
    BeamSampleOutput,
    ContrastiveSearchOutput,
    GenerateOutput,
    SampleEncoderDecoderOutput,
SampleDecoderOnlyOutput)



logger = logging.get_logger(__name__)


from common.configuration_utils import GenerationConfig
from common.pretrained_model import GenerationMode
from common.prefetch_cache import PrefetchCache


class LookaheadPreTrainedModel(PreTrainedModel):
    _batch_prefetch = True
    _stream_prefetch = False


    def __init__(self, config):
        super().__init__(config=config)

    def _get_generation_mode(
        self, generation_config: GenerationConfig, assistant_model: Optional["PreTrainedModel"]
    ) -> GenerationMode:
        """
        Returns the generation mode triggered by a [`GenerationConfig`] instance.
        """
        if generation_config.constraints is not None or generation_config.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif generation_config.num_beams == 1:
            if generation_config.do_sample is False:
                if (
                    generation_config.top_k is not None
                    and generation_config.top_k > 1
                    and generation_config.penalty_alpha is not None
                    and generation_config.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                elif generation_config.use_prefetch and generation_config.prefetch_size>0 and generation_config.prefetch_length>0 and generation_config.use_cache:
                    generation_mode = GenerationMode.LOOKAHEAD_GENERATION
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                if generation_config.use_prefetch and generation_config.prefetch_size>0 and generation_config.prefetch_length>0 and generation_config.use_cache:
                    generation_mode = GenerationMode.LOOKAHEAD_GENERATION
                else:
                    generation_mode = GenerationMode.SAMPLE
        else:
            if generation_config.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif generation_config.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        # Assisted generation may extend some generation modes
        if assistant_model is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
        return generation_mode


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
        """

        if synced_gpus is None:
            # if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            #     synced_gpus = True
            # else:
            #     synced_gpus = False
            synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    # warnings.warn(
                    #     "You have modified the pretrained model configuration to control generation. This is a"
                    #     " deprecated strategy to control generation and will be removed soon, in a future version."
                    #     " Please use a generation configuration file (see"
                    #     " https://huggingface.co/docs/transformers/main_classes/text_generation )"
                    # )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        model_kwargs['use_prefetch'] = generation_config.use_prefetch
        model_kwargs['debug_prefetch'] = generation_config.debug_prefetch
        model_kwargs['prefetch_size'] = generation_config.prefetch_size
        model_kwargs['prefetch_length'] = generation_config.prefetch_length
        model_kwargs['prefetch_mode'] = generation_config.prefetch_mode
        model_kwargs['do_sample'] = generation_config.do_sample
        model_kwargs['inputs_embeds_position'] = generation_config.inputs_embeds_position

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        # 7. determine generation mode
        generation_mode = self._get_generation_mode(generation_config, assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
            if assistant_model.config.is_encoder_decoder:
                assistant_model_kwargs = copy.deepcopy(model_kwargs)
                inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                    inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
                )
                assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, assistant_model_kwargs, model_input_name
                )
                model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

            # 12. run assisted generate
            return self.assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                do_sample=generation_config.do_sample,
                logits_processor=logits_processor,
                logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.LOOKAHEAD_GENERATION:
            # 11. run greedy search
            return self.lookahead_generation(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return self.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                sequential=generation_config.low_memory,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def lookahead_prepare_inputs_for_generation(self,
                                                input_ids,
                                                past_key_values=None,
                                                attention_mask=None,
                                                inputs_embeds=None,
                                                **kwargs):

        position_ids = kwargs.get("position_ids", None)

        prefetch_kwargs = kwargs.get('prefetch_kwargs', {})
        prefetch_cursors = prefetch_kwargs.get('prefetch_cursors', None)
        prefetch_size = prefetch_kwargs.get('prefetch_size', 63)
        prefetch_length = prefetch_kwargs.get('prefetch_length', 12)
        prefetch_mode = prefetch_kwargs.get('prefetch_mode', 'trie')
        batch_indices = prefetch_kwargs.get('batch_indices', None)

        if past_key_values is None:
            if inputs_embeds is not None and input_ids is not None:
                model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": input_ids}
                length = input_ids.size(1)
            elif input_ids is not None:
                model_inputs = {"input_ids": input_ids}
                length = input_ids.size(1)
            elif inputs_embeds is not None:
                model_inputs = {"inputs_embeds": inputs_embeds}
                length = input_ids.size(1)
            else:
                raise ValueError('either input_ids or inputs_embeds is not None')
            attention_mask = attention_mask[:, :, :length, :length]

            model_inputs.update(
                {"past_key_values": past_key_values,
                 "use_cache": kwargs.get("use_cache"),
                 "attention_mask": attention_mask,
                 "prefetch_kwargs": prefetch_kwargs
                 })

            if position_ids is not None:
                # full position_ids, truncate in forward call
                model_inputs["position_ids"] = position_ids

        else:
            if prefetch_mode == 'llma':
                cs = []
                for x in prefetch_cursors:
                    xs = list(range(x-7,x+1))
                    xs = [0]*(8-len(xs)) + xs
                    cs.append(xs)
                cs = torch.tensor(cs).to(input_ids.device)
            else:
                cs = torch.tensor([[x - 1, x] for x in prefetch_cursors]).to(input_ids.device)
            qids = torch.gather(input_ids, 1, cs)
            prefetch_qids = qids.tolist()

            if prefetch_mode in ('input', 'output'):
                mode = prefetch_mode
                prefetch_mode = 'trie'
            elif prefetch_mode == 'llma':
                mode = 'input'
            else:
                mode = 'mix'
            sub_prefetch_size = max(prefetch_size // len(prefetch_qids), 1)
            prefetch_ids, prefetch_masks, prefetch_sizes = self.prefetch_cache.bat_get(prefetch_qids,
                                                                                    prefetch_size=sub_prefetch_size,
                                                                                    prefetch_length=prefetch_length,
                                                                                    prefetch_cursors=prefetch_cursors,
                                                                                    mode=mode,
                                                                                    indices=batch_indices,
                                                                                    prefetch_mode=prefetch_mode)
            sizes = list(set([len(x) for x in prefetch_ids]))
            assert len(sizes) == 1
            prefetched = True if sizes[0] > 1 else False
            input_id_slice = torch.tensor(prefetch_ids).to(input_ids.device)

            min_cur = min(prefetch_cursors)
            prefetch_mask_tensor = torch.from_numpy(prefetch_masks[:, None].astype(np.int64)).to(input_ids.device)
            prefetch_attention_mask = torch.cat(
                [attention_mask[:, :, min_cur: min_cur + sizes[0], :min_cur], prefetch_mask_tensor], dim=-1)

            prefetch_kwargs.update({'prefetch_qids': prefetch_qids,
                                'prefetch_ids': prefetch_ids,
                               'prefetch_masks': prefetch_masks,
                               'prefetch_sizes': prefetch_sizes,
                               'prefetch_qids': prefetch_qids,
                               'prefetch_cursors': prefetch_cursors,
                               'batch_indices': batch_indices,
                               })

            model_inputs = {'prefetch_kwargs': prefetch_kwargs}

            model_inputs.update(
                {
                    "input_ids": input_id_slice,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": prefetch_attention_mask
                }
            )
            if position_ids is not None:
                positions = torch.sum(prefetch_attention_mask, 3).squeeze(1)[0]
                model_inputs["position_ids"] = self._get_position_ids(position_ids, positions)

        # print(f"{input_ids.shape=} {input_id_slice.shape=} {position_ids.shape=} {attention_mask.shape=} {attention_mask_cache.shape=}")

        return model_inputs


    def _get_position_ids(self, full_position_ids, positions):
        return full_position_ids[..., positions]


    def _lookahead_update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
            logits_processor: Optional[LogitsProcessorList] = None,
            input_ids:  Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        bs = input_ids.size(0)

        prefetch_kwargs = model_kwargs['prefetch_kwargs']
        prefetch_ids = prefetch_kwargs.get('prefetch_ids', [])
        eos = prefetch_kwargs.get('eos', 50005)
        device = outputs.logits.device
        dtype = outputs.logits.dtype

        encoding = model_kwargs.get("past_key_values", None) is None
        dls = []
        edls = []
        if encoding:
            # encoding stage
            past_key_values = outputs.past_key_values
            _, n_head, past_length, head_dim = past_key_values[0][0].size()
            max_gen_size = model_kwargs['attention_mask'].size(-1) - past_length
            kvs = []
            zeros = torch.zeros((bs, n_head, max_gen_size, head_dim), dtype=dtype).to(device)
            for k, v in past_key_values:
                kf = torch.cat([k, zeros], dim=2)
                vf = torch.cat([v, zeros], dim=2)
                kvs.append((kf, vf))
            past_key_values = kvs

            _, nt, nv = outputs.logits.shape
            next_tokens_scores = logits_processor(input_ids, outputs.logits[:, -1]).view(bs, 1, nv)

            input_ids = torch.cat([input_ids, torch.ones((bs, max_gen_size), dtype=torch.long).to(device) * eos], dim=1)
            prefetch_cursors = [past_length] * bs

            if prefetch_kwargs.get('do_sample', False):
                probs = nn.functional.softmax(next_tokens_scores, dim=-1)
                bs, nt, nv = probs.shape
                next_tokens = torch.multinomial(probs.view(bs * nt, nv), num_samples=1).view(bs, nt)
            else:
                next_tokens = torch.argmax(next_tokens_scores, dim=-1, keepdim=False).long()

            input_ids[:, past_length:past_length + 1] = next_tokens

            model_kwargs["past_key_values"] = past_key_values
            model_kwargs['next_tokens'] = next_tokens
            model_kwargs['next_token_list'] = next_tokens.tolist()
            model_kwargs['next_tokens_scores'] = next_tokens_scores
            dls.extend([1]*bs)
            edls.extend([1]*bs)

        else:
            prefetch_cursors = prefetch_kwargs.get('prefetch_cursors', None)
            min_cur = min(prefetch_cursors)
            max_cur = max(prefetch_cursors)
            prefetch_masks = prefetch_kwargs['prefetch_masks']
            prefetch_sizes = prefetch_kwargs['prefetch_sizes']

            # TODO: accurate logit_processor
            # next_tokens_scores = logits_processor(input_ids[:,:max_cur], outputs.logits)
            bs, nt, nv = outputs.logits.shape
            next_tokens_scores = logits_processor(input_ids[:, :max_cur].repeat(1, nt).view(bs*nt, -1),
                                                  outputs.logits.view(bs*nt, -1)).view(bs, nt, -1)

            if prefetch_kwargs.get('do_sample', False):
                probs = nn.functional.softmax(next_tokens_scores, dim=-1)
                next_tokens = torch.multinomial(probs.view(bs * nt, nv), num_samples=1).view(bs, nt)
            else:
                next_tokens = torch.argmax(next_tokens_scores, dim=-1, keepdim=False).long()

            next_token_list = next_tokens.tolist()
            update_next_token_list = [[] for _ in range(len(next_token_list))]
            for ib in range(bs):
                max_match_index = -1
                max_match_count = 0
                max_prefetch_ids_slice = None
                max_next_token_slice = None

                prefetch_ids_ = prefetch_ids[ib][1:]
                org_prefetch_length = len(prefetch_ids_)

                # if 50005 in prefetch_ids_:
                #     prefetch_ids_ = prefetch_ids_[:prefetch_ids_.index(50005)+1]

                cur = prefetch_cursors[ib]
                for i in range(len(prefetch_ids_)):

                    mask_indices, = np.nonzero(prefetch_masks[ib, i + 1, cur - min_cur + 1:])
                    if mask_indices.size == 0:
                        continue
                    prefetch_ids_slice = [prefetch_ids_[j] for j in mask_indices]
                    # next in logic rather than next in position
                    next_token_slice = [next_token_list[ib][0]] + [next_token_list[ib][j + 1] for j in mask_indices]

                    c = len(prefetch_ids_slice)
                    for j, p in enumerate(prefetch_ids_slice):
                        if next_token_slice[j] != p:
                            c = j
                            break

                    if c > max_match_count:
                        max_match_count = c
                        max_match_index = i
                    if c >= max_match_count:
                        max_prefetch_ids_slice = prefetch_ids_slice
                        max_next_token_slice = next_token_slice

                dls.append(org_prefetch_length+1)
                edls.append(max_match_count+1)
                # prefix_plus_count = prefetch_cursors[i_bs] + 2
                prefix_length = cur + 1
                if max_match_count > 0:
                    match_idx = np.nonzero(prefetch_masks[ib, max_match_index + 1, cur - min_cur + 1:])[0][
                                : max_match_count]

                    if len(prefetch_ids_) != max_match_count and max_match_index + 1 != max_match_count:
                        kv_idx = match_idx + prefix_length
                        kv_idx_tensor = torch.from_numpy(kv_idx).to(device)
                        for kv in model_kwargs["past_key_values"]:
                            k, v = kv
                            k[ib, :, prefix_length: prefix_length + max_match_count] = k[ib, :, kv_idx_tensor]
                            v[ib, :, prefix_length: prefix_length + max_match_count] = v[ib, :, kv_idx_tensor]
                    next_token_list_ = next_token_list[ib][0: 1] + [next_token_list[ib][x + 1] for x in match_idx]
                    update_next_token_list[ib] = next_token_list_ + (
                                org_prefetch_length - len(next_token_list_) + 1) * [-1]
                    next_tokens = torch.tensor(next_token_list_).to(device)
                    input_ids[ib, cur + 1: cur + max_match_count + 2] = next_tokens
                else:
                    # max_match_count = 0
                    next_token_list_ = next_token_list[ib][:1]
                    update_next_token_list[ib] = next_token_list_ + org_prefetch_length * [-1]
                    input_ids[ib, cur + 1] = next_token_list_[0]

                prefetch_cursors[ib] += max_match_count + 1

                if prefetch_kwargs.get('debug_prefetch', False):
                    lengths = np.sum(prefetch_masks[ib, :, cur - min_cur:], axis=1) - 1
                    larr = np.concatenate([lengths[:-1][(lengths[1:] - lengths[:-1]) <= 0], lengths[-1:]], axis=0)
                    ls = ','.join(larr.astype(np.int32).astype(np.str_))
                    prefetch_qids = prefetch_kwargs['prefetch_qids'][ib]
                    size_str = ','.join([str(x) for x in prefetch_sizes[ib]])
                    print(
                        f'ib:{ib}/{bs} size:{max_match_count}/{len(prefetch_ids_)} from:{size_str} '
                        f'length:{ls} index:{max_match_index} query:{prefetch_qids} '
                        f'prefetch:{max_prefetch_ids_slice} next:{max_next_token_slice}')
            model_kwargs['next_tokens'] = torch.tensor(update_next_token_list).to(device)
            model_kwargs['next_token_list'] = update_next_token_list
            model_kwargs['next_tokens_scores'] = []
        model_kwargs['input_ids'] = input_ids
        prefetch_kwargs['prefetch_cursors'] = prefetch_cursors
        prefetch_kwargs['dls'].extend(dls)
        prefetch_kwargs['edls'].extend(edls)
        model_kwargs['prefetch_kwargs'] = prefetch_kwargs
        return model_kwargs

    def _early_stop(self,
                    unfinished_sequences,
                    output_ids,
                    batch_indices,
                    model_kwargs):

        # unfinished_sequences, output_ids, batch_indices, model_kwargs

        prefetch_kwargs = model_kwargs['prefetch_kwargs']
        input_ids = model_kwargs['input_ids']

        unfinished_sequence_list = unfinished_sequences.tolist()
        unfinished_index_list = []
        for i, (seq, ) in enumerate(unfinished_sequence_list):
            if seq == 0:
                idx = batch_indices[i]
                output_ids[idx, :input_ids.size(-1)] = input_ids[i]
            else:
                unfinished_index_list.append(i)

        output_batch_indices = [batch_indices[i] for i in unfinished_index_list]

        bs = input_ids.size(0)
        finished_count = bs - len(unfinished_index_list)

        if finished_count > 0 and bs > 1 and finished_count != bs:
            unfinished_indices = torch.tensor(unfinished_index_list).to(unfinished_sequences.device)
            unfinished_sequences = unfinished_sequences[unfinished_indices]

            model_kwargs['input_ids'] = input_ids[unfinished_indices]
            model_kwargs['position_ids'] = model_kwargs['position_ids'][unfinished_indices]
            model_kwargs['attention_mask'] = model_kwargs['attention_mask'][unfinished_indices]
            prefetch_kwargs = model_kwargs['prefetch_kwargs']
            prefetch_cursors = prefetch_kwargs['prefetch_cursors']
            prefetch_kwargs['prefetch_cursors'] = [prefetch_cursors[i] for i in unfinished_index_list]
            batch_indices = prefetch_kwargs['batch_indices']
            prefetch_kwargs['batch_indices'] = [batch_indices[i] for i in unfinished_index_list]

            past_key_values = []
            for kv in model_kwargs['past_key_values']:
                k, v = kv
                k = k[unfinished_indices]
                v = v[unfinished_indices]
                past_key_values.append((k, v))
            model_kwargs['past_key_values'] = tuple(past_key_values)

        return unfinished_sequences, output_ids,output_batch_indices, model_kwargs


    def _update_cache(self, past_key_values, kv_idx, prefix_and_next_count=None, max_match_count=None, max_match_index=None):
        update_past_key_values = []
        for k, v in past_key_values:
            if max_match_index + 1 == max_match_count:
                k = k[:, :, :prefix_and_next_count + max_match_count]
                v = v[:, :, :prefix_and_next_count + max_match_count]
            else:
                k = torch.concat([k[:, :, :prefix_and_next_count], k[:, :, kv_idx]], 2)
                v = torch.concat([v[:, :, :prefix_and_next_count], v[:, :, kv_idx]], 2)
            update_past_key_values.append((k, v))
        return tuple(update_past_key_values)

    def lookahead_generation(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values

        if not hasattr(self, 'prefetch_cache'):
            self.prefetch_cache = PrefetchCache()

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        prefetch_kwargs = {
            'prefetch_size': model_kwargs.pop('prefetch_size', 63),
            'prefetch_length': model_kwargs.pop('prefetch_length', 12),
            'prefetch_mode': model_kwargs.pop('prefetch_mode', 'trie'),
            'use_prefetch': True,
            'debug_prefetch': model_kwargs.pop('debug_prefetch', False),
            'eos': eos_token_id[0] if eos_token_id is not None else 2,
            'do_sample':  model_kwargs.pop('do_sample', False),
            'edls': [],
            'dls': [],
            'fts': []
        }

        prefetch_size = prefetch_kwargs.get('prefetch_size', 63)
        stop_max_length = stopping_criteria.max_length or max_length
        attn_max_length = stop_max_length + prefetch_size
        attention_mask = model_kwargs.get('attention_mask', None)
        input_device = input_ids.device
        if attention_mask is None:
            bs = input_ids.size(0)
            full_attention_mask = torch.tril(torch.ones((bs,1,attn_max_length, attn_max_length),dtype=torch.long),
                                             0).to(input_device)
        elif len(attention_mask.shape) == 2:
            # from [bs, src_len] to [bs,1,max_len,max_len]
            bs, src_len = attention_mask.shape
            pad_len = attn_max_length - src_len
            attention_mask = attention_mask.long()
            if pad_len > 0:
                pad_mask = torch.ones((bs, pad_len), dtype=torch.long).to(attention_mask.device)
                attention_mask = torch.cat([attention_mask, pad_mask], 1)
            full_attention_mask = torch.tril(attention_mask[:,None,None].expand(-1,-1,attn_max_length,-1),0)
        elif len(attention_mask.shape) == 4:
            bs, _, src_len, tgt_len = attention_mask.shape
            attention_mask = attention_mask.long()
            if src_len < attn_max_length or tgt_len < attn_max_length:
                full_attention_mask = torch.tril(
                    torch.ones((bs, 1, attn_max_length, attn_max_length), dtype=torch.long),
                    0).to(input_device)
                full_attention_mask[:,:,:src_len,:tgt_len] = attention_mask
            else:
                full_attention_mask = attention_mask
        else:
            raise ValueError(f'unsupport attention_mask.shape:{attention_mask.shape}')
        model_kwargs['attention_mask'] = full_attention_mask
        prefetch_kwargs['max_length'] = stop_max_length
        prefetch_kwargs['eos'] = eos_token_id[0] if eos_token_id is not None else 2

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new_ones((input_ids.shape[0], 1))  # ones([bs, 1])

        # import time
        # pts = time.time()
        prefetch_length = prefetch_kwargs.get('prefetch_length', 8)
        prefetch_mode = prefetch_kwargs.get('prefetch_mode', 'trie')

        input_id_list = input_ids.tolist()
        # example format: [..., 50007, 50000, 50006]
        for i, ids in enumerate(input_id_list):
            ids = ids[1:-3]+[50005]  # TODO: only for antglm
            if prefetch_mode == 'llma':
                self.prefetch_cache.llma_put(ids, mode='input', idx=i)
            else:
                self.prefetch_cache.put(ids, prefetch_length=prefetch_length + 1, mode='input', idx=i)
        # pitv = time.time()-pts
        # print(f'prefetch_1:{round(pitv*1000,3)}ms')

        input_bs, input_length = input_ids.shape
        max_input_length = model_kwargs["position_ids"].size(-1)
        eos = prefetch_kwargs.get('eos', 2)
        output_ids = torch.cat(
            [input_ids, eos * torch.ones((input_bs, max_input_length - input_length), dtype=torch.long).to(input_ids.device)],
            dim=1)
        batch_indices = [i for i in range(input_bs)]
        prefetch_kwargs['batch_indices'] = batch_indices
        model_kwargs['input_ids'] = input_ids
        model_kwargs['prefetch_kwargs'] = prefetch_kwargs
        ts = time.time()

        # if use early stop func when batch size > 1
        # if use prefetch, use_early_stop must be true, or it will exceed max length and cause error
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            input_ids = model_kwargs.pop('input_ids', None)
            model_inputs = self.lookahead_prepare_inputs_for_generation(input_ids, **model_kwargs)
            prefetch_kwargs = model_inputs.pop('prefetch_kwargs', {})

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                prefetch_kwargs=prefetch_kwargs
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            model_kwargs['prefetch_kwargs'] = prefetch_kwargs

            model_kwargs = self._lookahead_update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                input_ids=input_ids,
                logits_processor=logits_processor
            )

            next_tokens = model_kwargs['next_tokens']
            next_tokens_scores = model_kwargs['next_tokens_scores']
            next_token_list = model_kwargs['next_token_list']

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            # input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            batch_indices = model_kwargs['prefetch_kwargs']['batch_indices']
            for k, tids in enumerate(next_token_list):
                tids = [x for x in tids if x != -1]
                batch_index = batch_indices[k]
                if prefetch_mode != 'llma':
                    self.prefetch_cache.stream_put(tids, prefetch_length=prefetch_length + 1, final=False, mode='output',
                                            idx=batch_index)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                # unfinished_sequences = unfinished_sequences.mul(
                #     next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                # )
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens[:, :, None].ne(eos_token_id_tensor).prod(dim=2).prod(dim=1, keepdim=True))

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            prefetch_cursors = prefetch_kwargs['prefetch_cursors']
            for i in range(input_ids.size(0)):
                cur = prefetch_cursors[i]
                if stopping_criteria(input_ids[i:i+1,:cur+1], None):
                    unfinished_sequences[i] = 0


            unfinished_sequences, output_ids, batch_indices, model_kwargs = self._early_stop(
                unfinished_sequences, output_ids, batch_indices, model_kwargs
            )
            te = time.time()
            prefetch_kwargs['fts'].append(te-ts)
            ts = te
            if len(batch_indices) == 0:
                for i in range(input_bs):
                    if prefetch_mode != 'llma':
                        self.prefetch_cache.stream_put([], prefetch_length=prefetch_length + 1, final=True, mode='output',
                                                idx=i)
                max_cur = max(prefetch_cursors)
                input_ids = output_ids[:, :max_cur+1]
                break


        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                scores = {'dls':model_kwargs['prefetch_kwargs']['dls'], 'edls': model_kwargs['prefetch_kwargs']['edls'], 'fts': model_kwargs['prefetch_kwargs']['fts']}
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)

        # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
        if self.config.is_encoder_decoder:
            base_model = getattr(self, self.base_model_prefix, None)

            # allow encoder kwargs
            encoder = getattr(self, "encoder", None)
            # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
            # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
            # TODO: A better way to handle this.
            if encoder is None and base_model is not None:
                encoder = getattr(base_model, "encoder", None)

            if encoder is not None:
                encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                model_args |= encoder_model_args

            # allow decoder kwargs
            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

        prefetch_kwargs = ['use_prefetch', 'prefetch_size', 'prefetch_length', 'debug_prefetch',
                            'prefetch_kwargs',
                           ]
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args and key not in prefetch_kwargs:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

