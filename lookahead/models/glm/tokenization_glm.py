# coding=utf-8
# Copyright 2022 shunxing1234 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from shutil import copyfile
from typing import Optional, Tuple, List, Union, Dict, Any

import sentencepiece as spm
import torch
from transformers import (
    PreTrainedTokenizer, RobertaTokenizer, GPT2Tokenizer, BertTokenizer
)
from transformers.file_utils import to_py_obj
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.tokenization_utils_base import (
    BatchEncoding,
    TextInput,
    PreTokenizedInput,
    EncodedInput,
    TruncationStrategy
)
from transformers.utils import logging, PaddingStrategy, TensorType
from transformers.utils.generic import _is_torch_device

logger = logging.get_logger(__name__)


class GLMBatchEncoding(BatchEncoding):
    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).
        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
        Returns:
            [`BatchEncoding`]: The same instance after modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in self.data.items()}
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self


class GLMTokenizerMixin:
    @property
    def sop_token(self) -> Optional[str]:
        return "<|startofpiece|>"

    @property
    def sop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the start token in the vocabulary,
        used when training a model with autoregressive blank filling.
        """
        return self.convert_tokens_to_ids(self.sop_token)

    @property
    def eop_token(self) -> Optional[str]:
        return "<|endofpiece|>"

    @property
    def eop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end token in the vocabulary,
        used when training a model with autoregressive blank filling.
        """
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def mask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[MASK]")

    @property
    def gmask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[gMASK]")

    @property
    def smask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[sMASK]")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id, self.smask_token_id, self.gmask_token_id]

    @property
    def dblock_token_id(self) -> int:
        return self.convert_tokens_to_ids("[dBLOCK]")

    def encode_whitespaces(self, content):
        content = content.replace('\n', '<n>')
        for i in range(10, 0, -1):
            content = content.replace('\t' * i, f'<|tab_{i}|>')
        for i in range(50, 1, -1):
            content = content.replace(' ' * i, f'<|blank_{i}|>')
        return content

    def decode_whitespaces(self, content):
        content = content.replace('<n>', '\n')
        for i in range(10, 0, -1):
            content = content.replace(f'<|tab_{i}|>', '\t' * i)
        for i in range(50, 1, -1):
            content = content.replace(f'<|blank_{i}|>', ' ' * i)
        return content

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        # 中文标点
        try:
            if chr(cp) in [
                '。', '？', '！', '，', '、', '；', '：', '「', '」', '『', '』', '‘',
                '’', '“', '”', '（', '）', '〔', '〕', '【', '】', '—', '…', '–', '．',
                '《', '》', '〈', '〉'
            ]:
                return True
        except Exception:
            logger.exception('invalid codepoint')

        return False

    def decode_postprocess(self, text):
        """
        fix chinese punctualtion normalize to english counterpart
        """
        if len(text) <= 1:
            return text
        fix_puncts_map = {
            ',': '，', ':': '：', ';': '；', '!': '！', '?': '？', '(': '（', ')': '）', '-': '－', '%': '％', '¥': '￥'
        }
        fix_idxs = []
        for pu in fix_puncts_map:
            idx = 0
            while idx != -1 and idx < len(text):
                idx = text.find(pu, idx, len(text))
                if idx > 0:
                    # 中文在标点前面
                    if self._is_chinese_char(ord(text[idx - 1])) or idx - 1 in fix_idxs:
                        fix_idxs.append(idx)
                    # 中文在标点后面
                    elif idx < len(text) - 1 and (self._is_chinese_char(ord(text[idx + 1])) or idx + 1 in fix_idxs):
                        fix_idxs.append(idx)
                    idx += 1
                elif idx == 0:
                    if self._is_chinese_char(ord(text[idx + 1])) or idx + 1 in fix_idxs:
                        fix_idxs.append(idx)
                    idx += 1
        parts = []
        idx = 0
        last_idx = 0
        for idx in sorted(fix_idxs):
            parts.extend(text[last_idx: idx])
            parts.append(fix_puncts_map[text[idx]])
            last_idx = idx + 1
        if idx > 0 and idx < len(text) - 1:
            parts.append(text[idx + 1:])
        if len(parts) > 0:
            return ''.join(parts)
        return text

    def _build_input_for_multiple_choice(self, context, choices):
        context_id = context["input_ids"]
        if torch.is_tensor(context_id):
            context_id = context_id.tolist()

        division = len(context_id)
        mask_position = context_id.index(self.mask_token_id)

        token = torch.tensor(context_id, dtype=torch.long)
        attention_mask = [context["attention_mask"].expand(division, -1)]
        position_id = torch.arange(division, dtype=torch.long)
        block_position_id = torch.zeros(division, dtype=torch.long)

        choice_ids, choice_indices = [], []

        for choice_str in choices:
            choice = torch.tensor(self(choice_str, add_special_tokens=False, padding=False)['input_ids'],
                                  dtype=torch.long)
            choice_ids.append(choice)
            choice_indices.append(torch.arange(len(token), len(token) + len(choice), dtype=torch.long))
            attention_mask.append(torch.tril(torch.ones((len(choice), len(choice)), dtype=torch.long)))

            token = torch.cat((token, torch.tensor([self.sop_token_id], dtype=torch.long), choice[:-1]))
            position_id = torch.cat((position_id, torch.tensor([mask_position] * len(choice), dtype=torch.long)))
            block_position_id = torch.cat((block_position_id, torch.arange(1, 1 + len(choice), dtype=torch.long)))

        attention_mask = torch.block_diag(*attention_mask)
        attention_mask[division:, :division] = context["attention_mask"].unsqueeze(0)

        return {
            "input_ids": token,
            "position_ids": torch.stack((position_id, block_position_id)),
            "attention_mask": attention_mask,
            "choice_ids": choice_ids,
            "choice_indices": choice_indices
        }

    def _pad_batch(self, tokens, position_ids, attention_mask, max_seq_length):
        pad_length = max_seq_length - len(tokens)
        attention_mask = torch.nn.functional.pad(
            attention_mask,
            (0, pad_length, 0, pad_length),
            mode="constant",
            value=0,
        )
        tokens = torch.cat((tokens, torch.zeros(pad_length, dtype=torch.long)))
        position_ids = torch.cat((position_ids, position_ids[..., -1:].expand(-1, pad_length)), dim=-1)
        return tokens, position_ids, attention_mask

    def _collate(self, samples):
        TILE = 1
        length_to_pad = (max(map(lambda spl: len(spl["input_ids"]), samples)) + TILE - 1) // TILE * TILE

        token_batch, position_id_batch, attention_mask_batch = [], [], []
        choices_batch, choice_target_ids_batch = [], []

        for sample in samples:
            token, position_id, attention_mask = self._pad_batch(
                sample["input_ids"], sample["position_ids"], sample["attention_mask"], length_to_pad
            )
            token_batch.append(token)
            position_id_batch.append(position_id)
            attention_mask_batch.append(attention_mask)
            choices_batch.append(sample["choice_ids"])
            choice_target_ids_batch.append(sample["choice_indices"])
        return {
            "input_ids": torch.stack(token_batch),
            "position_ids": torch.stack(position_id_batch),
            "attention_mask": torch.stack(attention_mask_batch).unsqueeze(1),
            "choice_ids": choices_batch,
            "choice_indices": choice_target_ids_batch,
        }

    def build_inputs_for_multiple_choice(self, model_input: BatchEncoding, choices, max_length=None):
        samples = [{key: value[i] for key, value in model_input.items()} for i in range(len(model_input["input_ids"]))]
        samples = [self._build_input_for_multiple_choice(sample, choice) for sample, choice in
                   zip(samples, choices)]
        inputs = self._collate(samples)
        return GLMBatchEncoding(inputs)

    def build_inputs_for_generation(self, model_input: BatchEncoding, max_gen_length=512, targets=None, padding=False):
        mask_ids = self.mask_token_ids
        input_ids = model_input.input_ids
        batch_size, seq_length = input_ids.shape[:2]
        position_id, block_position_id = list(range(seq_length)), [0 for _ in range(seq_length)]
        position_ids, block_position_ids = [], []
        labels = None
        if targets is not None:
            is_batched = isinstance(targets, (list, tuple))
            targets = self(targets, add_special_tokens=False, padding=False).input_ids
            if not is_batched:
                targets = [targets]
            assert len(targets) == len(input_ids)
            targets = [(target + [self.eop_token_id])[:max_gen_length] for target in targets]
            if not padding:
                max_gen_length = max(map(len, targets))
            targets = [[self.sop_token_id] + target for target in targets]
            labels = [target[1:] for target in targets]
            targets = [target + [self.pad_token_id] * (max_gen_length + 1 - len(target)) for target in targets]
            labels = [label + [-100] * (max_gen_length - len(label)) for label in labels]
            targets = torch.tensor(targets, dtype=input_ids.dtype, device=input_ids.device)
            labels = torch.tensor(labels, dtype=input_ids.dtype, device=input_ids.device)
            labels = torch.cat((input_ids.new_full((batch_size, seq_length), -100), labels), dim=1)
        for i in range(batch_size):
            mask_positions = []
            for mask_id in mask_ids:
                mask_positions += (input_ids[i] == mask_id).nonzero(as_tuple=True)[0].tolist()
            if not mask_positions:
                raise ValueError("Cannot find mask token in the input")
            mask_positions.sort()
            mask_pos = mask_positions[0]
            position_ids.append(position_id + [mask_pos] * max_gen_length)
            block_position_ids.append(block_position_id + list(range(1, max_gen_length + 1)))
        position_ids = torch.tensor(position_ids, dtype=input_ids.dtype, device=input_ids.device)
        block_position_ids = torch.tensor(block_position_ids, dtype=input_ids.dtype, device=input_ids.device)
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        attention_mask = model_input.attention_mask
        attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length + max_gen_length, -1)
        generation_attention_mask = torch.cat([attention_mask.new_zeros((seq_length, max_gen_length)),
                                               torch.tril(attention_mask.new_ones((max_gen_length, max_gen_length)))],
                                              dim=0).unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = torch.cat((attention_mask, generation_attention_mask), dim=2)
        attention_mask = attention_mask.unsqueeze(1)
        if targets is None:
            input_ids = torch.cat((input_ids, input_ids.new_full((batch_size, 1), self.sop_token_id)), dim=-1)
        else:
            input_ids = torch.cat((input_ids, targets[:, :-1]), dim=1)
        batch = {"input_ids": input_ids, "position_ids": position_ids}
        if labels is None:
            batch["generation_attention_mask"] = attention_mask
        else:
            batch["attention_mask"] = attention_mask
            batch["labels"] = labels
        return BatchEncoding(batch)


class GLMRobertaTokenizer(RobertaTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    @property
    def gmask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support gMASK")

    @property
    def smask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support sMASK")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id]


class GLMChineseTokenizer(PreTrainedTokenizer, GLMTokenizerMixin):
    vocab_files_names = {"vocab_file": "cog-pretrain.model"}
    truncation_side: str = "left"

    def __init__(self, vocab_file, **kwargs):
        super().__init__(**kwargs)
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        self.vocab = self.get_vocab()
        # self.check_special_tokens()

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def check_special_tokens(self):
        '''
        bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token
        special tokens should init, check special token is not None
        '''
        for name, special_token in zip(
                ['bos', 'eos', 'unk', 'sep', 'pad', 'cls', 'mask'],
                [
                    self.bos_token, self.eos_token, self.unk_token,
                    self.sep_token, self.pad_token, self.cls_token, self.mask_token
                ]
        ):
            assert special_token is not None, f'should init special token [{name}] in tokenizer_config.json'

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.
        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.
        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.
        Returns:
            `List[str]`: The list of tokens.
        """
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok) for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)
        tokenized_text = []
        for idx, token in enumerate(tokens):
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                # 去除特殊字符后面token的 _,添加特殊字符后会引起这个改变，和之前保持一致
                part_tokens = self._tokenize(token)
                if idx > 0 and tokens[idx - 1] in no_split_token:
                    if part_tokens and part_tokens[0].startswith('▁') and len(part_tokens[0]) > 1:
                        new_token = part_tokens[0][1:]
                        if new_token in self.vocab:
                            part_tokens[0] = new_token
                tokenized_text.extend(part_tokens)
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)

    def prepare_for_tokenization(
            self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.
        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            kwargs:
                Keyword arguments to use for the tokenization.
        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        return (self.encode_whitespaces(text), kwargs)

    def encode(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
            add_special_tokens: bool = False,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            return_tensors: Optional[Union[str, TensorType]] = None,
            **kwargs,
    ) -> List[int]:
        """
        FROM: https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/tokenization_utils_base.py#L2274
        modify param:add_special_tokens to False
        """
        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    def decode(
            self,
            token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
            spaces_between_special_tokens: bool = False,
            **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.
        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.
        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.
        Returns:
            `str`: The decoded sentence.
        """
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)
        decode_text = self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )
        return self.decode_postprocess(
            self.decode_whitespaces(decode_text)
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        assert token_ids_1 is None
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + eos


class GLMGPT2Tokenizer(GPT2Tokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        assert token_ids_1 is None
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + eos


class GLMBertTokenizer(BertTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    @property
    def gmask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support gMASK")

    @property
    def smask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support sMASK")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id]


class GLMTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        if config_tokenizer_class == "GLMRobertaTokenizer":
            tokenizer_class = GLMRobertaTokenizer
        elif config_tokenizer_class == "GLMChineseTokenizer":
            tokenizer_class = GLMChineseTokenizer
        elif config_tokenizer_class == "GLMGPT2Tokenizer":
            tokenizer_class = GLMGPT2Tokenizer
        elif config_tokenizer_class == "GLMBertTokenizer":
            tokenizer_class = GLMBertTokenizer
        else:
            raise NotImplementedError("Not implemented tokenizer type:", config_tokenizer_class)
        return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
