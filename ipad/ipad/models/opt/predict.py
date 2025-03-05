# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


from __future__ import print_function

import time
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import importlib
import copy
import types
import random
import warnings
import pandas as pd
import os 

import torch

import sys
# from transformers import AutoTokenizer, OPTForCausalLM

# model = OPTForCausalLM.from_pretrained("/mntnlp/nanxiao/opt-350m/opt-350m")
# tokenizer = AutoTokenizer.from_pretrained("/mntnlp/nanxiao/opt-350m/opt-350m")
# from transformers import AutoTokenizer
from transformers import AutoTokenizer, GPT2TokenizerFast
from modeling_opt import OPTForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/mntnlp/nanxiao/opt/opt7b")
model = OPTForCausalLM.from_pretrained("/mntnlp/nanxiao/opt/opt7b")

for layer in model.model.decoder.layers:
    layer.patch()

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

print('done!')
