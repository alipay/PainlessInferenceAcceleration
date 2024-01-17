from transformers import AutoTokenizer
import torch
import time 

from pia.lookahead.models.mixtral.modeling_mixtral import MixtralForCausalLM

model_dir = "/mntnlp/common_base_model/Mixtral-8x7B-Instruct-v0.1"
# config = AutoConfig.from_pretrained('/mntnlp/common_base_model/Mixtral-8x7B-Instruct-v0.1')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = MixtralForCausalLM.from_pretrained(model_dir,
                                           trust_remote_code=False,
                                           low_cpu_mem_usage=True,
                                           torch_dtype=torch.float16,
                                           device_map="auto")


# prompt = "Hello, I'm am conscious and"
# inputs = tokenizer(prompt, return_tensors="pt")
# inputs['input_ids'] = inputs.input_ids.cuda()
# inputs['attention_mask'] = inputs.attention_mask.cuda()
# position_ids = None


# model(**inputs)

prompt = "Hello, I'm am conscious and"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.cuda()
attention_mask = inputs.attention_mask.cuda()
position_ids = None

for use_lookahead in [False, True]:
    debug_lookahead = False
    decoding_length = 63
    branch_length = 12
    ts = time.time()
    max_new_tokens = 256
    decoding_kwargs = {"use_lookahead": use_lookahead,
                       "debug_lookahead": debug_lookahead,
                       "decoding_mode": 'hier',
                       "decoding_length": decoding_length,
                       "branch_length": branch_length}
    outputs = model.generate(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             use_cache=True,
                             max_new_tokens=max_new_tokens,
                             repetition_penalty=1.0,
                             do_sample=False,
                             decoding_kwargs=decoding_kwargs
                             )
    output_ids = outputs
    input_length = input_ids.size(-1)
    output_ids = output_ids[0, input_length:].tolist()
    output_text = tokenizer.decode(output_ids)
    input_text = tokenizer.decode(input_ids[0])
    te = time.time()
    if use_lookahead:
        print(f'with lookahead:{te - ts:.3f}s')
    else:
        print(f'without lookahead:{te - ts:.3f}s')
    print(f'prompt:{prompt}')
    print(f'input text:{input_text}')
    print(f'output text:{output_text}')
