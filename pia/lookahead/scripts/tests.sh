cd ../tests &&
python test_lookahead_cache.py &&
python test_triton_rms_norm.py &&

cd ../examples &&
python glm_example.py &&
python glm_batch_example.py &&
python chatglm_example.py &&
python llama_example.py &&
python llama_batch_example.py &&
python opt_example.py && 
python baichuan_example.py && 
python bloom_example.py && 
python gptj_example.py && 
python gpt2_example.py && 
python qwen_example.py &&
python mistral_example.py &&
python mixtral_quant_example.py


