# cd ../tests &&
# python test_lookahead_cache.py &&
# python test_triton_rms_norm.py &&

cd ../examples &&
echo glm_example &&
python glm_example.py &&

echo glm_batch_example &&
python glm_batch_example.py &&

echo chatglm_example &&
python chatglm_example.py &&

echo chatglm3_example &&
python chatglm3_example.py &&

echo llama_example &&
python llama_example.py &&

echo llama_batch_example &&
python llama_batch_example.py &&

echo opt_example && 
python opt_example.py &&

echo opt_batch_example && 
python opt_batch_example.py &&

echo bloom_example && 
python bloom_example.py &&

echo gptj_example && 
python gptj_example.py &&

echo gpt2_example && 
python gpt2_example.py &&

echo qwen_example && 
python qwen_example.py &&


# 4.30.2
pip install transformers==4.30.2 &&

echo baichuan_7b_example && 
python baichuan_7b_example.py &&

echo baichuan_13b_example && 
python baichuan_13b_example.py &&

# 4.36.0
pip install transformers==4.36.0 &&

echo baichuan2_7b_example && 
python baichuan2_7b_example.py &&

echo baichuan2_13b_example && 
python baichuan2_13b_example.py &&

echo internlm_example &&
python internlm_example.py &&

echo mistral_example &&
python mistral_example.py &&

echo mixtral_quant_example &&
python mixtral_quant_example.py &&


echo finished!