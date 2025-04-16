# model=/mntnlp/common_base_model/Meta-Llama-3-8B-Instruct
# model=/mntnlp/common_base_model/Qwen2-72B-Instruct/Qwen__Qwen2-72B-Instruct
# model=/mnt/prev_nas/nanxiao/chat_80b
# model=/mnt/nas_acr89/jingyue/moe-lite
# model=/mnt/nas_sgk32/jingyue/Bailing-4.0-MoE-Plus_A29B-4K-Chat-20241120-DeepSeek-fp8-dummy
model=/mnt/prev_nas/chatgpt/pretrained_models/Qwen2.5-72B-Instruct


python benchmark_serving.py --backend vllm --model $model --dataset-name random --num-prompts 20000 --request-rate 20.0 --random-input-len 112 --random-output-len 334 --random-range-ratio 0.8 --ignore-eos  --port=8765
# python benchmark_serving.py --backend vllm --model $model --dataset-name sharegpt --dataset-path /mntnlp/nanxiao/dataset/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 20000 --request-rate 20.0 --ignore-eos --port=8765



