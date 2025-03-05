# model=/mntnlp/common_base_model/Meta-Llama-3-8B-Instruct
# model=/mntnlp/common_base_model/Qwen2-72B-Instruct/Qwen__Qwen2-72B-Instruct
# model=/mnt/nas_acr89/jingyue/moe-lite
model=/mnt/nas_sgk32/jingyue/Bailing-4.0-MoE-Plus_A29B-4K-Chat-20241120-DeepSeek-fp8-dummy

vllm serve $model --swap-space 16  --disable-log-requests --trust-remote-code --pipeline-parallel-size 1 --tensor-parallel-size 8 --gpu-memory-utilization 0.9 --enable-chunked-prefill --max-num-batched-tokens 2048 --port=8765
