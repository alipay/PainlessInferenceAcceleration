model=/mntnlp/common_base_model/Meta-Llama-3-8B-Instruct

vllm serve $model --swap-space 16  --disable-log-requests --trust-remote-code --pipeline-parallel-size 1 --tensor-parallel-size 8 --gpu-memory-utilization 0.9 --enable-chunked-prefill --max-num-batched-tokens 2048 --port=8765
