model=/mntnlp/common_base_model/Meta-Llama-3-8B-Instruct

python benchmark_serving.py --backend vllm --model $model --dataset-name random --num-prompts 20000 --request-rate 20.0 --random-input-len 112 --random-output-len 334 --random-range-ratio 0.8 --ignore-eos  --port=8765
# python benchmark_serving.py --backend vllm --model $model --dataset-name sharegpt --dataset-path /mntnlp/nanxiao/dataset/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 20000 --request-rate 20.0 --ignore-eos --port=8765



