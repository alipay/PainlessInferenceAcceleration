python framework.py > flood.log &&
cd ../benchmark && sh vllm_serve.sh & sleep 300 && sh vllm_client.sh > vllm.log