#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")

# dont use reasoning model as we dont need it for our recipe chat bot.
# unsloth.ai uploaded GGUF dynamic quantized weights, but vllm doesn't recognize? must use llama.cpp?
# vllm serve unsloth/Ministral-3-14B-Instruct-2512-GGUF \
# vllm serve mistralai/Ministral-3-14B-Instruct-2512-GGUF \
# above 2 dont work with vllm serve, why?? need to do something special?
vllm serve mistralai/Ministral-3-14B-Instruct-2512 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.6 \
    --max-num-batched-tokens 16 \
    --host 0.0.0.0 \
    --port 8989 \
    --api-key test \
    2>&1 | tee serve_vllm_ministral_3_14b_instruct_2512_$now.log

# reduce max-model-len from 262144 to 50k since we don't need so much
# set gpu-memory-utilization to 0.6 to reduce memory usage to be polite on shared GPUs
# vllm serve mistralai/Ministral-3-14B-Reasoning-2512 \
#     --tensor-parallel-size 2 \
#     --tokenizer_mode mistral \
#     --config_format mistral \
#     --load_format mistral \
#     --enable-auto-tool-choice \
#     --tool-call-parser mistral \
#     --reasoning-parser mistral \
#     --max-model-len 50000 \
#     --gpu-memory-utilization 0.6 \
#     --host 0.0.0.0 \
#     --port 8989 \
#     --api-key test \
#     2>&1 | tee serve_vllm_mistral_3_14b_reasoning_2512_$now.log

# 192.168.3.184
# (APIServer pid=2411522) INFO 12-11 00:07:04 [api_server.py:1847] Starting vLLM API server 0 on http://0.0.0.0:8989
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:38] Available routes are:
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /openapi.json, Methods: GET, HEAD
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /docs, Methods: GET, HEAD
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: GET, HEAD
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /redoc, Methods: GET, HEAD
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /health, Methods: GET
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /load, Methods: GET
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /pause, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /resume, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /is_paused, Methods: GET
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /tokenize, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /detokenize, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/models, Methods: GET
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /version, Methods: GET
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/responses, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/messages, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/completions, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/audio/transcriptions, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/audio/translations, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /ping, Methods: GET
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /ping, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /invocations, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /metrics, Methods: GET
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /classify, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/embeddings, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /score, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/score, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /rerank, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v1/rerank, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /v2/rerank, Methods: POST
# (APIServer pid=2411522) INFO 12-11 00:07:04 [launcher.py:46] Route: /pooling, Methods: POST
# (APIServer pid=2411522) INFO:     Started server process [2411522]
# (APIServer pid=2411522) INFO:     Waiting for application startup.
# (APIServer pid=2411522) INFO:     Application startup complete.

# Dynamo bytecode transform time: 9.59 s
# (Worker_TP0 pid=2412213) INFO 12-11 00:06:43 [backends.py:257] Cache the graph for dynamic shape for later use
# (Worker_TP0 pid=2412213) INFO 12-11 00:06:50 [backends.py:288] Compiling a graph for dynamic shape takes 15.52 s
# (Worker_TP0 pid=2412213) INFO 12-11 00:06:52 [monitor.py:34] torch.compile takes 25.11 s in total
# (Worker_TP0 pid=2412213) INFO 12-11 00:06:53 [gpu_worker.py:359] Available KV cache memory: 52.36 GiB
# (EngineCore_DP0 pid=2412130) INFO 12-11 00:06:53 [kv_cache_utils.py:1286] GPU KV cache size: 686,320 tokens
# (EngineCore_DP0 pid=2412130) INFO 12-11 00:06:54 [kv_cache_utils.py:1291] Maximum concurrency for 100,000 tokens per request: 6.86x
# (Worker_TP1 pid=2412214) 2025-12-11 00:06:54,210 - INFO - autotuner.py:256 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
# (Worker_TP0 pid=2412213) 2025-12-11 00:06:54,211 - INFO - autotuner.py:256 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
# (Worker_TP1 pid=2412214) 2025-12-11 00:06:54,230 - INFO - autotuner.py:262 - flashinfer.jit: [Autotuner]: Autotuning process ends
# (Worker_TP0 pid=2412213) 2025-12-11 00:06:54,231 - INFO - autotuner.py:262 - flashinfer.jit: [Autotuner]: Autotuning process ends
# Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|█| 51/51 [00:02<00:
# Capturing CUDA graphs (decode, FULL):  96%|█████▊| 49/51 [00:01<00:00, 26.29it/s](Worker_TP1 pid=2412214) INFO 12-11 00:06:59 [custom_all_reduce.py:216] Registering 8160 cuda graph address
# es
# Capturing CUDA graphs (decode, FULL): 100%|██████| 51/51 [00:02<00:00, 25.33it/s]
# (Worker_TP0 pid=2412213) INFO 12-11 00:06:59 [custom_all_reduce.py:216] Registering 8160 cuda graph addresses
# (Worker_TP0 pid=2412213) INFO 12-11 00:06:59 [gpu_model_runner.py:4466] Graph capturing finished in 6 secs, took 0.08 GiB
# (EngineCore_DP0 pid=2412130) INFO 12-11 00:06:59 [core.py:254] init engine (profile, create kv cache, warmup model) took 36.48 seconds
# (EngineCore_DP0 pid=2412130) [2025-12-11 00:07:00] INFO tekken.py:187: Non special vocabulary size is 130072 with 1000 special tokens.
# (EngineCore_DP0 pid=2412130) [2025-12-11 00:07:01] INFO tekken.py:187: Non special vocabulary size is 130072 with 1000 special tokens.

# (APIServer pid=2411522) INFO:     192.168.3.113:41334 - "GET /v1/models HTTP/1.1" 200 OK
# (APIServer pid=2411522) INFO 12-11 00:12:16 [chat_utils.py:574] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
# (APIServer pid=2411522) INFO:     192.168.3.113:41334 - "POST /v1/chat/completions HTTP/1.1" 200 OK
# (APIServer pid=2411522) INFO 12-11 00:12:34 [loggers.py:236] Engine 000: Avg prompt throughput: 16.7 tokens/s, Avg generation throughput: 72.8 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
# (APIServer pid=2411522) INFO 12-11 00:12:44 [loggers.py:236] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 96.7 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.3%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
# (APIServer pid=2411522) INFO 12-11 00:12:54 [loggers.py:236] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 96.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.4%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
# ^[[B^[[B(APIServer pid=2411522) INFO 12-11 00:13:04 [loggers.py:236] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 95.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.6%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
# (APIServer pid=2411522) INFO 12-11 00:13:14 [loggers.py:236] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 95.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.7%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
# (APIServer pid=2411522) INFO 12-11 00:13:24 [loggers.py:236] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 95.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
# (APIServer pid=2411522) INFO 12-11 00:13:34 [loggers.py:236] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 94.7 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.0%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
# (APIServer pid=2411522) INFO 12-11 00:13:44 [loggers.py:236] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 94.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.1%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
# (APIServer pid=2411522) INFO 12-11 00:13:54 [loggers.py:236] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 93.8 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.2%, Prefix cache hit rate: 0.0%, MM cache hit rate: 0.0%
