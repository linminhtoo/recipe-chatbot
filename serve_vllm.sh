#!/bin/bash

# reduce max-model-len from 262144 to 100k since we don't need so much
vllm serve mistralai/Ministral-3-14B-Reasoning-2512 \
  --tensor-parallel-size 2 \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --reasoning-parser mistral \
  --max-model-len 100000 \
  --host 0.0.0.0 \
  --port 8989 \
  --api-key test

