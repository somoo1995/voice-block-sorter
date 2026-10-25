#!/usr/bin/env bash
# 모델 내려받기 + graph.pbtxt 생성 (서버는 띄우지 않음)
# 인터넷(Hugging Face)이 필요한 유일한 스크립트. 모델은 이미 models/ 에 있으니 시연 당일(와이파이 끔)엔 실행하지 않는다.
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$PWD/ovms/lib PATH=$PWD/ovms/bin:$PATH
R=$PWD/models
pull() { echo "=== $(date +%T) $*"; ovms --pull --model_repository_path "$R" "$@"; echo "=== rc=$? $(date +%T)"; }
pull --source_model OpenVINO/Qwen3-4B-int4-ov --task text_generation --target_device GPU --tool_parser hermes3 --reasoning_parser qwen3
pull --source_model OpenVINO/Qwen3-Embedding-0.6B-int8-ov --task embeddings --target_device CPU
pull --source_model OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov --task rerank --target_device CPU
pull --source_model OpenVINO/whisper-large-v3-turbo-int8-ov --task speech2text --target_device NPU
pull --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation --target_device GPU --tool_parser hermes3 --reasoning_parser qwen3
# 같은 모델을 GPU 에 올리는 graph (config.json 의 graph_path 로 고른다)
W=$R/OpenVINO/whisper-large-v3-turbo-int8-ov; RR=$R/OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov
sed 's/target_device: "NPU"/target_device: "GPU"/' "$W/graph.pbtxt" > "$W/graph_gpu.pbtxt"
sed 's/target_device: "CPU"/target_device: "GPU"/' "$RR/graph.pbtxt" > "$RR/graph_gpu.pbtxt"
# 4B 의 KV 캐시를 동적 → 1GB 고정(SETUP §7.7)
Q4=$R/OpenVINO/Qwen3-4B-int4-ov
sed 's/cache_size: 0,/cache_size: 1,/' "$Q4/graph.pbtxt" > "$Q4/graph_kv1.pbtxt"
echo ALLDONE
