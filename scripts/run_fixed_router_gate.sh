#!/usr/bin/env bash
set -euo pipefail
cd /home/yxd/audio_pic_gate
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/home/yxd/audio_pic_gate/LLaVA-NeXT:/home/yxd/audio_pic_gate/lmms-eval:/home/yxd/audio_pic_gate
export HF_HOME=/home/yxd/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 WANDB_MODE=disabled BEATS_EMBEDDING_CACHE=''
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Existing version is immutable; this script resumes only complete stage outputs.
# Initial data selection/acquisition commands are in the run's commands.json/logs.
.venv-gate/bin/python -m utility_gate.fixed_router_gate.pipeline "${1:?phase required}"
