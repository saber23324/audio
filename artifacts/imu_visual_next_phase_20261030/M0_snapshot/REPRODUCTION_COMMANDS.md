# Reproduction and inspected training sources

Current revision: `016e6b483768a5918cdec55c40de156014f9d3ff`. Use`.venv-gate/bin/python`, PYTHONNOUSERSITE=1, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=LLaVA-NeXT:lmms-eval:repo, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, OMP_NUM_THREADS=4, CUDA_VISIBLE_DEVICES=4.

New M0 command: `python -m utility_gate.imu_visual.snapshot --phase inventory`; old replay commands are not to be relaunched into old writable roots. Exact previous executed subprocess argv/environment, seeds, optimizer metadata and retries remain in `/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/commands.json` (JSONL), `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/commands.json`, and new commands.json.

Environment captured in previous execution (current model actual load also performed):
```json
{
  "python": "3.10.20 (main, Jun 11 2026, 15:17:37) [GCC 14.3.0]",
  "platform": "Linux-5.15.0-25-generic-x86_64-with-glibc2.35",
  "libraries": {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "transformers": "4.53.3",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "Pillow": "12.3.0",
    "av": "15.0.0",
    "flash-attn": "2.8.0.post2",
    "matplotlib": "3.10.9"
  },
  "torch_cuda_build": "12.4",
  "cudnn": 90100,
  "GPUs_and_driver": "0, NVIDIA H800, 580.65.06, 81559 MiB\n1, NVIDIA H800, 580.65.06, 81559 MiB\n2, NVIDIA H800, 580.65.06, 81559 MiB\n3, NVIDIA H800, 580.65.06, 81559 MiB\n4, NVIDIA H800, 580.65.06, 81559 MiB\n5, NVIDIA H800, 580.65.06, 81559 MiB\n6, NVIDIA H800, 580.65.06, 81559 MiB\n7, NVIDIA H800, 580.65.06, 81559 MiB\n",
  "used_GPUs": [
    4,
    5,
    6,
    7
  ],
  "formal_LLM_dtype": "float16",
  "Router_and_Gate_dtype": "float32",
  "BEATs_batch_size": 1,
  "gate_CPU_threads": 4,
  "timing_model_loading_included": false,
  "timing_Gold_scoring_included": false,
  "playback_wait_included": false,
  "feature_cache_for_formal_frontend": false,
  "filesystem_page_cache": "uncontrolled",
  "formal_policy_order": "per-source hash rotation of fixed37 variants",
  "note": "large model and datasets unchanged; environment metadata captured during frozen confirmation without tuning"
}

```

Sources and hashes:
- `streamingtom/train_adbt_videomme.py` SHA `36588057973a3d9244acb787874dd9d67b6d97cc9c855d5d7af5e8043bb21a1f`
- `utility_gate/adaptive_grounded64/runtime.py` SHA `32085399e1dff22f4716ded113581d49158083a785b619222054ff10ed0af7ab`
- `utility_gate/gate_first_native/train_fusion.py` SHA `fce1a1694c508e5c3c55a9c81806592ee5d05386dd09afb084683fb35c4581db`
- `utility_gate/fixed_router_gate/gate_train.py` SHA `c7e98bad239d35e171859afec947b12ba9719cb2663657633b0efe948d717707`
- `utility_gate/fixed_router_gate/gate_models.py` SHA `15e44074206f485e4f1f3b3787120144aefb362b7c40edf4ed967f2f375665cd`
- `utility_gate/fixed_router_gate/online_inference.py` SHA `5891e2d2555093ec1c24ea98cf3c40090c0d894ec629f5f9decaca46e943a6e0`
