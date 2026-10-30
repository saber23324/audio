# Frozen grounded dual-image QA

This is a separate executable experiment over the existing AudioRouter. No legacy
model, checkpoint, evaluation entry point or default flag is changed. All model
parameters stay frozen. The experiment lives in
`artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/`.

The source file `AGENTS_AUDIO_GROUNDED_DUAL_IMAGE_QA.md` defines the task. The final
report is `GO_NO_GO.md` in the artifact directory. Read `STATUS.json`, the pretest
lock, per-stage reports, and `final_audit.json` for actual completion and limits.

MUSIC uses authoritative `data/json_update` annotations at official commit
`cc8809514a2ee3d7ae81a3021c15e7c0679170d7`, verified both against Git blobs and the
pinned public media mirror. Its published QA splits share video IDs. This run
reserves 500 test real-video IDs and then selects 200 train video IDs excluding
the reserved set. Synthetic videos are excluded. MUSIC's original YouTube
mapping is unavailable; IDs/bytes/path disjointness does not establish complete
physical-source independence. Valor is a separate fixed 128-source, 256-QA
supplement with known YouTube ID exclusions. Scores are never pooled.

Four question-independent observation times use duration fractions
0.125/0.375/0.625/0.875. Each moment has native RGB/derived image slots, with
explicit same-time role labels. Native multi-image preparation really inserts
729 content + 1 newline tokens per image: 8 images/4 moments = 5840. The existing
video path pools to 196 per frame; Stage1/Stage3/D11 deliver 64 per frame, 256 in
total. D11 independently uses the same frozen Stage1 query bank for both images
and mixes corresponding output slots. It changes visual values and is not
claimed to be clean-value resampling or identical to native multi-image fusion.

The official frozen EZ-VSL VGGSound-144k checkpoint is loaded strictly. Only its
AVL component generates maps; no OGL model, GT box, class name, question or answer
is a localizer input. Native-rate PCM is cropped at BOTH boundaries
`[max(0,t-2),t]` before mono conversion/resampling. It is resampled to 16k with
zero boundary state, then left-padded by 1 second for the model's native 3-second
input. That distribution shift is explicit. Constant/nonfinite maps or invalid
audio revert to RGB. Maps are relative scores, not match/visibility probabilities.

Initial developer runs with the inherited prefix-resampler are retained under
their original filenames and `source_snapshot/development_prefix_resampling/`.
The strict version uses `*_rawcrop` developer roles and
`maps/predicted_rawcrop_v1/`. It repeats the pilot before selecting alpha=0.20
and mixture=0.25. Only strict-version outputs enter the final confirmation.

MUSIC reports native answer-label generated short-answer normalized EM, plus
stricter EM without numeric aliases. Normalization is frozen before baseline
inference: case/whitespace/underscores/terminal punctuation and whole-answer
numerals 0..11. Canonical-answer sequence Gold NLL is a custom full-vocabulary
teacher-forced sum over answer tokens, excluding EOS; it is not official
42-class classifier NLL. Valor uses strict generated uppercase ABCD and
four-candidate normalized first-token NLL; full-vocabulary NLL is separate.
The raw `gold_nll_per_token` details originate from canonical full-vocabulary
teacher scoring. The definitive dataset-specific metric is `gold_nll` with its
explicit `nll_protocol`; do not mix the two datasets' NLL scales.

Commands are in `commands.json` and `commands_executed.jsonl`. Main CLIs:

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/yxd/audio_pic_gate/LLaVA-NeXT:/home/yxd/audio_pic_gate/lmms-eval:/home/yxd/audio_pic_gate
export HF_HOME=/home/yxd/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
# Set CUDA_VISIBLE_DEVICES to one of the audited free GPUs for model jobs.
.venv-gate/bin/python -m utility_gate.grounded_dual_image.common
.venv-gate/bin/python -m utility_gate.grounded_dual_image.data metadata
.venv-gate/bin/python -m utility_gate.grounded_dual_image.data acquire --role music_dev_pilot
.venv-gate/bin/python -m utility_gate.grounded_dual_image.baseline --role music_dev_pilot --regress
.venv-gate/bin/python -m utility_gate.grounded_dual_image.experiment --role music_dev_pilot_rawcrop --candidates --fixture
.venv-gate/bin/python -m utility_gate.grounded_dual_image.lock assets
.venv-gate/bin/python -m utility_gate.grounded_dual_image.lock freeze
.venv-gate/bin/python -m utility_gate.grounded_dual_image.experiment --role music_confirm --shard 0 --shards 4 --baselines
.venv-gate/bin/python -m utility_gate.grounded_dual_image.report
```

These are a reading guide, not a request to overwrite this completed run. Existing
role manifests and pretest locks refuse replacement. Use a new run ID and new
uninspected sources for further tuning. Technical resumes keep completed QA rows;
they do not replace wrong answers. Confirmation runs verify source, manifest,
configuration and checkpoint hashes before inference. The source snapshots retain
the actual locked implementation.

Direct pipeline timing uses 16 fixed development sources twice, 224 real greedy
calls across seven strategies, including fresh frame/audio decoding, localizer,
extra visual encoding, resampling and LLM. Gold teacher scoring is excluded;
weights and OS disk caches are warm, feature/PCM caches are cleared. These are
shared-host measurements with per-process PyTorch peak allocations.

No GT heatmap enters formal QA. The fixed old VGG-SS development diagnostic uses
bboxes only after predictions for scoring; neither it nor the older category
proxy is a new QA result. AVQA-Hard's index/official label join and media-context
audit are saved; a changed-media-context score is not called an exact Hard
replication. Optional crop/heatmap-only/Oracle QA/Gate2 branches are not run.
