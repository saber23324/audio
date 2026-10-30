# Adaptive Grounded64 / Gate-16 execution

This is an incremental, opt-in experiment namespace. It never replaces AudioRouter's legacy path, and it never writes an old checkpoint or result. Actual run root:

`artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/`

The existing taskbook is `AGENTS_ADAPTIVE_GATE_GROUNDED64_TRAIN_EVAL.md`; the requested Gate16-named alias was absent. Main comparison is new Utility Gate Grounded64 versus the actually retrained Stage1 RGB64. Old Stage1 improvement is a separate contrast.

Execution order is `common.initialize -> data plan/acquire/finalize -> features -> train_vision_router -> development -> generate_utility -> train_gate -> select_policy -> development online controls/cost/tests -> locks.freeze -> confirmation eval_qa -> report -> finalize`. Training and Utility labels only accept explicit official-training-derived source roles. Confirmation loaders require the immutable pretest lock. The source snapshot, raw predictions and optimizer/RNG states remain in the run root.

The new Stage1 is explicitly warm-start adaptation from the old Stage1, three epochs and1,968 actual AdamW updates. It reuses the actual old phase4 first-answer-token CE, not an invented Full-Vision KD loss. Only query_tokens and visual_key train (933,888 active parameters). All large models, old Stage3 and Localizer stay frozen. Official native MUSIC open answers use the same Qwen1.5 chat template. Fixed chronological2s arrivals replace noncausal duration-based respacing.

F1 is `Zrgb + beta*(Zground-Zrgb)` through the same new query bank, exactly64/frame. `beta`, Gate thresholds and random-control probability are validation-only choices. There is no target ON percentage, full-video score sorting or per-video quota. OFF bypasses Localizer and the second image encoder. The r head is a frozen joint-event/synthetic-pairing proxy, not a visibility probability.

`eval_qa.py` processes each arriving frame/window before creating its branch's predicted Grounded Image. Histories append immutable CPU snapshots. Audio is cropped to both raw PCM boundaries before resampling. No GT box/map, audio token, predicted class/tag, QA type, answer or future state enters the Gate/LLM visual interface. The canonical native question is used only in downstream QA after observation; official gold is used only after greedy generation for scoring.

V0/V1/V2/A0/A1/F0/G0/G1/G2/G3/G4/G5/G6/G7 are actually executed; Random uses three seeds. R-only and Energy are extra controls. F0 and G7 are identical fixed-F1 methods but both are executed. G8/F2 is optional/not_run. H0 is historical native5,840-token context, never a matched-budget result. Same-mask duplicate, shifted-map and zero-waveform-map controls share G1's exact arrival decisions.

The Python environment requires `PYTHONNOUSERSITE=1`; otherwise a conflicting user torchvision installation fails import. Use `.venv-gate/bin/python`, offline frozen HF assets, local logs (`WANDB_MODE=disabled`) and GPUs4–7. The exact child argv/environment are recorded by `pipeline.py`; prior manual commands are recorded in `logs/manual_commands.json`.

MUSIC numeric IDs/media hashes are separated, but original YouTube lineage is unknown. Valor uses known YouTube IDs disjoint against the historical inspected ledger. Pretraining overlap is unknown. New inspected-source identities are appended only to this run's `all_seen_sources.jsonl`; old manifests and source registries are preserved.

