# AGENTS — Next Stage: IMU-as-Image → Frozen Visual VLM → 64-Token Router → Causal Gate

> Status: **Codex execution specification / research protocol**, drafted 2026-10-30.  
> Scope: first **audit and archive the completed AudioRouter/Gate phase**, then initiate a **new, separately scoped IMU-as-Image phase**.  
> **No IMU experiments have been executed by the author of this task specification.** Every new performance claim must come from actual Codex execution and saved artifacts.

---

## 0. Mission and scientific boundaries

**Research goal:** Enable an existing **frozen visual-only VLM** to interpret synchronized wearable IMU measurements **through the same vision-encoder interface used for images**, then train a lightweight **visual token Router** and a **causal Compute Gate** to preserve useful video/IMU evidence with substantially lower computation than unconditional multi-image processing.

The immediate goals are:

1. **M0 — archive current AudioRouter research**. Inspect actual code, configuration, training logs, and checkpoint tensors; write a faithful and complete model inventory (parameter shapes/counts, exact branches, stagewise training, performance/limitations). Do not infer technical specifics from this document alone.
2. **Establish a demonstrable IMU-as-image bridge**: convert raw, synchronized, causal IMU windows into readable images (6-axis waveform plots for accelerometer + gyroscope when available; multi-wearable acceleration-only panels when that is what the dataset actually provides). Verify a frozen VLM can consume them and that they yield task-specific benefits **over the same RGB-only input**.
3. **Train a matched IMU-aware 64-token Router**, while retaining all original frozen large-model components. First validate unconditional RGB+IMU fusion and compare with repeated-RGB/sham inputs. Do not assume the IMU plot is useful before experiments.
4. **Only after the Router is fixed, train its matching Gate**: online choose whether an additional IMU plot is worth constructing/encoding at each already-observed timestamp. Search accuracy/cost trade-offs on a legal development split, not by setting a presumed ON rate.
5. Evaluate against an **uncompressed native RGB+IMU plot reference**, the **RGB-only 64-token** reference, matched sham/duplicate, matching random and no-gate baselines; report accuracy, conditional rescue/harm, real encoder and LLM costs, number of added plot images, generalization and real IMU specificity.

**Non-goals:** do not upgrade or fine-tune the frozen main VLM; do not inject raw IMU token embeddings, numerical IMU sequences, IMU-based answer labels, or modality-specific textual answer summaries directly into the LLM as part of the **proposed** path. Text/IMU-only baselines are permitted but must be explicitly identified as different-model controls. Do not introduce audio in this stage; isolate IMU.

**Strict claim discipline:** prior Gate results demonstrate *actual conditional compute savings* under some settings, **not proven accuracy non-inferiority or robust beneficial selection**; old R_A shows strong MUSIC in-domain adaptation but weak Valor transfer and no established audio-specific benefit. A similar result for IMU is not presumed; no SOTA claims without benchmark-matched external comparisons.

---

## 1. Primary Google paper: exactly what to reproduce, and what not to infer

**Google Research — Mayank Daswani et al. (2024), _Plots Unlock Time-Series Understanding in Multimodal Models_**

- Paper: https://arxiv.org/abs/2410.02637
- Full text: https://arxiv.org/html/2410.02637v2
- Google Research explanation: https://research.google/blog/unlocking-the-power-of-time-series-data-with-multimodal-models/

### 1.1 What this paper actually does

- Convert raw multi-dimensional time-series **numerical samples into conventional visual plots** and submit the plot to a **pretrained multimodal model's existing image encoder**.
- Compare **plot images** with **raw numerical values encoded as text**. Tested models include the 2024-era Gemini 1.5 and GPT-4o families; **not** the present frozen LLaVA-OneVision configuration.
- Evaluate IMU fall/near-fall/daily-living, IMU activity recognition, synthetic series-pattern tasks, and other time-series reasoning. In the fall setting, use three accelerometer axes and three gyroscope axes; the fall benchmark is a 15-second 128-Hz segment, **not** a ready-made aligned video+IMU QA benchmark.
- Demonstrate useful task performance **without new IMU-specific neural training**. This is a **zero-training image-encoding baseline**, not a paper that learned a Gate, trained an IMU Router or performed streaming causal video fusion.
- Their API token/cost savings compare **plot versus verbose numeric text**; they do **not** mean a second plot image is free or that an IMU+RGB two-image input is cheaper than RGB-only.

### 1.2 How to adapt the idea to our architecture

Represent the real, calibrated and synchronized IMU window as `Plot_t = Render(IMU_[max(0,t-W), t])`. Start with W=2 seconds to match the previous causal-control convention; on long-action tasks evaluate additional strictly past-only windows (e.g. W=4, W=8) **on the validation split** as a separate sensitivity analysis; do not silently change the context between methods.

Two honest visual paths:

- **Native-VLM upper reference (not compressed):** `[RGB_t, Plot_t] → native image processor → frozen VLM`. Check the actual native processing path, image expansion, order and token count; it may be extremely costly.
- **Proposed 64-token path:** `[RGB_t, Plot_t] → frozen image encoder/projector → trainable lightweight IMU-visual resampler → exactly 64 visual-compatible tokens → frozen VLM`. With Gate OFF: `RGB_t → existing V1 → exactly 64 tokens`. With Gate ON: fused RGB+IMU evidence → exactly 64 tokens. The LLM **never** receives extra IMU tokens outside these 64.

**Important conceptual distinction:** If values from an IMU plot directly contribute to compressed output tokens, they are image-encoded IMU information (visual-compatible tokens), **not literally original-RGB-only values**. Do not inaccurately reuse the old `Z=P(A,V)V` claim unless implementing the explicit optional control-only routing branch below.

### 1.3 Plot specification faithful to the principle

- Plot types: `P1=plain 6-axis waveform panels`, `P2=accelerometer/gyroscope norm+axes with fixed legend`; for WEAR: `P3=4 wrist/ankle acceleration traces (12 channels)` with truthful per-wearable labels. Additional spectrogram/GAF/recurrence plot may be optional validation studies, not defaults.
- Deterministic rendering: time on x-axis relative to timestamp t, physical units on y-axis (`m/s²` or `g`, `rad/s` or `deg/s` after documented unit conversion), canonical channel ordering, readable fonts, high contrast, labels limited to sensor identity and physical units. **Never print activity names, ground-truth actions, annotation IDs or QA answers** on the plot.
- Resampling: log raw sampling rate, clock sync, dropout/missingness and interpolation; apply **anti-aliasing** before downsampling; keep raw original samples; avoid `resample_poly` on uncalibrated irregular samples without verifying timing assumptions.
- Normalization: use **train-split-derived** sensor-specific fixed scaling; record magnitude saturation/clipping. Per-window min-max scaling can erase the very amplitude signatures required for falls and is not a valid unreported default. Run a controlled alternative if needed.
- IMU frame: distinguish sensor body coordinates from camera/world coordinates. A wearable on the wrist measures **wrist motion**, not automatically scene/object displacement. **Do not create spatial RGB heatmaps from IMU unless validated calibration and a task-specific geometric mapping exist.** The image is primarily a time-series chart, unlike acoustic sound-source boxes.
- For missing gyro (WEAR), plot acceleration only; do not fabricate gyro channels. For multiple IMUs, label the actual device position, orientation and measured channels.

---

## 2. Related work: what carries over and what is substantively different

| Reference | Relevant mechanism | What to borrow | What **not** to claim |
|---|---|---|---|
| **Google, Plots Unlock... (2024)** https://arxiv.org/abs/2410.02637 | Raw sensor sequence → plot image → existing VLM vision input | First sanity baseline; plotting and prompting | No trained Gate, no video synchronization proof, no 64-token streaming guarantee |
| **IMG2IMU** https://arxiv.org/abs/2209.00945 | IMU → visually interpretable spectrogram; transfer from image pretraining | Alternative plot type / sensor-aware augmentation | Not identical to direct frozen VLM QA |
| **IMU2IMG** https://doi.org/10.1145/3714394.3756208 | Inertial signals → RGB images for vision-foundation inference | Alternative sensor-to-image conversion | Different benchmark and trained pipeline; verify artifacts before adopting |
| **IMU2CLIP** https://aclanthology.org/2023.findings-emnlp.883/ | Align IMU recordings with video/text in CLIP space | Optional diagnostic sensor/video correspondence | This does **not** mean plotting raw IMU directly gives CLIP-equivalent embeddings |
| **ImageBind** https://github.com/facebookresearch/ImageBind | Align IMU to images in learned embedding space | Cross-modal diagnostic/teacher baseline | It is a different IMU encoder and modality-fusion mechanism |
| **AnyMAL** https://aclanthology.org/2024.emnlp-industry.98/ | IMU/sensors connected to an LLM via modality adapters | Strong alternative architecture for related work | Our main frozen VLM does not accept nonvisual sensor tokens |
| **IJCAI 2025 Vision for Time Series survey** https://ijcai.org/proceedings/2025/1178 | Line plot, spectrogram, GAF, recurrence plots | Controlled representations to compare | Survey taxonomy is not experimental support for our own benchmark |
| **WEAR (2024)** https://mariusbock.github.io/wear/ | Egocentric RGB + synchronized wearable accelerometers | Sensor complementarity, subject/session splits, action intervals | Not an official natural-language IMU QA benchmark |
| **HoloAssist (2023)** https://holoassist.github.io/ | Egocentric RGB + IMU + hand/action/mistake labels | Fine-grained temporal tasks and streaming examples | Do not present automatically formed QA as official QA |

Additional review: **UTD-MHAD** https://personal.utdallas.edu/~kehtar/UTD-MHAD.html, **Ego-Exo4D** https://docs.ego-exo4d-data.org/. Citation and dataset provenance must be included in generated reports.

---

## 3. Data selection and acquisition plan (prioritize actually synchronized video + IMU)

**Do not start multi-GPU training before verifying the exact downloaded raw IMU streams, actual RGB/video availability, and timeline synchronization.** A paper saying IMU is a modality does not guarantee every distributable split includes it.

| Priority | Dataset | Available relevant data | Recommended role | Caveats / access |
|---|---|---|---|---|
| **P0 smoke** | **UTD-MHAD** — https://personal.utdallas.edu/~kehtar/UTD-MHAD.html | 861 synchronized action sequences; RGB AVI, inertial MAT (accelerometer + gyroscope), 27 actions, 8 actors | End-to-end loader / signal plots / frozen VLM sanity / known-action labels | Small, can saturate; subject-disjoint evaluation; variable wrist/thigh location, 50 Hz; not suitable as sole claim |
| **P1 main controlled** | **WEAR** — https://mariusbock.github.io/wear/ | Head-mounted RGB video, four wearable **accelerometers** at 50 Hz, 22 people, 18 workout actions, action-localization intervals | Primary synchronized multi-sensor streaming experiment and sensor ablations | **No guaranteed gyroscope**; label conventions + participant-disjoint splits; CC BY-NC-SA 4.0; sizeable raw data |
| **P1 main complex** | **HoloAssist** — https://holoassist.github.io/ | 169 h reported site; RGB/depth/IMU/poses and fine-grained action, mistake annotations; official splits | Complex operation understanding, error recognition, temporal windows | Verify IMU type/axes per recording, RGB-synced files, task annotations and timestamp offset; big video data and unusual split rules |
| **P2 large-domain generalization** | **Ego-Exo4D v2** — https://docs.ego-exo4d-data.org/ | Project Aria RGB, **2 IMUs** (reported 800 and 1000 Hz), other sensors; synchronized procedural and sport actions | Large-scale cross-environment check, procedural segment QA | Requires license acceptance, VRS/Aria tools, multi-clock metadata, high storage; avoid exo/ego mix without calibration |
| **P2 optional** | **MMAct** — https://mmact19.github.io/2019/ | Multi-camera video + acceleration, gyroscope, orientation, action labels | Occlusion, cross-scene / cross-subject generalization | Data access by request; verify distributed modality files; older 2019 site claims 37 classes, 2021 challenge has different subset |

**Mandatory gating in acquisition (per dataset):**

- Record dataset version, publication/reference, license and access confirmation, downloaded byte count, file SHA256, modality-specific availability, action/QA labels, timezone/clock format, sample rates and official train/val/test protocol.
- Inspect >=10 full paired sessions across >=3 participants if available. Plot the IMU magnitude and video thumbnails at matched timestamps; verify sync offsets against physical movement, do not assume array index alignment.
- Split by **participant and/or raw recording/take** first; no overlapping windows, augmented views or different camera versions from one take across splits. Maintain both `session_id` and `participant_id`; grouped confidence intervals are hierarchical when feasible.
- Respect historical source ledger for any reused source; maintain a new IMU ledger. Cross-dataset duplicate risk and foundation model pretraining overlap remain documented uncertainties.
- If subject-disjoint and session-disjoint cannot simultaneously satisfy adequate labels, explicitly choose one primary split and the other as secondary stress test; **do not silently leak participants across a claimed subject-generalization split**.
- HoloAssist/WEAR action annotations may not provide **official QA**: construct deterministic, transparent prefix-resolvable multiple-choice questions only from legitimately timestamped labels, and label these as **derived QA**. Do not hallucinate human annotated QA or score predictions against unsupported pseudo-ground truth.

### 3.1 Recommended task hierarchy

1. **P0:** video-aligned action recognition from an observed 8-second prefix (only if the label is unambiguously valid for that prefix); use fixed canonical label vocabulary and accuracy/macro-F1.
2. **P1:** **temporal action change / action onset / step completion** from already-observed time intervals, using official interval labels. Evaluate event localization with tolerance and/or fixed choice QA where the answer is determined entirely by frames/sensors up to time t.
3. **P1:** fine-grained action and mistake recognition in HoloAssist using vetted official annotations and completed events. Avoid claims of causal advance mistake prediction if the ground-truth decision needs future outcomes.
4. **P1 challenge:** predefined IMU-relevant counterfactual/ambiguity strata where RGB alone is uncertain but motion signatures plausibly differ, with **predeclared non-model-based rules** and blinded human verification of a sample. Do not select test questions post hoc by VLM errors.
5. **P2:** same-domain unseen participants/sessions and cross-domain adaptation; label vocabulary differences require per-dataset evaluation (do not pool percentage accuracies as if tasks are identical).

For a quick sensor-only reproduction of the **Google plot-versus-text** result, a public fall-detection IMU-only dataset may be used **in a separate appendix experiment**. It does **not** validate RGB+IMU fusion unless paired video is truly supplied.

---

## 4. M0 — Mandatory completed-phase asset audit BEFORE changing the project

**Perform this phase first and produce all deliverables, then proceed automatically to data investigation.** The local codebase is the primary source of truth; the historical points below are *leads for verification*, not authoritative checkpoint contents.

### 4.1 Locate and inventory

Inspect actual repository roots / likely components (search rather than invent missing paths):

- `AGENTS.md`, all phase-specific AGENTS files; `utility_gate/runtime.py` and `utility_gate/grounded_dual_image/`, `utility_gate/adaptive_grounded64/`, `utility_gate/gate_first_native/`, and latest fixed-R_A Gate budget search implementation and artifacts.
- Stage1 original vision-only, newly trained V1, original Stage3 AudioRouter, frozen SigLIP, projector, LLM, BEATs, EZ-VSL, fixed F0, trained residual Fusion Router `R_A`, original correspondence head `r`, old/new Utility heads and finally selected Accuracy-first and Cost-aware Gate checkpoints.
- Training scripts, exact optimizer/LR/scheduler/epochs/batch size, data partitions and shuffling seeds, loss functions, checkpoint selection and model recovery logs. Include the fact that the old audited Stage1 phase4 used **first-answer-token CE** rather than automatically assuming Full-Vision KD.
- Actual `state_dict` parameter shapes, trainable/frozen flags, total/active parameter counts per component, strict checkpoint-load missing/unexpected keys, dtype/device, memory footprint, projector output dimension, visual tokens before/after Router, and native multi-image actual token expansion.
- Source ledgers, protocol and timestamp definitions, beam/greedy decode, system prompts, evaluation metrics, condition gates and profile.

The history reported approximately **933,888 active parameters** for new Stage1, **696,320** for `R_A`, **573,953** for a past u-only Gate, and additional small-head variants. **Codex must verify which checkpoint these numbers correspond to**; do not insert these approximate numbers as assertions about the final deployed Gate. Capture exact final values for Accuracy-first and Cost-aware Gate separately.

### 4.2 Snapshot outputs (required)

Write in an isolated new artifact folder, e.g. `artifacts/imu_visual_next_phase_20261030/M0_snapshot/`:

- `CURRENT_MODEL_SNAPSHOT.md`: reproducible architecture diagram, component shapes, only-trainable vs frozen, causal dataflow, Gate ON/OFF, exact 64-token interface, and descriptions of original AudioRouter versus current image-based R_A.
- `MODEL_PARAMETER_INVENTORY.csv` and `model_inventory.json`: paths, parameter name prefix, shape, count, frozen/trainable, checkpoint SHA256, precision, load results.
- `PHASE_HISTORY_AND_RESULTS.md`: Stage1/Stage3 → Grounded dual-image → newly trained V1 & Gate → Gate-first/Router-second → final fixed-R_A Gate search; show **dataset/QA count, prompt, sampling and source protocol for each reported number**, and avoid comparing different queues directly.
- `REPRODUCTION_COMMANDS.md`: exact runnable commands, environment/package versions, GPU, seeds, repo commit, model versions; if unavailable, report missing, do not guess.
- `MODEL_LIMITATIONS.md`: source provenance limitations, model-vs-dataset domain shift, extra-encoder cost even with 64 LLM tokens, real-vs-sham and non-inferiority findings, early gating cost.
- `snapshot_sha256.json`: guarantee all old checkpoints and existing reports are read-only/unchanged; log the new artifact directory independently.

**Suggested verified historical anchor to include with proper qualification:** previous 4-frame/frozen-LLM experiments always produced 256 visual tokens in the 64-token paths; native paired-image baselines could expand to 5,840 visual tokens. New R_A showed large MUSIC-domain gain **not specific to real acoustic heatmaps** and mixed/negative Valor transfer. Recent fixed-R_A Gate Cost mode reduced extra encoder calls with real cost savings, but noninferiority was unproven. **Do not say Gate has already preserved original accuracy on all benchmarks or reached SOTA.**

### 4.3 No mutation

- Make new code in `utility_gate/imu_visual/` (or an equally clearly isolated directory) and new artifacts in `artifacts/imu_visual_next_phase_20261030/`.
- Reuse old model code by **import/read-only references**, not by modifying original checkpoints. SHA checks before/after; additive tests only.
- If code cannot find the stated checkpoint, mark `not_found` and select **only a verified corresponding saved checkpoint**, never fabricate a path.

---

## 5. M1 — Data intake, time alignment, QA feasibility, and probe subset

Tasks:

1. Implement per-dataset adapters returning exactly:

```python
{
  'source_id': str, 'participant_id': str, 'session_id': str,
  'video_path': str, 'video_clock': ..., 'imu_path': str,
  'imu_channels': ['acc_x', 'acc_y', 'acc_z', 'gyro_x', ...],
  'imu_sensor_positions': [...], 'imu_rate_hz': float | None,
  'imu_unit_map': {...}, 'imu_times_sec': array,
  'task_id': str, 'answer': str | int,
  'question': str, 'answer_source': 'official_label|derived_verified',
  'label_start': float, 'label_end': float,
  'observed_prefix_end': float, 'available_modalities': {...}
}
```

2. For each video time `t`, RGB extraction uses only frames with timestamp <= t. IMU extraction uses only samples <= t (including interpolation whose bracketing samples are both <= t); no centered smoothing, bidirectional filtering across t, or normalization fitted to future/test samples.
3. Use previously established 4 decision instants at t=2,4,6,8 seconds for a **directly comparable 8-second-prefix protocol**, if duration and labels support it; report excluded duration sources before any prediction. HoloAssist/WEAR may additionally use rolling causal windows as a separate longer-stream protocol.
4. Perform timestamp alignment diagnostics, unit validation, finite-value assertions, stable canonical order, missing-channel policy, participant leakage test, per-modality hash. Create 6–12 manually inspectable examples **only from development sources**.
5. Maintain `data_manifest.jsonl`, `source_split_ledger.jsonl`, `imu_sync_audit.csv`, `IMU_DATA_FEASIBILITY.md`, `qa_definition_v1.md`, and `qc_exclusions.jsonl`.

**No-go for a dataset:** no accessible synchronized raw RGB and IMU, or no legitimate prefix-verifiable label, or repeated invalid timestamp alignment. Do not replace raw sensor with synthetic patterns and call it real evaluation.

---

## 6. M2 — Reproduce Google-style IMU plots as images; frozen-VLM sanity test (NO TRAINING)

### 6.1 Image renderers

Implement deterministic `render_imu_plot(sensor_window, config) -> RGB PIL.Image`, version all configs and plotted attributes:

- `P1`: separate accelerometer and gyroscope subplot rows (or truthful per-sensor acceleration rows when no gyro), actual time axis and fixed train-derived y-scale, channel legend.
- `P2`: axes plus resultant acceleration/gyro magnitudes; plot labels remain physical units and sensor IDs.
- `P3`: controlled spectrogram/other images **only after** P1 success, with equivalent information/axes; no visual words describing the correct class.

Make synchronized thumbnails and rendered plots into an inspectable development-only HTML gallery. Freeze file naming and resize rules before test use. Document color, DPI, font fallback, anti-aliasing and any signal pre-processing.

### 6.2 Frozen inference comparisons

On a legal *development-only* subset (e.g. 100–300 independent recordings if available), with identical questions/answer candidates and deterministic decoding:

| ID | VLM evidence | Purpose |
|---|---|---|
| I0 | RGB video only (native and/or V1 64) | visual baseline |
| I1 | IMU plot only (native image input) | can VLM read the chart? |
| I2 | RGB + same-window real IMU plot (native, high-token reference) | is combining available modalities useful? |
| I3 | RGB + RGB repeated image, same native shape | control for more image tokens |
| I4 | RGB + plot of zero/flat IMU with identical axes | control for generic chart formatting |
| I5 | RGB + same-structure IMU from a different source | false correspondence control |
| I6 | RGB + past-shifted IMU (causal, fixed shift) | timing relevance control |
| I7 | numeric IMU text (separate alternative baseline) | Google plot-versus-text comparison; **not** part of proposed final architecture |

Check the LLaVA-OneVision native image-processing pathway, processor crop/resolution, actual visual token count, image order, prompt behavior, and whether chart labels remain perceptible. This VLM may be less effective at chart reading than Gemini/GPT4o; treat it as a hypothesis.

**System prompt template (identical native variants except explicit image relation):**

> You are answering a question about the video observed up to the current time. The first visual observation shows the camera scene. When a second image is present, it is a time-series chart of an IMU worn by the camera wearer or at the sensor location indicated in the chart. Its horizontal axis is time; acceleration and angular velocity are physical measurements, not activity labels. Use the chart only for movement evidence it actually supports. Do not assume it describes motion of every visible object. Answer using the observed prefix only and use the required answer format.

Use parallel Chinese description for report, but keep exact runtime prompt saved. For I1 (IMU-only) do not falsely claim a camera image is present. Need a consistent neutral prompt comparison as well; do not tune based on confirmation.

### 6.3 Critical go/no-go for subsequent trained fusion

Advance to full M3 if: **(a)** plots are demonstrably readable by this frozen visual backbone on an IMU-relevant task (above label-prior/rand/zero controls, or inspectable nontrivial per-example effects), **and (b)** raw RGB+IMU synchronous data and legitimate labels exist. If I1/I2 are weak, perform only the predeclared small P1/P2 prompt/plot variants on development sources; do not immediately scale up Gate search. Report `IMU_BRIDGE_INCONCLUSIVE` if data too limited. M0/M1/M2 outputs still mandatory.

---

## 7. M3 — Train the IMU-aware **64-token** visual Router before Gate

### 7.1 Lock the starting path

- Freeze LLaVA-OneVision (SigLIP, projector, LLM), and original Stage1/Stage3/R_A/old Gates.
- Start V1 vision-only Router `R_V` from a verified old/new Stage1 checkpoint. **Keep it fixed for matched controls initially**, avoiding yet another untracked Stage1 retrain. If V1 adaptation is necessary, run a separately labeled matched training budget on both RGB and IMU branches.
- New IMU Fusion Router `R_IMU(phi)` receives frozen vision-encoder features of RGB_t and Plot_t; outputs exactly 64 visual-compatible tokens per observed decision time.
- Initialize residual fusion at zero (identity-equivalent to `R_V`) so QA cannot initially be harmed by arbitrary feature offsets. Verify exact output identity at step 0.

Suggested first implementation, with exact matching tensor shapes audited from M0:

```text
RGB_t -------------------------> frozen SigLIP/Projector ---> R_V -> Z_v(64,d)
IMU_[t-W,t] -> rendered plot ---> frozen SigLIP/Projector ---> R_V -> Z_i(64,d)
                                                             │
                        Z_fused(64,d) = Z_v + F_phi(Z_v,Z_i) <--- trainable residual adapter
                                                             │
                                                       frozen visual-only VLM
```

`R_V(Plot_t)` is a baseline convenience, not evidence it already understands a signal chart. Implement an optional dedicated trainable plot adapter **only if** the base attempt fails and with a pre-registered capacity-matched RGB duplicate control.

Alternative (optional, more faithful to original AudioRouter's RGB-only-values claim): IMU plot tokens influence only query/routing weights, while **compressed values are exclusively RGB features**: `Z = P(V, E_plot) V`. Document this as a separate configuration; do not conflate it with direct RGB+plot fusion.

### 7.2 Router loss/training

- Train with **Gate forced ON on training sources** (it is not yet trained); the Router must see sufficient true IMU plots.
- Start from the *actual audited* prior Stage1/fusion training pipeline, preserving optimizer/loss choices unless a change is explicitly named. The prior Stage1 phase4 used first-answer-token CE; a new full-answer CE or KD objective is a **new, labeled experiment**, not claimed to be previous behavior.
- Primary supervision: task-specific teacher-forced answer loss on **correct observed-prefix labels** (for action classes, canonical choice CE and/or full-answer Gold NLL). Optional full-native RGB+plot teacher KD may use legal train/val sources only, fixed teacher model and matching prompt protocol. Clearly separate teacher self-annotation from ground truth.
- Handle RGB-only initialization, no-improvement fallback, signal dropouts/heterogeneous devices, input-size normalization, overfitting control and EMA/early stopping as documented validations.
- Train at least **three explicitly comparable Router variants** if budget permits:
  - `R_REAL`: trained on RGB+true synchronized IMU plots.
  - `R_DUP`: independently trained with RGB+same-RGB second view; identical architecture/steps/data/optimizer/seed scheme.
  - `R_SHAM`: independently trained on RGB+sham/zero or valid cross-source IMU chart, same architecture/steps. Or run as controlled paired zero-shot substitutions if compute prevents full retraining; clearly separate two claims.
- Compare within-domain unseen participants/sessions and cross-domain as available. Strict source-level validation and hidden held-out confirmation.

### 7.3 Router checks

After training, run **Always-on** evaluation on legal validation sources, showing `R_REAL`, `R_DUP`, `R_SHAM`, fixed `R_V`, and native RGB+plot reference. Report per-class macro-F1, accuracy and detailed movement-relevant confusion, Gold NLL, feature residual norms, 64-token exactness, and freezing/HASH.

**A Router that only improves as much with duplicate RGB as with real IMU is a general extra-image adapter, not demonstrated IMU fusion.** This does not prevent studying compute allocation but limits claims accordingly.

Do **not** train Gate against a Router that will subsequently be modified; this directly avoids the earlier F0 → R_A utility-target mismatch.

---

## 8. M4 — Freeze `R_IMU`, then train an online, cost-aware Gate

### 8.1 What Gate decides

At each observed time t, before rendering the extra IMU plot or running a second SigLIP/Projector pass:

```text
observe RGB frame and causal raw IMU statistics at time <=t
   ├─ OFF: no plot generated; RGB -> frozen R_V -> 64 tokens
   └─ ON : render actual IMU_[t-W,t] plot; RGB+plot -> frozen R_IMU -> 64 tokens
append exactly 64 visual-compatible tokens into immutable streaming visual history
```

Gate may use only **current/past RGB features that are already needed for the video baseline**, plus inexpensive causal IMU statistics (e.g. six-axis mean/variance, norm, jerk, spectral energy computed only from available past samples, sensor quality/missingness). No high-cost separate sensor encoder or chart image generation **before** the gate by default. This is how the Gate can save **encoder FLOPs**, not only LLM token cost.

A small 1D causal IMU encoder is optional if it is cheaper than plot rendering + second vision encoding; measure its actual expense. The Gate must not read `question`, gold answer, task label, future timestamps, source ID or any model prediction generated using future inputs.

### 8.2 Correct utility supervision

The new Gate needs labels for **this exact frozen R_IMU**, not old F0 or audio R_A.

- For 4 observed times, enumerate all `2^4 = 16` complete choices **only on legal train/dev sources** (if computation feasible) and run actual full-sequence answer decoding/NLL. Do not sum isolated-window values as if utility were additive.
- Compute `U(S) = Loss(RGB_only) - Loss(R_IMU with mask S)` on legal training labels, and a compute-aware target `J(S) = task_gain(S) - lambda * measured_incremental_cost(S)`; the utility may contain interactions, not simply per-window deltas.
- For an online policy, only supervise decision t with features available at t; sequence-level label construction may use *offline train labels* and full choices **as training reward** but deployed decisions cannot look at future. Clearly separate an offline, noncausal `Oracle` bound from the achievable causal Gate.
- Compare masked supervised learning, pairwise ranking / listwise trajectory objectives, and a very small causal sequence policy. **Do not repeat 40-model brute-force search by default**; select a short preregistered capacity/signal ablation set after M3.
- Choose loss and checkpoint using validation only; report learned ON rates, calibration and induced cost. Any Gate that defaults to all-OFF or all-ON is valid output but not proof of active selection.

### 8.3 Gate controls and budget selection

Predeclare:

| ID | Selection mechanism | Added plot images per four frames |
|---|---|---|
| G0 | Vision-only Always OFF | 0 |
| G1 | IMU Router Always ON | 4 |
| G2 | learned Gate (`RGB history + cheap IMU`) | model chosen, 0–4 |
| G3 | cheap IMU-only Gate | model chosen, 0–4 |
| G4 | RGB-only Gate, matched capacity | model chosen, 0–4 |
| G5 | matching random ON, >=3 fixed seeds | match **pretest validation** ON rate, never test rate |
| G6 | same selection mask, replace true IMU plot with duplicated RGB | identical selection, extra image count, Router and prompt |
| G7 | same selection mask, replace true IMU with zero/sham/cross-source IMU plot | control for true sensor evidence |
| G8 | on-device rule/energy threshold (predeclared) | independent cheap heuristic |

Report **accuracy-first** and **compute-aware** Pareto frontier as two separately locked policies. Study online maximum additional images `{0,1,2,3,4}` with a *causal budget counter*, and report actual `mean k`, distribution k, actual plot render count, actual second SigLIP calls and peak memory. **Do not use full-video retrospective Top-K** as a streaming method.

Outputs always 4×64=256 visual tokens for a 4-frame prefix regardless of ON count; additional compute is in the image/sensor front end. Compare Gate overhead explicitly to savings against Always-on and against pure Vision-only.

---

## 9. M5 — Frozen confirmation, IMU specificity and non-inferiority

**A single unseen confirmation set must be locked before final score inspection.** If source size is small, prioritize independent subjects/sessions and predeclare minimum source counts. Do not keep reusing formerly viewed confirmations for new tuning.

Primary comparisons on same samples, fixed prompts/labels/frames:

1. **IMU bridge:** `Native(RGB+REAL IMU plot) - Native(RGB+RGB)` and `Native(RGB+REAL) - Native(RGB+SHAM)`.
2. **Router usefulness:** `R_REAL(always) - V1`, `R_REAL(always) - R_DUP(always)`, `R_REAL(always) - R_REAL(sham/zero)`.
3. **Gate selection:** `Gate(R_REAL) - V1`, `Gate(R_REAL) - R_REAL(always)`, `Gate(R_REAL) - RandomRateMatched(R_REAL)`.
4. **Sensor specificity:** `Gate same mask REAL - same mask DUP/SHAM/zero/cross-source`.
5. **Compute:** end-to-end *resident-model*, uncached-feature measurement; number of rendered IMU charts, extra SigLIP calls, VLM-prefill/decode, total serial compute and peak memory, with and without gate-cost payment.

Metrics:

- Accuracy, macro-F1, balanced accuracy, optionally temporal localization F1/mAP, per-task NLL (do not pool heterogeneous NLL definitions), exact correct/total, participant/source-cluster bootstrap CI, paired rescue/harm, stratified results for IMU-necessary vs purely visual labels, missing sensor and occlusion classes.
- Primary 95% CI computed on **source/participant clusters** (e.g. 2,000 bootstrap resamples, fixed seed). For sparse strata, explicitly report insufficient data. Zero observed flips with a degenerate bootstrap `[0,0]` is **not** an equivalence proof.
- A predeclared non-inferiority margin relative to Always-on (e.g. **1 percentage point**) must be justified by a power check and validated with the correct one-sided lower CI. Do not call a 0–4 plot saving “quality-preserving” merely because point accuracy differences are small.
- Keep on-device online latency/observed prefix compute distinct from answer-arrival TTFT; do not claim realtime speedups from 3-source shared-host profiling.

Final study judgment explicitly split into:

- `GO-IMU-BRIDGE`: untrained frozen VLM can read useful actual IMU plots on aligned tasks.
- `GO-IMU-ROUTER`: compressed 64-token `R_REAL` improves useful task accuracy/fidelity over baseline and appropriately matched training controls.
- `GO-GATE-SELECT`: trained Gate has demonstrably better **same-budget** choices than locked random/heuristic controls.
- `GO-GATE-EFFICIENCY`: actual front-end runtime/compute saved relative to Always-on under predeclared task-quality constraints.
- `GO-IMU-SPECIFIC`: REAL IMU beats synchronized false/zero or duplicate evidence, so it is legitimate to attribute benefit to IMU content.
- `GO-GENERALIZATION`: unseen participants/sessions and at least one additional task/domain (when label-compatible) preserve the claim.

Each receives `GO / NO-GO / INCONCLUSIVE`, with exact evidence and failure reasons. Do not merge them into a single marketing GO.

---

## 10. M6 — Interpretation guardrails and known failure modes

### A. Modality and directionality

An inertial sensor measures **the attached object's acceleration and angular rate**. In an egocentric headset it may primarily reflect camera wearer head motion; in WEAR it measures wrists and ankles. IMU-as-image is *not* automatically a spatial heatmap aligned to the RGB image. No image-grounding/ROI claims without a separate validated geometrical construction.

### B. Dataset and metric integrity

- Generic action classification may be solvable from RGB alone or from IMU alone. An IMU-specific claim requires REAL-vs-SHAM, corresponding participant/scene split controls and task-specific improvements; “the model got a high score” is insufficient.
- Derived QA tasks must have genuine prefix-resolvable answers; otherwise report official action/mistake metrics, **not** “IMU-VQA official leaderboard”.
- Do not combine Google-paper fall data (without synchronized RGB) with HoloAssist/WEAR as though they share the same labels/condition.
- No inference from future frames, normalization using all-clip future statistics, zero-phase offline filtering, future-sampled time windows, or future-resolved action label used before it becomes observable.
- If official task evaluation is full-video, label this **8-second causal prefix protocol** separately and do not cite its score as official leaderboard SOTA.

### C. Specific errors seen in the audio-to-image phase to avoid

- Previously real grounded images often matched duplicated/sham images; therefore every IMU experiment must include content-specific controls.
- Previous Gate trained on F0 did not transfer cleanly after Router changed to R_A; therefore **Router is frozen BEFORE generating Gate labels and training Gate**.
- Previous source/QA numerical independence did not necessarily establish original media provenance; new IMU datasets must track participant/session/take IDs and leakage via augmented clips.
- LLM token compression and front-end computational savings are **different quantities**; avoid claiming full-system efficiency from 64 output tokens alone.
- Do not mistake domain-specific learned QA adaptation for general-purpose IMU understanding.

### D. Engineering fail-safe

If a dataset is inaccessible, continue with other verified datasets; log exact error and request/access requirements, do not invent file contents. If frozen VLM plot-reading fails, report what failed and move to the predeclared small renderer test; do not fabricate a positive signal to justify Gate training. If GPU run is interrupted, save step/checkpoint/version/seed, deterministic replay as far as feasible, and audit restored state.

---

## 11. Implementation and experiment stages — mandatory milestones

| Milestone | Task | Required tangible artifact | Exit condition |
|---|---|---|---|
| **M0** | Existing stage model/parameter/config snapshot | `CURRENT_MODEL_SNAPSHOT.md`, JSON/CSV inventory, SHA | Actual checkpoints and model paths audited |
| **M1** | Acquire and sync paired video+IMU, build valid causal labels | dataset QC, manifests, split ledger, sample gallery | Real aligned RGB+IMU+target verified |
| **M2** | Google-style plots and frozen native VLM baselines | plot renderer, per-task score, duplicate/zero controls, token audit | Plausibility and limits known |
| **M3** | Train dedicated 64-token IMU Router | R_REAL/R_DUP checkpoints, curves, 64-token identity tests | Router frozen/locked before Gate labels |
| **M4** | Complete-mask intervention and Gate-only training | 16-mask train/dev utility, G2–G8 policies, budget curves | Pretest policy locked on dev |
| **M5** | Truly unseen source confirmations | per-QA/subject predictions, CIs, rescue/harm, sham controls | No test-driven tuning |
| **M6** | Cost profiling and failure diagnosis | real GPU timing, extra plot count, compute decomposition | Cost and QA separated |
| **M7** | Cross-session/cross-domain validation and conclusion | final report with six separate GO decisions | Qualified claims, hashes, provenance |

*The stages are sequential for the selected main dataset.* Once M0–M2 has a verified usable dataset, execute M3–M7 without asking for a new instruction, subject to actual data availability and compute. If a mandatory upstream feasibility criterion fails, complete all available diagnostic artifacts and label downstream `not_run` rather than fabricate results.

### Suggested additive repository layout (adjust to actual audited project)

```text
utility_gate/imu_visual/
  data_adapters/
  causal_imu_window.py
  render_imu_plot.py
  imu_visual_prompts.py
  native_imu_probe.py
  router_imu64.py
  train_router_imu.py
  complete_mask_utility.py
  gate_imu.py
  train_gate_imu.py
  eval_imu_qa.py
  profile_imu_cost.py
  README.md

tests/
  test_imu_causality.py
  test_imu_sync_and_units.py
  test_imu_plot_no_label_leak.py
  test_imu_router64_identity.py
  test_imu_gate_conditional_encoder.py
  test_imu_native_vs_compressed.py
  test_imu_split_disjointness.py

artifacts/imu_visual_next_phase_20261030/
  M0_snapshot/
  M1_data/
  M2_frozen_imu_plots/
  M3_router_train/
  M4_gate_train/
  M5_frozen_confirmation/
  M6_pipeline_cost/
  M7_final/
  models/  # only NEW checkpoints
  commands.json
  train_val_lock.json
  all_seen_sources.jsonl
  final_audit.json
```

### Mandatory tests before final confirmation

- `t−W ≤ sensor_sample_time ≤ t` for every injected IMU sample; no future framing or bidirectional filters; exact input shape and data hash equality on repeated runs.
- ON/OFF path identity: OFF only encodes original RGB, plot and extra SigLIP calls **exactly zero**; compare with V1 unmodified fixture. ON plot must only use same-window raw IMU; equal 64 visual tokens per t, no text sensor sequence or feature shortcut appended to LLM.
- Confirm **actual forward-hook** counts for `R_V`, `R_IMU`, SigLIP/Projector, plot renderer and Gate. Gate itself paid cost accounted for.
- Synthetic-IMU test: impulse timing, axis swap and polarity, amplitude normalization, timestamp offset, sample-rate mismatch and missing data flagging; rendered plot must not contain derived class text.
- Data disjointness by subject/session/take (not just names) and metadata independence; paired real/sham share all non-IMU inputs.
- Old checkpoint SHA invariance; new trainable parameter counts; test that Router and Gate are NEVER updated simultaneously; evaluate only fixed frozen checkpoints on confirmation.
- Testing target label provenance and canonical candidate list, no NLL aggregation across incompatible scoring protocols.

---

## 12. Reporting template and acceptance

Create `artifacts/imu_visual_next_phase_20261030/M7_final/FINAL_IMU_STAGE_REPORT.md` with sections:

1. **Snapshot** of actual previous model: component-level parameters, exact checkpoints, architecture, train/frozen map, training and validation history, limitations.
2. **Google paper mapping**: what was reused (plot image), what was changed (RGB+IMU time alignment, 64-token compression, online gate), and which paper results were NOT reproduced.
3. **Dataset provenance**: official source, format, sensor placements/axes, clock sync, participant/session splits, labels, data exclusions and official vs derived QA.
4. **Table A**: frozen native plot sanity, RGB-only, duplicate/sham/cross-source, macro-F1/accuracy and actual token counts.
5. **Table B**: trained R_REAL vs R_DUP/R_SHAM/V1/native on development and untouched confirmation.
6. **Table C**: Gate selections and 0–4 extra plot budget frontier, same-budget random/heuristic, selected Gate ON timing, rescue/harm, coupled windows.
7. **Compute**: plots rendered, SigLIP forward counts, LLM 64 visual tokens, total source/participant-level runtime and memory; show baseline and relative numbers.
8. **6-way GO / NO-GO / INCONCLUSIVE**, failure case audit, implications for later radar/event-camera sensors, but no premature generalization claims.
9. **Reproducibility and integrity**: command logs, environmental information, source & checkpoint SHA, pretest lock, full per-sample predictions, figures and limitation notes.

**If the new model succeeds only on a development split or small action dataset, the result is in-domain feasibility, not SOTA.** Strong claims require matching published protocols, independent source confirmation, and content-specific control success.

---

## 13. Execute-now instruction for Codex

> **Read the repository's existing `AGENTS.md` and this file. Do NOT start by redesigning the models. First inspect every existing Stage1/Stage3/V1/R_A/Gate checkpoint and produce the detailed M0 model-parameter and architecture snapshot, including hashes and real training settings.**
>
> Next study Google Research's _Plots Unlock Time-Series Understanding in Multimodal Models_. Use authentic synchronized video and IMU data, prioritizing UTD-MHAD as a short smoke test, WEAR for first complete controlled experiment, and HoloAssist for harder actions. Reproduce causal raw-IMU→time-series-plot→frozen-vision-VLM inference and measure whether RGB+real IMU adds useful information beyond RGB+duplicate/zero/wrong IMU.
>
> Once verified, adapt the existing light fusion Router to output exactly 64 visual-compatible tokens per time. **Train Router first with real IMU and matched duplicate controls. Freeze Router and train a new, matching causal Gate from its real whole-sequence QA utility, never from historical audio F0 or R_A labels.** The Gate must decide BEFORE any extra IMU chart or vision encoding occurs; no fixed ON percentage. Compare mean extra chart counts 0–4, real on-device compute, subject/source-level QA and matched random/sham.
>
> Do not modify old weights or claim false training results. Preserve all errors, negative results and uncertainty. If data access or a mandatory feasibility test fails, produce the corresponding diagnostic report and mark unexecuted later steps honestly. Write the full reproducible `FINAL_IMU_STAGE_REPORT.md`.

