# RGB+IMU scale-up continuation

Dedicated namespace; every old experiment is read-only. Taskbook: `AGENTS_RGB_IMU_DATA_SCALEUP_AND_GATE_ROUTER_TRAIN.md`.
Outputs: `artifacts/utility_gate/imu_visual_scaleup_train_v1`; raw/cached data: `/nvme_data/pkt/streaming/dataset/imu_visual_scaleup_train_v1`.

Official acquisition extends verified archive indexes and annotation prefix; complete objects only. Original ticks are retained and strict lower AND upper bounds checked for both accelerometer and gyroscope. Camera mapping requires exact unique-tick/frame count equality. Released synced streams have clock grade B, not a claim of certified native timing precision. Fresh holdout recordings cannot intersect the historical seen registry.

Main training warm-starts each matching v3 REAL/DUP/ZERO checkpoint. Only original 696,320 Fusion parameters update, forced ON, canonical per-frame B1. Fresh AdamW lr1e-5 wd.01 clip1,3 epochs, same seeded question order. Frozen VLM/V1; 4x64 tokens. Gate uses only typed current/past RGB and27 causal IMU statistics, trained after the selected REAL checkpoint's complete16-mask actual QA utility. No question/class/source strings enter Gate or plots. Accuracy-first ranks generated validation accuracy then NLL; cost-first uses1pp accuracy/+0.05NLL boundary and minimum extra images. No target ON quota. Constants are identified honestly.

Entrypoints under `.venv-gate/bin/python -m utility_gate.imu_visual.scaleup_train`:
- `acquire {video,camera,imu,annotations,pairs}`
- `build_data build`, `build_data frames --shard i --shards n`, `build_data finish`
- `driver all`: train/dev features+native QA → 3 actual matched warm-start Router fits → all16 true QA train/dev → four fresh tiny Gate fits/selection → validation QA → locked new-source confirmation → actual cost.
- `integrity`, `report` once all outputs exist.

Each failed attempt is retained. Training is never skipped because historical IMU-only or32QA accuracy was low. Additional scratch/three-scale fits require separate explicit entries; warm-start and sample-count comparisons alone are not a controlled scale-effect curve.


Completed actual run:94HoloAssist recording prefixes/721QA; REAL/DUP/ZERO each1500updates, two scale fits and two matched scratch fits,4Gates each1890updates,10656complete16-mask QA,9724paid pipelines. New confirmation55QA/4recording families:V1=16,REALalways=37,Gate=38,DUP/zero/wrong same-mask=38,Fixed-k2=39. Learned selection and uniqueIMU claimNO-GO; measured Gate cost reduced18–21percent versusREALalways. Two selected Gate aliases are identical. Full report: artifacts/utility_gate/imu_visual_scaleup_train_v1/FINAL_RESEARCH_JUDGMENT.md. All historical weights/results preserved;358tests passed.
