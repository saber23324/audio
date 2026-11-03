# IMU visual next phase, 2026-10-30

This additive namespace follows `AGENTS_IMU_AS_IMAGE_CAUSAL_GATE_ROUTER_NEXT_PHASE.md`.
It never trains or writes a historical model. The actual run is archived in
`artifacts/imu_visual_next_phase_20261030/`; authentic input bytes are in
`/nvme_data/pkt/streaming/dataset/imu_visual_next_phase_20261030/`.

## What is implemented

- Actual loaded-module inventory, parameter-level shapes/trainability, historical
  optimizer/checkpoint provenance and fresh historical-file SHA verification.
- Bounded, independently checked HTTP ranges for official UTD-MHAD / HoloAssist
  archives, ZIP entry CRC checks, official WEAR code/data-format inspection.
- Typed strictly past-only physical sensor windows, integer .NET clock conversion,
  fixed calibration-source plot scales, axes/norm renderers without task labels.
- Frozen visual-only LLaVA runtime with no BEATs/EZVSL/audio object or sensor-token
  injection. Actual SigLIP, image, projector and V1 forward counters.
- Completed-recording UTD smoke and original-clock HoloAssist completed-event
  **derived QA**. HoloAssist media PTS is not assumed to be camera capture time:
  unique camera-pose ticks must exactly match video frame count. Count mismatches
  are excluded, not repaired by guessing an offset.
- Native real/duplicate/zero/wrong/past-shift controls, separately labeled numeric
  text alternative, grouped paired statistics and uncached resident cost profiling.

The renderer does not create spatial RGB heatmaps. HoloAssist measures head motion;
UTD measures right wrist or thigh; WEAR supplies four accelerometers, not gyroscopes.
The raw synchronized HoloAssist release is already sampled relative to RGB. The
adapter does not recover missing high-rate samples. Asynchronous acc/gyro are
combined by causal last-observation hold at their original-clock union, never by
interpolation with a later sample.

## Reproduction

Use `.venv-gate/bin/python`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`,
`PYTHONPATH=LLaVA-NeXT:lmms-eval:.`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=4`. The execution used GPUs 4–7.
Data acquisition additionally needs network access to the official public hosts.

```bash
python -m utility_gate.imu_visual.native_imu_probe --shard 0 --shards 4
python -m utility_gate.imu_visual.utd_appendix --shard 0 --shards 4
python -m utility_gate.imu_visual.holo_intake paired_probe --shard 0 --shards 4
python -m utility_gate.imu_visual.holo_rgb_clock
python -m utility_gate.imu_visual.holo_probe frames --shard 0 --shards 4
python -m utility_gate.imu_visual.holo_probe run --shard 0 --shards 4
python -m utility_gate.imu_visual.summarize
python -m utility_gate.imu_visual.profile_imu_cost
python -m utility_gate.imu_visual.integrity
```

Change the shard ID for each worker. Scored source outputs resume by skipping
existing files. Initialization, manifest building and cost collection refuse to
overwrite their archive. To perform a new experiment, first choose a new isolated
ROOT/DATA namespace and new untouched sources; do not change the archived locks.
The UTD post-primary controls are explicitly diagnostic, not preregistered evidence.

Router/Gate training is conditional on taskbook §6.3 and §11. No random or
historical audio Gate is substituted for an untrained IMU model. The final report
states the actual stage status; a `not_run` stage has no claimed checkpoint,
optimizer steps, Gate ON rate, selection result or independent confirmation.

Tests use the existing repository test environment, adding
`/tmp/audio_pic_gate_pilot_pytest` to `PYTHONPATH`. Full test output and exact
execution environment are archived under `logs/` and `M7_final/`.
