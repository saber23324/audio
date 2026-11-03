# Direct transfer of the audited Audio architecture to real IMU plots

The final canonical run is
`artifacts/utility_gate/imu_gate_router_transfer_20261030_v3/`.
This package directly reuses `GroundedFusionRouter` and the old
`FixedRouterGate` network, adapting only the auxiliary feature width to27
cheap IMU statistics. The frozen visual VLM, projector and V1 do not train.
No BEATs or EZVSL is instantiated in the IMU runtime.

The original class consumes196+196 visual features, d3584/key128/rank32,
and returns64 tokens. OFF is exactly fixed V1. ON is RGB+actual causal plot
through the same frozen visual encoder and the trained residual. All four-frame
compressed strategies contain256 visual tokens; native references are separate.

## Executed stages and preserved versions

- v1 actually fitted3 Routers and4 Gates but failed the original-sample lower
  boundary audit: causal last-observation hold could retain a sample up to
  16.49ms before`t-2`. All artifacts remain preserved as nonconforming.
- v2 fixed that boundary and actually refitted all3 matched Routers and4 Gates.
  Every original acc/gyro sample now lies inside`[t-2,t]`.
- The v2 numerical audit found that batch4 and per-frame batch1 Router execution
  can change fp16 rounding and Gold NLL despite equal generated tokens. Finalv3
  retains this task's actually trained v2 Routers, revalidates every trained epoch
  with canonical B1 inference, freezes the selected weights, regenerates all16
  whole-QA labels and actually fits four new matching Gates. Router optimization
  is not falsely counted as repeated atv3.

All failed and superseded weights/results remain untouched. The final report
identifies the training origin, dataset limits, constant endpoint Gate policies,
per-mask controls and the absence of a new blind confirmation cohort.

## Reproduction

Use the existing`.venv-gate/bin/python` environment. Set
`PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1`,
`PYTHONPATH=LLaVA-NeXT:lmms-eval:.`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=4`. The run used GPUs4–7.

`IMU_TRANSFER_ROOT` and `IMU_TRANSFER_DATA` optionally choose new isolated
directories. Initialization and training refuse to overwrite existing checkpoints.
To replicate the frozen protocol, initialize a new ROOT/DATA, copy the final
`data/` and only its data/plot/protocol locks (preserving references and hashes
of the archived raw input assets), then execute:

```bash
python -m utility_gate.imu_gate_router_transfer.audit
python -m utility_gate.imu_gate_router_transfer.driver features
python -m utility_gate.imu_gate_router_transfer.driver routers
python -m utility_gate.imu_gate_router_transfer.driver utility
python -m utility_gate.imu_gate_router_transfer.driver gates
python -m utility_gate.imu_gate_router_transfer.driver evaluate
python -m utility_gate.imu_gate_router_transfer.driver cost
```

The finalv3 itself reused actual same-taskv2 native results and frozen visual
features, and ran`revalidate_router` for the three previously fitted Routers
before the Utility/Gate stages. Exact subprocess argv/GPU/PID/return codes are
in`commands.jsonl`; all raw optimizer, epoch, prediction and cost records remain
under`logs/`, `checkpoints/`, `results/` and`locks/`.

`supplement` executes every validation QA under fixed K1/2/3 random schedules
with three fixed seeds, all four fitted objectives at fixed threshold0.5 (paying
neural Gate cost), and full native RGB/IMU/duplicate cost references. This is
development diagnosis, not another selection round or independent confirmation.

Runtime input media comes from verified original compressed-video byte ranges;
decoding is paid. Plots/features are not cached in the timed pipeline. Network
data acquisition, model loading and extra gold scoring are explicitly excluded.
Raw input paths may point to preserved earlier acquisition folders; their bytes
are reused read-only. Do not remove those archived raw data dependencies.

No images or timestamps are invented. HoloAssist head IMU and completed official
action intervals supply derived QA; participant IDs are unavailable. Previous
UTD/WEAR clock limitations and all current network failures are documented.
Known inspected sources are train/dev only and cannot be called a fresh test.
