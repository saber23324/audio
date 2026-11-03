# Actual phase history and results

Scores on different cohorts / prompts / timestamp grids are not directly comparable.

## artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/locks/stage1_retrained_lock.json

```json
{
  "created_utc": "2026-10-29T07:23:58.483376+00:00",
  "selected_epoch": 3,
  "selected_checkpoint": "/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/models/stage1_vision64_retrained/selected.pt",
  "checkpoint_sha256": "eb26d508875171cf6e18d6d86eabe34082dec97a72875a90e38e5581ae91b0d4",
  "mode": "warm_start_legacy_stage1_on_fresh_official_music_train",
  "warm_start_checkpoint_sha256": "47be825e896292f0efb855e1069e1277d159fd128ce06deaae7bdbe5e811f48a",
  "optimizer_steps": 1968,
  "changed_parameter_names": [
    "query_generator.query_tokens",
    "visual_key.weight"
  ],
  "frozen_backbone_state_sha_before": "0bae2d6aed55515c5eee5f3bd5d8dedaa08e7d9b18a994956c23e29a73321771",
  "frozen_backbone_state_sha_after": "0bae2d6aed55515c5eee5f3bd5d8dedaa08e7d9b18a994956c23e29a73321771",
  "frozen_backbone_gradients": 0,
  "fit_manifest_sha256": "3803a53c17ac72418cfc4108c152949f6a3f978a041ce6a0faea9b14939b5f75",
  "validation_manifest_sha256": "0688c6abfe0fa2c5797f164fbdfc1fceea8f8a4e40a774640dba3153eb87bffa",
  "selection_rule": "lowest held-out stage1_val first-answer-token CE among actually trained epochs1,2,3; tie earlier epoch; epoch0 reference not selectable",
  "elapsed_s": 927.4482201050268,
  "complete_history_sha256": "b195b9540ed5a42396f9bad568c1bc19f4857d580a27aa356126e41ad322a5eb"
}

```

## artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/config/protocol.json

```json
{
  "seed": 1234,
  "task": "Gate-16 as authorized actual adaptive taskbook, M0-M7",
  "model_training": {
    "stage1": {
      "mode": "warm_start_legacy_stage1_on_fresh_official_music_train",
      "warm_start_checkpoint": "/home/yxd/audio_pic_gate/results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt",
      "epochs": 3,
      "learning_rate": 1e-05,
      "weight_decay": 0.01,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-08,
      "max_grad_norm": 1.0,
      "gradient_accumulation_steps": 1,
      "batch_size_qa": 1,
      "objective": "exact old phase4 full-vocabulary cross entropy of FIRST answer token only; no Full Vision KD in audited legacy Stage1",
      "gradient_checkpointing": true,
      "bottleneck_stage": 1,
      "num_queries": 64,
      "hidden": 256,
      "heads": 8,
      "temperature": 0.05,
      "value_mode": "native",
      "latent_norm": "none",
      "only_active_parameters": [
        "query_generator.query_tokens",
        "visual_key.weight"
      ],
      "selection": "lowest held-out stage1_val first-answer-token CE among actually trained epochs1,2,3; tie earlier epoch; epoch0 reference not selectable",
      "protocol_adaptations": [
        "official MUSIC native open answer in same qwen_1_5 chat template; no fabricated multiple-choice alternatives",
        "fixed chronological2s first32 arrivals replace old noncausal full-duration uniform frame selection",
        "feature cache is frozen RGB only; no real/null audio in Stage1 fitting",
        "fresh optimizer from old final weight checkpoint; warm-start adaptation, not from-scratch reproduction",
        "local logs only, no W&B telemetry"
      ]
    },
    "utility_gate": {
      "epochs": 30,
      "learning_rate": 0.001,
      "weight_decay": 0.01,
      "batch_size": 64,
      "hidden": 64,
      "dropout": 0.1,
      "loss": "source-weighted Huber on newly measured signed add-one-grounded sequence-NLL utility; target scale from train only",
      "checkpoint_selection": "held-out gate_val MAE",
      "models": [
        "new_utility_only",
        "new_utility_on_frozen_combined_corr_trunk"
      ]
    }
  },
  "frame_policy": {
    "first_timestamp_s": 2.0,
    "period_s": 2.0,
    "max_frames": 32,
    "question_independent": true,
    "duration_not_used_to_respace": true
  },
  "scoring": {
    "MUSIC": "official updated native labels; normalized generated answer EM with prior frozen case/whitespace/underscore/terminal-punctuation/numeral0..11 normalization; canonical-string full-vocabulary sum NLL withoutEOS",
    "Valor": "original legacy multiple-choice prompt with official4options; strict generated uppercaseABCD;4candidate-normalized first-logit NLL, full-vocab separate"
  },
  "fuser": {
    "kind": "F1 same new Stage1 slots, fixed beta selected on legal validation",
    "beta_candidates": [
      0.25,
      0.5
    ],
    "overlay_alpha": 0.2,
    "highlight_area": 0.2,
    "color_rgb": [
      255,
      48,
      48
    ]
  },
  "utility_targets": {
    "scope": "actual add-one frame intervention with all other frames RGB new Stage1*",
    "primary": "baseline complete-answer sequence NLL minus add-one sequence NLL",
    "correctness_delta": "actual greedy EM difference",
    "aggregation": "per-source/time median over native questions, no global-video label replication",
    "uncertainty_band": 0.001
  },
  "policy_selection": {
    "u_threshold_candidates": [
      "-inf",
      -0.02,
      0.0,
      0.02,
      "inf"
    ],
    "r_threshold_candidates": [
      "-inf",
      "old_corr",
      0.0,
      "inf"
    ],
    "objective": "maximize real complete-mixture validation EM, then minimize complete-mixture Gold NLL, Visual retention tolerance2pp, final tie larger threshold; no activation-rate objective",
    "random": "three Bernoulli seeds [1234,2234,3234], probability locked to actual main Gate validation ON rate; no confirmation-rate fitting",
    "no_fixed_activation_rate": true
  },
  "development_manifest_sha256": {
    "stage1_train": "3803a53c17ac72418cfc4108c152949f6a3f978a041ce6a0faea9b14939b5f75",
    "stage1_val": "0688c6abfe0fa2c5797f164fbdfc1fceea8f8a4e40a774640dba3153eb87bffa",
    "gate_train": "45bc2717d2c8b755326db86ef810e176baee623ba9920497f0e5727fde156d47",
    "gate_val": "fac7c9263e27b37ff4a85015dcd77bc4673e111cb169588128d668369df877f6"
  },
  "confirmation_sealed_manifests": {
    "confirmation_music": {
      "path": "/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/manifests/confirmation_music.jsonl",
      "sha256": "52b5c0c4434b677486f02826d78ce470c3e452da3dd36e69e2f96424fe98b8b9",
      "sources": 300,
      "qa": 439
    },
    "confirmation_valor": {
      "path": "/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/manifests/confirmation_valor.jsonl",
      "sha256": "931578eb194fec16b8751f224b459645cf5d8f213e83da30efb36c519f8c4dcf",
      "sources": 256,
      "qa": 512
    }
  },
  "source_scope": "Music IDs/bytes/paths disjoint, original YouTube lineage unknown; Valor knownYouTube disjoint against historical ledger, Music physical cross-lineage/pretraining not provable",
  "all_old_models_frozen": true,
  "large_model_optimizers": 0,
  "train_test_alias_check": true
}

```

## artifacts/utility_gate/gate_first_router_second_native_ablation_v1/locks/fusion_router_lock.json

```json
{
  "created_utc": "2026-10-29T12:47:51.878978+00:00",
  "selected_epoch": 3,
  "selected_checkpoint": "/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/models/fusion_router/selected.pt",
  "selected_sha256": "924ec02aed505f7592e51e3259ae5b8822fbcec1ba5c905205ee5242fd66fe2a",
  "Gate_A_lock_sha256_before_after": "2b2a8c81dadcdb1c436b1070c223c57754e40cffa798a331607a08c72617cbe7",
  "Gate_weights_SHA_unchanged": true,
  "backbone_state_sha_before": "0bae2d6aed55515c5eee5f3bd5d8dedaa08e7d9b18a994956c23e29a73321771",
  "backbone_state_sha_after": "0bae2d6aed55515c5eee5f3bd5d8dedaa08e7d9b18a994956c23e29a73321771",
  "V1_sha256_before_after": "eb26d508875171cf6e18d6d86eabe34082dec97a72875a90e38e5581ae91b0d4",
  "optimizer_steps": 1968,
  "changed_parameters": [
    "query",
    "key.weight",
    "value.weight",
    "output.weight"
  ],
  "parameters_trainable": 696320,
  "gradient_update_window_counts": {
    "gate_selected_on": 2277,
    "forced_on_exposure": 5553
  },
  "training_time_exposure_is_not_deployment_Gate_ON": true,
  "elapsed_s": 169.1807931679068,
  "resume_from": "artifacts/utility_gate/gate_first_router_second_native_ablation_v1/models/fusion_router/epoch2.pt",
  "zero_gradient_samples_recorded": 14,
  "selection": "legal val B2 always-on real EM then Gold NLL; trained epochs only",
  "Gate_u_not_relabelled_for_new_Router": "G_A remains trained for fixed F0; deployment with RF is frozen transfer"
}

```

## artifacts/utility_gate/gate_first_router_second_native_ablation_v1/config/protocol.json

```json
{
  "seed": 1234,
  "phase_order": [
    "M0",
    "A train Gate on fixed V1/F0",
    "freeze Gate",
    "B train new Router",
    "C native bypass",
    "new confirmation",
    "statistics"
  ],
  "timestamps_s": [
    2.0,
    4.0,
    6.0,
    8.0
  ],
  "audio_window_s": 2.0,
  "visual_tokens_per_frame": 64,
  "fixed_F0_beta": 0.25,
  "overlay_alpha": 0.2,
  "overlay_area": 0.2,
  "Gate_training": {
    "epochs": 30,
    "lr": 0.001,
    "weight_decay": 0.01,
    "hidden": 64,
    "dropout": 0.1,
    "batch_size_sources": 16,
    "seed": 1234,
    "variants": [
      "u_huber",
      "u_rank",
      "dual_rank",
      "vision_rank",
      "u_classification_if_complementary"
    ],
    "policies": "all16 masks, actual full-sequence generation+Gold NLL for every native QA",
    "aggregate": "median native-QA policy NLL benefit per source; conditional marginal averages over enumerated future choices; no future input in Gate",
    "tie_band": 0.001,
    "rank_tie_band": 0.001,
    "checkpoint_candidates": [
      1,
      10,
      20,
      30,
      "best_heldout_loss"
    ],
    "thresholds": [
      "-inf",
      -0.02,
      0.0,
      0.02,
      "inf"
    ],
    "class_thresholds": [
      "-inf",
      0.25,
      0.5,
      0.75,
      "inf"
    ],
    "r_thresholds": [
      "-inf",
      0.0,
      "old_corr",
      "inf"
    ],
    "selection": "real complete-policy val EM, then lower Gold NLL, Visual no worse than2pp, then larger thresholds; never activation rate",
    "minimum_sources_each_class": 20
  },
  "Router_training": {
    "kind": "R-A zero-output low-rank cross-view64 residual",
    "key_dim": 128,
    "value_rank": 32,
    "epochs": 3,
    "lr": 1e-05,
    "weight_decay": 0.01,
    "clip": 1.0,
    "seed": 1234,
    "loss": "verified legacy phase4 FIRST-answer-token CE only; no KD",
    "batch_size_qa": 1,
    "gradient_checkpointing": true,
    "exposure": "train Gate-selected ON plus explicit forced_on_exposure for other available windows; pure Gate trace retained",
    "selection": "best real Always-on validation EM then Gold NLL among trained epochs; frozen G_A path separately evaluated"
  },
  "random_seeds": [
    1234,
    2234,
    3234
  ],
  "random_probability": "actual selected Gate legal val ON rate, frozen before confirmation",
  "native": "fixed2 images/time, original then secondary, frozen identical sensor-uncertainty system prompt; actual tokens metered; C2 zero-shot transfer from fixed-F0 Utility",
  "optional_not_run": [
    "R-B joint resampler",
    "Gate_B refit",
    "native-specific Gate C8",
    "second-first image order"
  ],
  "scoring": "MUSIC prior normalized short-answer EM/canonical sequence NLL; Valor strict ABCD/four-candidate normalized NLL. No pooling NLL across datasets.",
  "development_manifest_sha256": {
    "train": "f6d82d82ab73049a1c5a03855ab9f150abac858258034cbbc60a37ba55246d87",
    "val": "46013f1a1e3549e5da84001a1cbe4fab80d204d9d66c8b7764d1865215fe0cb2"
  },
  "sealed_confirmation": {
    "confirmation_music": {
      "path": "/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/manifests/confirmation_music.jsonl",
      "sha256": "3ff83b1e1f014704894dabcb806771ab49810d779877090360a16c9015736750",
      "sources": 200,
      "qa": 288
    },
    "confirmation_valor": {
      "path": "/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/manifests/confirmation_valor.jsonl",
      "sha256": "2b8db049d77f4bf658849c202e42475690f3343cad9d06c4b748aea827eeac37",
      "sources": 254,
      "qa": 508
    }
  },
  "bootstrap": {
    "replicates": 2000,
    "seed": 1234,
    "unit": "Valor original known YT; MUSIC numeric ID only, original lineage unknown"
  },
  "decision": "stage-wise GO requires paired primary known-YT AV lower95% accuracy delta>0 against appropriate component and same-mask/duplicate/Sham/random controls, supported NLL and Visual drop<=2pp; all-off is no Gate active-use evidence; router Always-on diagnosis mandatory regardless; otherwise INCONCLUSIVE unless both components have no reproducible increment then current-config NO-GO"
}

```

## artifacts/utility_gate/fixed_router_gate_v1/data/protocol.json

```json
{
  "seed": 1234,
  "timestamps_s": [
    2.0,
    4.0,
    6.0,
    8.0
  ],
  "audio_window_s": 2.0,
  "tokens_per_frame": 64,
  "only_train_new_Gate": true,
  "Router_optimizer_count": 0,
  "train": {
    "families": [
      "audio_r",
      "audio_no_r",
      "visual",
      "acoustic",
      "history"
    ],
    "lambdas": [
      0.0,
      0.1,
      0.5,
      1.0
    ],
    "temperatures": [
      0.25,
      1.0
    ],
    "epochs": 30,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "batch_size_sources": 16,
    "hidden": 64,
    "dropout": 0.1,
    "seed": 1234,
    "target": "source-mean full canonical QA Gold NLL over all16 true masks, q=softmax(-(NLL+lambda*k)/temperature); invalid masks excluded",
    "objective": "sequence-level autoregressive all-mask cross entropy, no sum of window truth or independent marginal labels",
    "checkpoint_candidates": [
      1,
      10,
      20,
      30,
      "best_validation_distillation_loss"
    ],
    "thresholds": [
      0.0,
      0.25,
      0.5,
      0.75,
      1.0
    ]
  },
  "selection": {
    "accuracy_first": "lambda0 candidates: maximize actual val full-mask EM, then lower Gold NLL, then smaller measured frontend cost; off/on included",
    "cost_aware": "within1pp absolute EM and+0.05 dataset-native Gold NLL of val Accuracy-first, minimize measured predicted frontend cost, then mean k, then maximize EM; no activation target",
    "accuracy_margin": 0.01,
    "NLL_margin": 0.05,
    "visual_harm_diagnostic_margin": 0.02,
    "cost_calibration": "first3 lexicographic all4-audio-valid fresh validation sources, causally firstK prefix and per-family prepaid encoder/gate cost",
    "decision_latency_median_or_mean": "mean across3 fixed sources; descriptive, model loaded, no feature cache, no Gold evaluation or playback wait",
    "random_seeds": [
      1234,
      2234,
      3234
    ],
    "random_probability": "separately actual T3/T4 val decision ON rate, no test fitting",
    "online_caps": [
      0,
      1,
      2,
      3,
      4
    ],
    "cap_rule": "counter of already encoded extra images; once exhausted skip further auxiliary processing and return V1; no future ranking"
  },
  "formal_native_reference": "N0 not_run_reference_only; user requires all formal paths64/frame",
  "map_failure": "raw invalid => forced OFF; ON with map invalid still encodes rendered RGB fallback once through R_A, counts as extra image; map validity reported",
  "scoring": "MUSIC normalized native open-answer EM and canonical-string sum NLL; Valor strict ABCD and4-candidate normalized Gold NLL. No mixed-dataset NLL",
  "development_manifest_sha256": {
    "train": "e0e83743f87757a1885cecc8a846c728243ae0abed68083a6e6385038ab6c4a4",
    "val": "145b065ac0fb22c0d560d89ce4011308c2695d851e6ee2f217995e6e4d269c0c"
  },
  "sealed_confirmation": {
    "confirmation_music": {
      "path": "/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/data/confirmation_music.jsonl",
      "sha256": "2a7a452a0c754464fdbdb17a70425b97629c977792df1b1b6953f42bb2248805",
      "sources": 200,
      "qa": 282
    },
    "confirmation_valor": {
      "path": "/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/data/confirmation_valor.jsonl",
      "sha256": "f1ac6dafdc4ac341c6f51eb51a14190cf5a19d17551c3f70b07b9f9952a8b3c5",
      "sources": 252,
      "qa": 504
    }
  },
  "bootstrap": {
    "replicates": 2000,
    "seed": 1234,
    "unit": "known Valor YouTube ID; MUSIC numeric ID only, original lineage unknown"
  },
  "judgment": {
    "GO_SELECT": "Accuracy-first actual nontrivial masks, lower95 paired Accuracy>0 and supported Gold NLL against predeclared best fixed endpoint and val-rate random mean on known-YT AV primary; not explained by always-off/on",
    "GO_EFFICIENCY": "Cost-aware vsR_A always within1pp Accuracy and+0.05 Gold NLL (source bootstrap CI), extra encoder count reduced and actual frontend+pipeline cost reduced beyond prepaid costs",
    "GO_AUDIO_SPECIFIC": "same exact mask real Grounded vsdup/Sham/zero each lower95 paired Accuracy>0 with supported Gold NLL; else at most generic second-path adaptation",
    "otherwise": "NO-GO current configuration if precise no increment/only duplicate benefit; INCONCLUSIVE for broad intervals or source/power limits; eachGO flag reported separately"
  },
  "source_scope": "8s observed-prefix QA, not official full-video score; MUSIC original lineage unknown; new known-YT Valor disjoint against known inspected identities only"
}

```

## artifacts/utility_gate/fixed_router_gate_v1/locks/selection_lock.json

```json
{
  "data_protocol_sha256": "1e0747990197def94b7623758e5bb06fefce572f0f3f76286e6a6eb4e6e5ba2e",
  "fit_scope": "fresh val only; all16 actual full-QA; source-level Gates; no confirmation predictions exist",
  "candidate_count": 977,
  "distinct_selected_val_masks": 282,
  "selected": {
    "accuracy_first": {
      "id": "history_l0_t0.25_e1_p0.5",
      "spec": {
        "kind": "learned",
        "family": "history",
        "checkpoint": "/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/models/history_l0_t0.25/epoch1.pt",
        "checkpoint_sha256": "3ae923809ecf7d0f995043902ecaa971c8b8b0b4af035de7f0763971a192f2fa",
        "threshold": 0.5,
        "lambda": 0.0,
        "temperature": 0.25,
        "epoch": 1
      },
      "lambda": 0.0,
      "metrics": {
        "qa": 335,
        "sources": 96,
        "correct": 225,
        "accuracy": 0.6716417910447762,
        "gold_nll": 2.0761107897845865,
        "mean_k": 3.0729166666666665,
        "ON_rate": 0.7682291666666666,
        "k_distribution": {
          "0": 0,
          "1": 1,
          "2": 30,
          "3": 26,
          "4": 39
        },
        "ON_per_time": [
          0.9791666666666666,
          0.8645833333333334,
          0.6041666666666666,
          0.625
        ],
        "estimated_frontend_s_from_pretest_real_cost": 0.6270000788261112,
        "Rescue_vs_V1": 47,
        "Harm_vs_V1": 17,
        "evaluation": "causal chosen mask retrieves that exact actual complete VLM generation from legal all16; selected policies rerun online independently"
      },
      "val_codes": [
        15,
        13,
        15,
        3,
        11,
        15,
        5,
        7,
        3,
        7,
        15,
        3,
        15,
        15,
        3,
        15,
        15,
        7,
        15,
        3,
        11,
        3,
        5,
        15,
        7,
        15,
        15,
        6,
        11,
        3,
        9,
        11,
        9,
        7,
        15,
        15,
        3,
        15,
        3,
        15,
        15,
        9,
        11,
        7,
        15,
        3,
        15,
        3,
        15,
        15,
        15,
        15,
        11,
        3,
        15,
        3,
        3,
        15,
        11,
        9,
        13,
        14,
        7,
        7,
        3,
        7,
        11,
        15,
        7,
        11,
        3,
        15,
        3,
        9,
        7,
        15,
        11,
        15,
        15,
        15,
        3,
        15,
        3,
        9,
        15,
        15,
        7,
        15,
        15,
        1,
        15,
        15,
        9,
        13,
        3,
        15
      ]
    },
    "cost_aware": {
      "id": "acoustic_l1_t0.25_e20_p0.25",
      "spec": {
        "kind": "learned",
        "family": "acoustic",
        "checkpoint": "/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/models/acoustic_l1_t0.25/epoch20.pt",
        "checkpoint_sha256": "f5de889e1ad2655b554339f0f6054037743fe6896850699b54fdbf3d9337bd9c",
        "threshold": 0.25,
        "lambda": 1.0,
        "temperature": 0.25,
        "epoch": 20
      },
      "lambda": 1.0,
      "metrics": {
        "qa": 335,
        "sources": 96,
        "correct": 222,
        "accuracy": 0.6626865671641791,
        "gold_nll": 1.82505164328378,
        "mean_k": 1.1770833333333333,
        "ON_rate": 0.2942708333333333,
        "k_distribution": {
          "0": 0,
          "1": 79,
          "2": 17,
          "3": 0,
          "4": 0
        },
        "ON_per_time": [
          0.3020833333333333,
          0.6979166666666666,
          0.0625,
          0.11458333333333333
        ],
        "estimated_frontend_s_from_pretest_real_cost": 0.506240589001714,
        "Rescue_vs_V1": 38,
        "Harm_vs_V1": 11,
        "evaluation": "causal chosen mask retrieves that exact actual complete VLM generation from legal all16; selected policies rerun online independently"
      },
      "val_codes": [
        6,
        2,
        1,
        10,
        1,
        6,
        2,
        1,
        1,
        10,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        10,
        2,
        2,
        10,
        2,
        1,
        2,
        2,
        1,
        10,
        2,
        6,
        1,
        2,
        2,
        1,
        2,
        10,
        1,
        2,
        1,
        2,
        1,
        1,
        2,
        2,
        10,
        2,
        1,
        1,
        2,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        2,
        1,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        6,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        2,
        10,
        1,
        2,
        1,
        6,
        10,
        1,
        10,
        1,
        2,
        10,
        2,
        1,
        1,
        1,
        2,
        5,
        1
      ]
    },
    "audio_r": {
      "id": "audio_r_l0_t1_e30_p0.5",
      "spec": {
        "kind": "learned",
        "family": "audio_r",
        "checkpoint": "/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/models/audio_r_l0_t1/epoch30.pt",
        "checkpoint_sha256": "25adefe4ef3523ddd83e8c943199acceadcb307412c9dd6b178fbe133d82dd38",
        "threshold": 0.5,
        "lambda": 0.0,
        "temperature": 1.0,
        "epoch": 30
      },
      "lambda": 0.0,
      "metrics": {
        "qa": 335,
        "sources": 96,
        "correct": 222,
        "accuracy": 0.6626865671641791,
        "gold_nll": 1.9983921545704337,
        "mean_k": 2.0833333333333335,
        "ON_rate": 0.5208333333333334,
        "k_distribution": {
          "0": 0,
          "1": 15,
          "2": 59,
          "3": 21,
          "4": 1
        },
        "ON_per_time": [
          0.84375,
          0.3125,
          0.5520833333333334,
          0.375
        ],
        "estimated_frontend_s_from_pretest_real_cost": 0.5915700134598106,
        "Rescue_vs_V1": 43,
        "Harm_vs_V1": 16,
        "evaluation": "causal chosen mask retrieves that exact actual complete VLM generation from legal all16; selected policies rerun online independently"
      },
      "val_codes": [
        11,
        5,
        5,
        5,
        5,
        9,
        7,
        5,
        11,
        13,
        1,
        2,
        11,
        5,
        1,
        10,
        5,
        5,
        2,
        10,
        9,
        5,
        5,
        11,
        9,
        11,
        11,
        10,
        9,
        10,
        1,
        3,
        3,
        5,
        5,
        9,
        3,
        5,
        5,
        5,
        5,
        8,
        3,
        1,
        5,
        13,
        13,
        5,
        7,
        9,
        5,
        5,
        5,
        1,
        6,
        9,
        10,
        5,
        1,
        6,
        10,
        6,
        1,
        5,
        5,
        5,
        1,
        1,
        13,
        13,
        11,
        5,
        6,
        5,
        5,
        13,
        5,
        12,
        15,
        1,
        7,
        5,
        6,
        1,
        7,
        13,
        5,
        13,
        9,
        1,
        5,
        9,
        5,
        9,
        11,
        13
      ]
    },
    "audio_no_r": {
      "id": "audio_no_r_l0_t0.25_e10_p0",
      "spec": {
        "kind": "learned",
        "family": "audio_no_r",
        "checkpoint": "/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/models/audio_no_r_l0_t0.25/epoch10.pt",
        "checkpoint_sha256": "4c848a158a88f6c694ebdd39d4917739b55218f05a89fbb64c1c1b79f3a9ad72",
        "threshold": 0.0,
        "lambda": 0.0,
        "temperature": 0.25,
        "epoch": 10
      },
      "lambda": 0.0,
      "metrics": {
        "qa": 335,
        "sources": 96,
        "correct": 222,
        "accuracy": 0.6626865671641791,
        "gold_nll": 2.116660291734923,
        "mean_k": 3.9791666666666665,
        "ON_rate": 0.9947916666666666,
        "k_distribution": {
          "0": 0,
          "1": 0,
          "2": 0,
          "3": 2,
          "4": 94
        },
        "ON_per_time": [
          0.9791666666666666,
          1.0,
          1.0,
          1.0
        ],
        "estimated_frontend_s_from_pretest_real_cost": 0.6630889998252516,
        "Rescue_vs_V1": 45,
        "Harm_vs_V1": 18,
        "evaluation": "causal chosen mask retrieves that exact actual complete VLM generation from legal all16; selected policies rerun online independently"
      },
      "val_codes": [
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        14,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        14,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15
      ]
    },
    "visual": {
      "id": "visual_l0_t1_e20_p0.5",
      "spec": {
        "kind": "learned",
        "family": "visual",
        "checkpoint": "/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/models/visual_l0_t1/epoch20.pt",
        "checkpoint_sha256": "c512ceb1268df6e28f7905116e417140d773eb98b6c80cb3f54289a5d4a76189",
        "threshold": 0.5,
        "lambda": 0.0,
        "temperature": 1.0,
        "epoch": 20
      },
      "lambda": 0.0,
      "metrics": {
        "qa": 335,
        "sources": 96,
        "correct": 223,
        "accuracy": 0.6656716417910448,
        "gold_nll": 2.0148384206787817,
        "mean_k": 2.2604166666666665,
        "ON_rate": 0.5651041666666666,
        "k_distribution": {
          "0": 3,
          "1": 21,
          "2": 32,
          "3": 28,
          "4": 12
        },
        "ON_per_time": [
          0.8229166666666666,
          0.5,
          0.4791666666666667,
          0.4583333333333333
        ],
        "estimated_frontend_s_from_pretest_real_cost": 0.5511999047202052,
        "Rescue_vs_V1": 41,
        "Harm_vs_V1": 13,
        "evaluation": "causal chosen mask retrieves that exact actual complete VLM generation from legal all16; selected policies rerun online independently"
      },
      "val_codes": [
        15,
        11,
        1,
        11,
        5,
        15,
        11,
        1,
        13,
        13,
        3,
        11,
        5,
        11,
        5,
        7,
        0,
        5,
        1,
        10,
        11,
        3,
        2,
        7,
        11,
        1,
        5,
        10,
        11,
        13,
        13,
        15,
        5,
        2,
        5,
        1,
        6,
        2,
        9,
        2,
        1,
        5,
        5,
        8,
        7,
        9,
        15,
        1,
        15,
        9,
        0,
        5,
        11,
        5,
        5,
        9,
        1,
        1,
        13,
        13,
        9,
        14,
        5,
        11,
        9,
        2,
        2,
        5,
        7,
        13,
        11,
        7,
        15,
        5,
        1,
        1,
        2,
        6,
        15,
        9,
        3,
        5,
        15,
        7,
        1,
        15,
        15,
        9,
        11,
        7,
        5,
        11,
        0,
        2,
        15,
        15
      ]
    },
    "acoustic": {
      "id": "acoustic_l0_t0.25_e10_p0.5",
      "spec": {
        "kind": "learned",
        "family": "acoustic",
        "checkpoint": "/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/models/acoustic_l0_t0.25/epoch10.pt",
        "checkpoint_sha256": "b2506bb228623d5758ac8a9115b7b1684314237aa2e2f6ef252fcb09c6d3e416",
        "threshold": 0.5,
        "lambda": 0.0,
        "temperature": 0.25,
        "epoch": 10
      },
      "lambda": 0.0,
      "metrics": {
        "qa": 335,
        "sources": 96,
        "correct": 222,
        "accuracy": 0.6626865671641791,
        "gold_nll": 2.1085478739498265,
        "mean_k": 3.8333333333333335,
        "ON_rate": 0.9583333333333334,
        "k_distribution": {
          "0": 0,
          "1": 1,
          "2": 3,
          "3": 7,
          "4": 85
        },
        "ON_per_time": [
          0.9791666666666666,
          0.9270833333333334,
          0.96875,
          0.9583333333333334
        ],
        "estimated_frontend_s_from_pretest_real_cost": 0.6126943745629333,
        "Rescue_vs_V1": 44,
        "Harm_vs_V1": 17,
        "evaluation": "causal chosen mask retrieves that exact actual complete VLM generation from legal all16; selected policies rerun online independently"
      },
      "val_codes": [
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        5,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        1,
        15,
        15,
        15,
        15,
        15,
        15,
        13,
        15,
        14,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        7,
        15,
        15,
        15,
        13,
        15,
        15,
        15,
        15,
        15,
        15,
        9,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        11,
        14,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        13,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        15,
        5,
        15,
        15,
        15
      ]
    },
    "history": {
      "id": "history_l0_t0.25_e1_p0.5",
      "spec": {
        "kind": "learned",
        "family": "history",
        "checkpoint": "/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/models/history_l0_t0.25/epoch1.pt",
        "checkpoint_sha256": "3ae923809ecf7d0f995043902ecaa971c8b8b0b4af035de7f0763971a192f2fa",
        "threshold": 0.5,
        "lambda": 0.0,
        "temperature": 0.25,
        "epoch": 1
      },
      "lambda": 0.0,
      "metrics": {
        "qa": 335,
        "sources": 96,
        "correct": 225,
        "accuracy": 0.6716417910447762,
        "gold_nll": 2.0761107897845865,
        "mean_k": 3.0729166666666665,
        "ON_rate": 0.7682291666666666,
        "k_distribution": {
          "0": 0,
          "1": 1,
          "2": 30,
          "3": 26,
          "4": 39
        },
        "ON_per_time": [
          0.9791666666666666,
          0.8645833333333334,
          0.6041666666666666,
          0.625
        ],
        "estimated_frontend_s_from_pretest_real_cost": 0.6270000788261112,
        "Rescue_vs_V1": 47,
        "Harm_vs_V1": 17,
        "evaluation": "causal chosen mask retrieves that exact actual complete VLM generation from legal all16; selected policies rerun online independently"
      },
      "val_codes": [
        15,
        13,
        15,
        3,
        11,
        15,
        5,
        7,
        3,
        7,
        15,
        3,
        15,
        15,
        3,
        15,
        15,
        7,
        15,
        3,
        11,
        3,
        5,
        15,
        7,
        15,
        15,
        6,
        11,
        3,
        9,
        11,
        9,
        7,
        15,
        15,
        3,
        15,
        3,
        15,
        15,
        9,
        11,
        7,
        15,
        3,
        15,
        3,
        15,
        15,
        15,
        15,
        11,
        3,
        15,
        3,
        3,
        15,
        11,
        9,
        13,
        14,
        7,
        7,
        3,
        7,
        11,
        15,
        7,
        11,
        3,
        15,
        3,
        9,
        7,
        15,
        11,
        15,
        15,
        15,
        3,
        15,
        3,
        9,
        15,
        15,
        7,
        15,
        15,
        1,
        15,
        15,
        9,
        13,
        3,
        15
      ]
    }
  },
  "best_fixed_endpoint_from_validation": {
    "kind": "on"
  },
  "bootstrap_seed": 1234,
  "val_source_order": [
    "music:00008586",
    "music:00008701",
    "music:00008171",
    "music:00006530",
    "music:00006723",
    "music:00006170",
    "music:00001778",
    "music:00000744",
    "music:00007914",
    "music:00003364",
    "music:00004912",
    "music:00001000",
    "music:00000536",
    "music:00007689",
    "music:00006650",
    "music:00001216",
    "music:00007603",
    "music:00008774",
    "music:00001105",
    "music:00002980",
    "music:00002192",
    "music:00005141",
    "music:00005748",
    "music:00006716",
    "music:00008975",
    "music:00007691",
    "music:00008916",
    "music:00004461",
    "music:00006196",
    "music:00004990",
    "music:00004138",
    "music:00006145",
    "music:00007519",
    "music:00005133",
    "music:00008495",
    "music:00000714",
    "music:00000793",
    "music:00008744",
    "music:00002330",
    "music:00002701",
    "music:00000370",
    "music:00004086",
    "music:00002891",
    "music:00005263",
    "music:00001555",
    "music:00004058",
    "music:00007826",
    "music:00003106",
    "music:00000138",
    "music:00003647",
    "music:00008432",
    "music:00001741",
    "music:00008677",
    "music:00005555",
    "music:00003560",
    "music:00008108",
    "music:00007497",
    "music:00005025",
    "music:00005354",
    "music:00006290",
    "music:00000077",
    "music:00005772",
    "music:00002817",
    "music:00003618",
    "music:00005869",
    "music:00006821",
    "music:00005932",
    "music:00004725",
    "music:00008213",
    "music:00003096",
    "music:00006989",
    "music:00003526",
    "music:00002600",
    "music:00000669",
    "music:00007116",
    "music:00006744",
    "music:00008968",
    "music:00005893",
    "music:00000577",
    "music:00007329",
    "music:00000837",
    "music:00004065",
    "music:00007321",
    "music:00002972",
    "music:00005672",
    "music:00006366",
    "music:00005366",
    "music:00003821",
    "music:00006511",
    "music:00008360",
    "music:00005867",
    "music:00005069",
    "music:00000331",
    "music:00003811",
    "music:00000476",
    "music:00001649"
  ],
  "random_probability": {
    "accuracy_first": 0.7682291666666666,
    "cost_aware": 0.2942708333333333
  },
  "degen_policy_note": "all-off/on are valid; selected constant or threshold0/1 may be a degenerate Gate. No artificial nonzero ON target.",
  "source_code_sha256": "4e12f42ee123ddae70e97bc5be428f91dedcd3504b002d494a7369a6abf5c710"
}

```

## Original Stage1/Stage3 and native dual-image history

Original stage4 training source uses first-answer-token CE, not Full-Vision KD. Original checkpoint args are retained in inventory; unavailable optimizer details must not be guessed. Adaptive V1 warm-started existing Stage1,1968steps,3epochs,2active tensors933888; frozen backbone unchanged. Gate-first G_A fit F0, then R_A trained1968steps3epochs with forced exposure. Its epoch3 interruption/recovery and14zero-grad samples remain in original logs. G_A→R_A is mismatched frozen transfer. Native5,840-token RGB+Grounded andRGB+RGB ran separately and did not establish robust active audio benefit.

Historical reports, with original prompts/cohorts/source constraints, remain read-only:
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M0_dataset_research.md`, SHA `ab5b95cdf83306bbcec98c8a0e66de3e2898eeded723d9b48c8bf8cd4ddc70d6`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M0_repo_audit.md`, SHA `3585a28c58727ac9f774ec89b4cd65cdacbb9596cdf4ba8d94bf47f463b31d76`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M1_dataset_baselines.md`, SHA `805359bb48d8525c2597372d40bb92e0da74e4bb546f19e1dc5687dd160b8581`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M1_music_dev_baselines.md`, SHA `805359bb48d8525c2597372d40bb92e0da74e4bb546f19e1dc5687dd160b8581`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M1_music_dev_pilot_baselines.md`, SHA `a756ac87b6c97f4a906775c579979a4228c92ae2ef5cb5004ad5b832dc3915cf`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M2_localizer_and_preprocess.md`, SHA `a19783097a1a901ced7bc8f009342104515bb5e8b1c0f2c69b367068c5604106`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M3_dual_image_prompt_ablations.md`, SHA `ea09fa6392879d1ec8c7e2606e6e972c35086f56adcb4dd0a34b02f26843e6f2`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M4_development_selection.md`, SHA `e8f82c001a182ebbd64513141a398dc4ab2f49740d43ecb855202f3c31449ccb`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M5_MUSIC_confirmation.md`, SHA `a7d24baa046d91fcb8940030a835a2d3ff2b4cb9fb84c029b70cd29c9cd37d44`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M5_Valor_confirmation.md`, SHA `58cce8eb094d97376c506a5b3e0297e98506d11967001b9c21e1caccf075bbd4`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/reports/M6_final_judgment.md`, SHA `4f93927858e8ff530147cfedaee2a870e9e1e8ff45134dae0305bdf9ec131545`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M0_stage1_training_audit.md`, SHA `c0f52cd4d2493f1ec231c44a6ae3345b11f90a41c898628e80dae3300616b38d`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M1_stage1_train.md`, SHA `5c70ef17eb4dbbdc183d97a5cd118fedaa1bb446fd4466b244cbb2248e150ff9`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M1_stage1_training_QA_addendum.md`, SHA `3ab8f3dc1b8a7a2dc7b9ba5ac3b305953bf87ebea61d1f74a4fdbe7fb3706699`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M1_stage1_val.md`, SHA `69288ed7b6a5ce1f6762b6edffc78284d90308f3664faac19a91495d32fb6dac`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M2_fuser64.md`, SHA `8a4eb0521def957a72a52c2d1b09d173a9fb09ceb07a4e3e76b7dd5231db0ec3`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M3_label_distribution.md`, SHA `2c23f3ff17647f6db51bc75c7b09271351f6f2e4f677eea0dfbaf4630fff25ed`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M3_utility_interaction_addendum.md`, SHA `f81a83d843b33d024ead668bd0b210b642f79ac971a740b138c6cf9f0444cf82`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M4_gate_train.md`, SHA `2e25e6b1eb18316ae9d880df0c3276959f3d20cccfbfb96ac9e294b1d394dee5`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M4_gate_val.md`, SHA `2771275d6a20e8a5da23258b3e8a0394853156ebfacd1b26f147f7051e54a23c`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M5_pretest_policy.md`, SHA `295ce53d3a2e85a8d126f21df68671b13127cc8ebc1aef0731a4996680e79576`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M6_confirmation_qa.md`, SHA `fc1c9e1421298e4a48d2a854793e005f2dfe5a1c9d8e831d26b35e2e3cced12c`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/M7_conclusion.md`, SHA `07e03a41edacaa99832b9e35032dc98d39ebe136fa3753afc31a144494d1c841`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/reports/README.md`, SHA `d92780a7b20c68d3cd48dcdaa31361c9a012c3afe2cde74bc7685c56c18ccefc`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/reports/M0_audit.md`, SHA `62846a4a42027cd37d56ab77303ef9af1407ff6b713d8f5b428a26f9ba41613c`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/reports/M1_utility_labels_and_interactions.md`, SHA `52734b9bda34d98493b710c7bd483143a147352d15b7649ea0e7282c71e3f2ca`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/reports/M2_gate_training_and_validation.md`, SHA `1c352d5d6ac24a9afa5d4c93771c5e37e622915c6cc0809c3cc192d9fbe9d3c3`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/reports/M3_router_training.md`, SHA `a14de7526321451438b7e07f570aa74c33fb9a52eedbee842b4e478d6ecf4e9a`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/reports/M4_router_qa_ablation.md`, SHA `ecf287c392e5de05ed1d9ace5993bface638b084b97e837163c6937206fa7e70`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/reports/M5_gate_only_native_qa.md`, SHA `a05a184d0547decbf6ec008290a3ba0a0d3dcd0f5c9aca4d18f3ef17d28d64a3`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/reports/M6_paired_analysis_and_cost.md`, SHA `fccb76d6b1d2b44228593781644bad1a8cf8a5558c2ca93b8ae3837f72a890cf`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/reports/M7_final_judgment.md`, SHA `fa3c0173f3c760c13651edb1e85de3248b1947ab74095022a35e1534e0642baf`
- `/home/yxd/audio_pic_gate/artifacts/utility_gate/gate_first_router_second_native_ablation_v1/reports/README.md`, SHA `bd138905e2c421e89b51a6ac025607361678ecb6dbcad4aa35fb996a6fe1129f`

## Latest fixed R_A Gate search (new 8s-prefix cohort)

192official-train sources/656QA;96fresh validation/335QA;16whole-mask actualQA.40Gates,14400newGate optimizer steps, noRouter updates. Confirmation:200newMUSIC numericIDs/282QA,252newknownYT Valor/504QA. Prompts native MUSIC short-answer S0 versus native Valor ABCD S0, greedy16newtokens, times2/4/6/8, rawaudio[t−2,t] cropped before resampling. NLL canonical-string sum forMUSIC;4-option normalized forValor; never pooled.

|Role|Strategy|Correct/QA|NLL|mean k|Serial compute s/QA|
|---|---|---:|---:|---:|---:|
|val|T0_V1|195/335|4.961191|0.0000|0.563605|
|val|T1_RA_always|222/335|2.116660|3.9792|0.815665|
|val|T2_old_gate|208/335|2.980278|1.2708|0.740876|
|val|T3_accuracy|225/335|2.076111|3.0729|0.823233|
|val|T4_cost|222/335|1.825052|1.1771|0.690948|
|confirmation_music|T0_V1|153/282|5.148867|0.0000|0.620074|
|confirmation_music|T1_RA_always|175/282|2.589514|3.9900|0.889384|
|confirmation_music|T2_old_gate|160/282|3.668367|1.1400|0.795819|
|confirmation_music|T3_accuracy|174/282|2.464998|3.0600|0.889404|
|confirmation_music|T4_cost|171/282|2.226360|1.1800|0.754297|
|confirmation_valor|T0_V1|310/504|0.966102|0.0000|0.786754|
|confirmation_valor|T1_RA_always|310/504|1.208270|3.9444|1.110920|
|confirmation_valor|T2_old_gate|310/504|1.059786|1.2619|0.979710|
|confirmation_valor|T3_accuracy|309/504|1.171458|2.5040|1.068081|
|confirmation_valor|T4_cost|308/504|1.126572|1.2143|0.933098|

Latest six-control/rate comparisons, source-level intervals and all negative outcomes: `/home/yxd/audio_pic_gate/artifacts/utility_gate/fixed_router_gate_v1/paired_summary.json`. SELECT/efficiency INCONCLUSIVE; audio specificity NO-GO. Zero observed flips are not equivalence evidence.
