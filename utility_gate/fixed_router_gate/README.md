# Fixed R_A / V1, Gate-only complete-mask experiment

This is an additive namespace. It never calls a Router or foundation-model
training entry, and it never writes historical experiment directories.
The immutable experiment is
[`fixed_router_gate_v1`](../../artifacts/utility_gate/fixed_router_gate_v1/).

`data.py` selects legal train/fresh-val/new-confirmation source roles before
scores; `isolation.py` checks known historical IDs, media bytes and aliases.
`features.py` permits train/val caching only. `enumerate_masks.py` executes all
16 requested masks with real full QA generation and native Gold scoring under
the **trained R_A**, with quality-forced OFF recorded. `gate_train.py` is the
only new optimizer: source-level autoregressive complete-mask distillation.
No single-window QA utilities are summed to produce labels.

`selection.py` selects Accuracy-first and Cost-aware on fresh validation;
`timing.py` measures uncached front ends before confirmation. `integration.py`
checks real P/Z/logits/generated IDs, historical trained R_A replay,
causality, and k=0..4 encoder calls. `evaluate.py` reruns every policy with
independent conditional front ends. Only ON invokes EZ-VSL and the second
SigLIP pass; final visual tokens are always 4×64. `locks.py` blocks confirmation
until full live validation and tests pass, then seals models, code and sources.

The stage commands, using the existing `.venv-gate` environment, are:

```bash
bash scripts/run_fixed_router_gate.sh development
bash scripts/run_fixed_router_gate.sh fit
bash scripts/run_fixed_router_gate.sh validation
# Run tests and then: python -m utility_gate.fixed_router_gate.locks
bash scripts/run_fixed_router_gate.sh confirmation
bash scripts/run_fixed_router_gate.sh close
# After close, verify derived metadata and add descriptive uncertainty/frontiers:
# .venv-gate/bin/python -m utility_gate.fixed_router_gate.verified_summary
```

Initial acquisition/protocol and exact subprocess commands are recorded in
the run's `commands.json` and logs. Existing complete source outputs are only
read on resume; a partial fit raises instead of silently restarting. To run a
new scientific experiment, use a new version namespace and new eligible
sources; do not reuse this confirmation cohort for tuning.

The authoritative outputs are `reports/M0_audit.md` through M6,
`reports/FINAL_JUDGMENT.md`, per-QA/per-window files, `paired_summary.json`,
`pareto.csv/png`, `pretest_lock.json`, model SHA records and `final_audit.json`.
Report GO-SELECT, GO-EFFICIENCY and GO-AUDIO-SPECIFIC independently. MUSIC
canonical-string NLL and Valor ABCD-normalized NLL are never pooled. These
are 8-second observed-prefix QA measurements, not official full-video scores;
MUSIC original YouTube lineage remains unknown. Native N0 is a historical
reference only, not a formal route with a different token budget.

The completed Chinese overview is
[`RESULTS_OVERVIEW_ZH.md`](../../artifacts/utility_gate/fixed_router_gate_v1/reports/RESULTS_OVERVIEW_ZH.md).
`verified_summary.py` is postprocessing only: it preserves the first emitted
summary, corrects the three-seed mean component dictionary, and asserts all
primary QA/mask/cost values and paired contrasts are unchanged. Its source SHA
is recorded separately from the frozen pretest inference code. It adds cost/ON
intervals and all-K descriptive comparisons; it never selects test policies.
