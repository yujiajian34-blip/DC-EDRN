# Map from the submitted manuscript to archived evidence

This map follows the local 2026-09-19 Track 2 manuscript. It does not imply acceptance. No new GPU measurements were generated during packaging.

| Claim | Evidence within candidate | Interpretation |
|---|---|---|
| F1 0.932 ± 0.008 | `results/primary_three_seed/summary.csv`, `metrics_by_seed.csv`, `predictions.csv.gz` | Three seeds; 885 filtered test rows; original thresholds |
| Original-942 comparisons | `results/historical_original942/` | Separate original protocol, not the 885-row primary result |
| Fixed-seed recalibration | `results/fixed_seed42/` | Existing weights, filtered validation threshold selection |
| Character perturbations | `results/robustness/` | Fixed transformations, not adaptive attacks |
| BIPIA F1 0.645, FPR 0.953 | `results/transfer/bipia_mixed_external_summary.csv` | Domain-transfer limitation |
| Table 6, batch 1: 66.85 → 48.61 ms | `results/performance_3090/screening_cost.csv` | `tokenize_transfer_forward_scores` rows, same-run RTX 3090 |
| Table 6, batch 32: 69.26 → 50.96 ms | Same file | Mean per-batch latency, not per-text latency |
| Unchanged threshold decisions on 3,827 rows | `results/performance_3090/parity.json` | Both original and recalibrated thresholds; not bitwise equality |
| Worker periodic 40/s: 19.6 vs 39.9/s; P95 19,854.3 vs 100.3 ms | `results/live_gateway_timerfix/summary.csv` | Means of three per-run summaries, including post-window drain |

The nearly 20-second P95 is a finite-window backlog result, not steady-state latency. The worker has a 20-second offered window, three repetitions, immediate service versus batching capped at 32 with up to 5 ms wait. Archived raw request/batch times support independent aggregation. CPU queue pilot results under `validation/` and older `results/fixed_seed42/screening_cost.csv` are not Table 6 or the final live-clock worker table.

The included source under `systems/code/` comes from the final returned GPU archives, with packaging changes recorded separately. `results/performance_3090/run_manifest.json` retains original run fingerprints and original code hashes; use `MANIFEST_SHA256.json` for packaged file identities. Device UUID is redacted. Historical intermediate failure logs and profiler traces were omitted; original evidence archives remain private.
