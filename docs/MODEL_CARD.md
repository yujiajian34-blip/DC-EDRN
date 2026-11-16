# Model card

DC-EDRN combines a duration-collapsed recurrent byte encoder, hashed word context and supervised three-way intent conditioning of span evidence. Intended use is research on prompt-injection screening before context assembly, with explicit measurement of rejection of instruction-like benign content.

## Checkpoints and operating point

`runtime/checkpoint_manifest.json` lists 18 original primary checkpoints, hashes and stored thresholds. `systems/reference_checkpoints/` contains three separate seed-42 performance references with identities recorded in `provenance/systems_sources.json`; do not interchange differently named artifacts without verifying contents and protocol. The runtime defaults to DC_EDRN:42, FP32, CPU, batch 32 and one CPU thread. The model truncates at 512 UTF-8 bytes and 128 lexical tokens.

Original validation-based checkpoint/threshold selection is retained in primary comparisons; the fixed-checkpoint supplement calibrates thresholds on filtered validation. Neither is a guarantee of future benign false-positive rates. The score is a sigmoid classifier output, not a calibrated probability of compromise.

## Evaluation and limits

Three seeds yield F1 0.932 ± 0.008 on 885 filtered internal test rows; held-out NotInject FPR is 0.024. No false positives were observed on the 500-row over-defense challenge. Fixed character normalization was evaluated against nine character-perturbation families; it is not enabled in the portable CLI and does not establish adaptive-attack robustness.

On the balanced 400-row BIPIA mixture, F1 is 0.645 and FPR 0.953. Performance may deteriorate under new task contexts, language, encoding or long inputs. Evaluation does not establish a general security guarantee or end-to-end LLM safety.

Training parameters: 323,754. With the auxiliary duration head removed: 323,271 inference parameters, approximately 1.23 MiB in FP32. Precomputation removes repeated projections but retains the recurrent scan. Optimized BiLSTM/ByteCNN baselines remain faster in the reported GPU comparison.

## Rights and data

No model license has yet been assigned. Dataset source notices and historical revision limitations are in THIRD_PARTY_NOTICES.md. Raw/derived prompt datasets remain outside this candidate. Public release of code and weights awaits author confirmation of provenance and intended terms.
