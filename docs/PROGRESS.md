# PROGRESS

## Current status
- **Active phase:** Phase 0 (environment, data, and evaluation simulator)
- **Next task:** set up the environment, download the dataset (decision 0a), check its decompressed size, and inspect the HDF5 format
- **Blockers:** the simulator source (decision 0b) is open and needed before the simulator-vs-dataset rendering check

## Decisions
Record each decision from PLAN.md Section 4 as it is resolved: number, choice, provenance ([PAPER] / [OFFICIAL-REFERENCE] / [PROJECT DECISION]), rationale, date.

| # | Decision | Choice | Provenance | Rationale | Date |
|---|---|---|---|---|---|
| 0a | Dataset source | HF dataset repo `quentinll/lewm-pusht`, file `pusht_expert_train.h5.zst` (~13.1 GB, zstd-compressed HDF5, MIT) | Owner | Official dataset release for the paper's PushT task | 2026-11-23 |
| 0b | Evaluation simulator source | OPEN | Owner | Must render frames that match the dataset | |

## Phase checklist
- [ ] Phase 0: environment, data, simulator
- [ ] Phase 1: encoder, projectors, action encoder, predictor
- [ ] Phase 2: SIGReg
- [ ] Phase 3: training loop and smoke training
- [ ] Phase 4: 10-epoch training run
- [ ] Phase 5: probing
- [ ] Phase 6: CEM planner and MPC evaluation
- [ ] Phase 7: surprise evaluation
- [ ] Phase 8: additional seeds and write-up (stretch)

## Experiments
| Run ID | Config | Seed | Git hash | Runtime | Peak VRAM | Key results | Artifacts |
|---|---|---|---|---|---|---|---|

## Deviations from the paper
(none yet)

## Log
Newest entries first. For each meaningful event, record: what happened, the root cause, what was tried, the verified fix, and the evidence.
