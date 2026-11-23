# PROGRESS

## Current status
- **Active phase:** Phase 0 (environment, data, and evaluation simulator)
- **Stages done:** 0.1 (environment + GPU smoke test), 0.2 (dataset download + inspection)
- **Next task:** owner review of the data findings (episode count/length mismatch, 185×101 group structure → split unit), then plan stage 0.3 (loader, split, windows, alignment tests, throughput)
- **Blockers:** exact package name/install source for the decision-0b simulator (needed before stage 0.4)

## Decisions
Record each decision from PLAN.md Section 4 as it is resolved: number, choice, provenance ([PAPER] / [OFFICIAL-REFERENCE] / [PROJECT DECISION]), rationale, date.

| # | Decision | Choice | Provenance | Rationale | Date |
|---|---|---|---|---|---|
| 0a | Dataset source | HF dataset repo `quentinll/lewm-pusht`, file `pusht_expert_train.h5.zst` (~13.1 GB, zstd-compressed HDF5, MIT) | Owner | Official dataset release for the paper's PushT task | 2026-11-23 |
| 0b | Evaluation simulator source | (i) PushT env from the LeWM authors' benchmark library, used for the environment only; exact package source pending from owner | Owner | Most likely to render frames matching the dataset | 2026-11-23 |
| — | Python / PyTorch | Python 3.12 (uv-managed), torch 2.11.0+cu128 from the PyTorch cu128 wheel index | [PROJECT DECISION] | cu128 wheels include sm_120 (Blackwell); 3.12 has broad wheel availability | 2026-11-23 |

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

## Data format note (verified 2026-11-23)
Source: `pusht_expert_train.h5.zst` from the HF **dataset** repo `quentinll/lewm-pusht`. 13,136,247,974 B compressed, 46,300,921,856 B decompressed; zstd checksum verified on decompression. Local paths: `data/raw/…zst`, `data/pusht_expert_train.h5` (both gitignored). Inspector: `uv run python -m lewm.data.inspect --path data/pusht_expert_train.h5 --out outputs/inspect/`.

**Layout:** flat. Each per-step array has N = 2,336,736 rows. There are no groups and no attrs.

| Dataset | Shape | Dtype | Storage |
|---|---|---|---|
| `pixels` | (N, 224, 224, 3) | uint8 | HWC RGB. Blosc filter (id 32001), chunks of 100 frames (15 MB uncompressed per chunk), 34.0 GiB stored, ≈15.6 KB/frame |
| `action` | (N, 2) | float32 | uncompressed |
| `proprio` | (N, 4) | float32 | uncompressed; equals `state[:, [0, 1, 5, 6]]` exactly |
| `state` | (N, 7) | float32 | uncompressed |
| `episode_idx`, `step_idx` | (N,) | int64 | uncompressed |
| `ep_len` | (E,) | int32 | E = 18,685 |
| `ep_offset` | (E,) | int64 | |

- **Episode boundaries.** Episode `e` covers rows `ep_offset[e] : ep_offset[e] + ep_len[e]`. The four index fields are mutually consistent:
  - `ep_offset` is the cumulative sum of `ep_len`.
  - Each run of `episode_idx` equals `ep_len`, and the ids run 0..E−1 in order.
  - `step_idx` goes 0..len−1 within each episode.
- **Episode count and length differ from the paper.** There are **18,685** episodes, against 20,000 in the paper. Mean length is **125.06** steps, against 196 (min 49, median 123, max 246, std 36.4). No episode is shorter than 20 steps, so every episode fits at least one 4-frame / 20-step window.
- **Group structure.** There are **185 distinct initial states, each shared by exactly 101 episodes**. Each group occupies a contiguous block of 101 episode indices, and all 101 episodes in a group have the same length.
  - No two trajectories are identical. Actions differ from step 0 (max |Δa| ≈ 0.2–0.5).
  - Agent positions diverge by ~10 px at t=1 and 30–70 px by t=5–20.
  - Final block poses spread widely (std ≈ 30–60 px, ~0.6–1.5 rad). Note that 0/2π angle wrap-around inflates the raw angle deviations.
  - First frames of two siblings in the same group differ by at most 10 intensity levels.
- **Action semantics.**
  - Over all within-episode steps, the correlation between `action[t]` and `agent_pos[t+1] − agent_pos[t]` is 0.99 per dimension. Shifting the action by one step (`action[t+1]`) gives 0.86.
  - So `action[t]` is the action applied between frame t and frame t+1, consistent with the paper's indexing of a_k between o_k and o_{k+1}.
  - The last row of each episode has an action (not zero/NaN) but no successor frame.
  - Value range: min [−1.50, −1.47], max [2.04, 1.79], mean ≈ 0, std ≈ 0.21 per dimension. The units and any normalization are **not** documented in the file.
- **State fields.**
  - `state[0:2]` = agent x, y. `state[5:7]` correlates 0.99 with the agent's position change from t to t+1, so it is probably agent velocity.
  - `state[2:4]` = block x, y (range ≈ 0–540).
  - `state[4]` = block angle in [0, 2π).
  - These labels are inferred from ranges and correlations, not from documentation. 0.14% of agent positions and 0.007% of block positions fall outside [0, 512].
- **Pixels.** Sampled frames span 56–255 with mean ≈ 248, a mostly white background. The grid (`outputs/inspect/grid_pixels.png`) shows the grey T block, a fixed green goal T and the blue agent.
- **Window counts** (one window spans 20 environment steps):
  - start at every step: 1,981,721 windows, ≈ 15,483 steps/epoch at batch size 128
  - start every 5 steps: 404,202 windows, ≈ 3,158 steps/epoch

## Log
Newest entries first. For each meaningful event, record: what happened, the root cause, what was tried, the verified fix, and the evidence.

### 2026-11-23 — Stage 0.2: dataset download and inspection
- Downloaded `pusht_expert_train.h5.zst` with `hf download … --repo-type dataset`; the command exited with code 0. `zstd -lv` reported 43.1 GiB decompressed against 351 GiB free. The owner approved decompression, which took 57 s and verified the checksum.
- Wrote `src/lewm/data/inspect.py`, a read-only inspector, and `tests/test_inspect.py` (synthetic file with the real layout, plus a corrupted-`step_idx` negative case). Findings are in the data format note above.
- Inspector bug found and fixed: the `ep_` keyword match also hit `step_idx`, producing a spurious "episode id" candidate. The match now uses the basename prefix, and a test asserts `step_idx` is not a candidate.
- Grid bug found and fixed: it crashed when there were fewer episodes than grid rows (`IndexError`). It now caps the row count and spreads the sampled episodes across the file. Before that change, the first 6 episodes were identical in the grid. Checking why revealed the 185 × 101 group structure; it was not a plotting bug.
- `uv run pytest -q`: 6 passed.

### 2026-11-23 — Stage 0.1: environment and GPU
- Moved the paper to `references/LeWorldModel.pdf`. Added `pyproject.toml` (Python 3.12, torch from the cu128 index), the `.gitignore` entries and the `src/lewm` layout.
- `scripts/check_gpu.py`: torch 2.11.0+cu128, CUDA 12.8, RTX 5070 Ti, sm_120, 15.53 GiB, bf16 supported. The BF16 autocast matmul and linear forward/backward gave bf16 activations, fp32 gradients, all finite: PASS.
- `tests/test_cuda_smoke.py`: 2 passed.
