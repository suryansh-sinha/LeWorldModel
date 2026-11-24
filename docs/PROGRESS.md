# PROGRESS

## Current status
- **Active phase:** Phase 4 (evaluation). 4a probes done on the ×N retrain; 4b (simulator, CEM) next
- **Done:** Phase 0 (data); Phase 1 (model); Phase 2 (SIGReg and loss); Phase 3 (training, full 10-epoch run)
- **Next task:** paused; the owner will decide the follow-up later. Diagnosis: model prediction error, not CEM, limits planning (candidate: retrain at stride 1; alternative: accept 60%) Checkpoint: `outputs/runs/pusht_scaleN/last.pt`
- **Blockers:** none

## Decisions
One line per decision: what was chosen, and why.

| # | Decision | Choice | Provenance | Rationale | Date |
|---|---|---|---|---|---|
| 0a | Dataset source | HF dataset repo `quentinll/lewm-pusht`, file `pusht_expert_train.h5.zst` (~13.1 GB, zstd-compressed HDF5, MIT) | Owner | Official dataset release for the paper's PushT task | 2026-11-23 |
| 0b | Evaluation simulator source | (i) the `swm/PushT-v1` Gymnasium env from `stable-worldmodel` (galilai-group), used only for environment validation and downstream evaluation. None of the library's models, training code, dataset loaders, planning solvers or CEM are used; all LeWM components, including the data loader, are written in this repo | Owner | Most likely to render frames matching the dataset; its documented state and proprio layout matches the HDF5 file | 2026-11-23 |
| — | Train/val split unit | 90/10 split by **starting-state group** (185 groups × 101 episodes), seed 3072, made before windowing; held-out eval and probe sets also respect group boundaries | Owner ([PROJECT DECISION]; ratio [OFFICIAL-REFERENCE]) | Chosen over a random episode-level split, which would leak: the 101 sibling episodes in a group share their first frame and closely related early trajectories, so val would contain near-copies of training data. Splitting by group means val tests unseen starting states | 2026-11-23 |
| — | Group id | `group = episode // 101`; val = first 19 of `default_rng(3072).permutation(185)`, train = remaining 166; saved to `configs/splits/pusht_group_split.json` | [PROJECT DECISION] | Follows the verified contiguous 185 × 101 layout; asserted on load | 2026-11-23 |
| — | Action scaling in the Dataset | Dataset returns raw actions; standardization (PLAN default) is applied in Phase 3 | [PROJECT DECISION] | Keeps the loader a pure row reader; stats come from train groups only | 2026-11-23 |
| — | Weight init | trunc-normal std 0.02 for Linear/Conv/pos-emb/CLS, zero biases, adaLN modulation zero-init | [PROJECT DECISION] | Paper silent; standard ViT/DiT init | 2026-11-24 |
| — | Predictor dropout | 0.1 on the attention output and MLP output inside each residual branch; none in the encoder | [PROJECT DECISION] | Paper gives the rate only | 2026-11-24 |
| — | Predictor final norm | plain LayerNorm (no adaLN) before the projector; adaLN-Zero (shift, scale, gate) in each of the 6 blocks | [PROJECT DECISION] | Paper: "AdaLN applied at each layer" | 2026-11-24 |
| — | SIGReg integration range | Full ℝ: 2 × trapezoid over [0.2, 4] (the EP integrand is even in t) | Owner | App. A defines the EP integral over all of ℝ; the quadrature on [0.2, 4] covers t > 0 only, so ×2 recovers the full range. Initially implemented without the ×2 (half range), changed by owner 2026-11-24 | 2026-11-24 |
| — | SIGReg batch scaling | **×N (N = batch size 128): the conventional N·T Epps–Pulley statistic**, via `sigreg_scale_n: true` in `configs/pusht.yaml`. Supersedes the original "no ×N" choice (2026-11-24), which follows App. A's formula literally | Owner | The first 10-epoch run without ×N trained "fine" by its metrics, but the probes showed **representational (dimensional) collapse**: effective rank of z ≈ 3/192, probe r ≈ 0.5–0.6. A 1-epoch ablation confirmed the cause: effective rank 1.0 without ×N vs 25.5 with ×N; block MLP r 0.54 → 0.92. λ = 0.1 unchanged. See the Phase 4 log entry | 2026-11-24 |
| — | SIGReg weight | w(t) = exp(−t²/2) (paper's width parameter = 1) | [PROJECT DECISION] | Paper gives the form, not the width | 2026-11-24 |
| — | SIGReg directions | 1024 fresh directions per step, shared by the 4 time steps within a step | [PROJECT DECISION] | Paper silent on sharing; simplest | 2026-11-24 |
| 9 | Window stride | **Stride 5** for the full run (2,824 train steps/epoch, 28,240 total, ≈ 1.8 h at 4.40 it/s) | Owner | Fits the time available; stride 1 (13,845 steps/epoch, ≈ 8.7 h) can be revisited later. Each window still holds 4 frames 5 steps apart; stride only sets how often windows start | 2026-11-24 |
| — | Dataset frame reads | Read the contiguous rows `r0:r0+16`, then keep every 5th (`[::5]`), instead of h5py list indexing `pixels[[r0, r0+5, r0+10, r0+15]]` | Owner | Identical frames (equality test), but 27× faster in one process (33 → 909 windows/s), and the DataLoader goes from 2.5 to 15.4 batches/s with 12 workers. Training becomes GPU-bound | 2026-11-24 |
| — | AdamW weight decay | 1e-3 on all parameters, including biases and norms | [PROJECT DECISION] | Paper silent; wd is tiny | 2026-11-24 |
| — | Action standardization | Per-dimension mean/std over all action rows of train-group episodes (mean ≈ [−0.0072, 0.0060], std ≈ [0.2106, 0.2059]), applied on the GPU, saved in every checkpoint | [PROJECT DECISION] (PLAN default) | Planning in Phase 4 needs the same scaling | 2026-11-24 |
| — | Checkpoints / resume | Saved at epoch end (`epoch_k.pt` + `last.pt`; smoke keeps only `last.pt`); `--resume` continues from the next epoch. Per-epoch data order is seeded with `seed + epoch` | [PROJECT DECISION] | Simplest; a crash loses at most part of one epoch | 2026-11-24 |
| — | Validation loss | Every epoch, on 20 fixed val-group batches (2,560 windows, seeded), in eval mode | Owner | Spots overfitting over the 10 epochs | 2026-11-24 |
| — | Probe protocol | Frames: every 5th row of all val-group episodes (~50k), post-projector z, eval mode. Labels: agent `state[0:2]`, block `state[2:4]`, raw angle `state[4]`, z-scored with probe-train stats. Probes: linear least squares with bias; MLP 192→256→out (GELU, AdamW 1e-3, 100 epochs, batch 1024). Two splits, both reported: **random** 75/25 frame split (seed 0; comparable to Table 1) and **group** split (14 / 5 val groups; unseen starting states) | Owner (both splits); rest [PROJECT DECISION] | The paper doesn't state its probe split. The group split is stricter: 14 starting states don't cover the block poses, so block and angle drop | 2026-11-24 |
| — | Simulator install | `stable-worldmodel` 0.1.1 with only PushT's runtime deps (`opencv-python-headless`, `pygame`, `pymunk`, `shapely`, `imageio`), not the heavy `[env]` extra (JAX/craftax, ALE, mujoco, …). `torchvision` is pinned as a direct dependency from the cu128 index | [PROJECT DECISION] | The PyPI torchvision didn't match our cu128 torch (`torchvision::nms does not exist`); uv sources only apply to direct deps. torch stays 2.11.0+cu128 | 2026-11-24 |
| — | Env reset and goal | `env.reset(options={"state": s_t, "goal_state": s_{t+25}})` with dataset 7-d states; goal image = dataset frame at t + 25 | [PROJECT DECISION] | Exact reset is supported through reset options; the green target T is drawn at a fixed pose, as in the dataset | 2026-11-24 |
| — | Action units | Dataset actions are already in env units (relative mode: PD target = agent pos + 100 × action). No conversion | Verified | Replaying 25 dataset actions reproduces `state[t+25]` within ~1–2 px | 2026-11-24 |
| — | CEM action clipping | De-normalize CEM actions, then clip to the env's declared `[-1, 1]` | Owner (PLAN default) | The env doesn't enforce it, but only 0.12% of dataset rows exceed it; it also keeps CEM away from actions rarely seen in training | 2026-11-24 |
| — | CEM details | Samples are clipped (in normalized units) before evaluation, so the cost scores exactly what would be executed. Context = last ≤3 real frames at 5-step spacing plus their executed (clipped, re-normalized) blocks. Output = final CEM mean μ_T; no variance floor; per-pair seed = pair index. An episode succeeds at the first step where the env's check passes, within 50 steps | [PROJECT DECISION] | Paper silent on these details | 2026-11-24 |
| — | Eval pairs | 50 (episode, t) pairs from val groups, `default_rng(0)`, t uniform in [0, L − 26], goal at t + 25; saved to `configs/eval/pusht_eval_pairs.json` | [PROJECT DECISION] (PLAN protocol) | Fixed once; don't tune on them | 2026-11-24 |
| — | Python / PyTorch | Python 3.12 (uv-managed), torch 2.11.0+cu128 from the PyTorch cu128 wheel index | [PROJECT DECISION] | cu128 wheels include sm_120 (Blackwell); 3.12 has broad wheel availability | 2026-11-23 |

## Phase checklist
- [x] Phase 0: data
- [x] Phase 1: model
- [x] Phase 2: SIGReg and loss
- [x] Phase 3: training (smoke run, stride decision, 10-epoch run)
- [ ] Phase 4: evaluation (probes, planning, optional surprise)
- [ ] Stretch: λ = 0 run, more seeds, README

## Experiments
| Run ID | Config | Seed | Git hash | Runtime | Peak VRAM | Key results | Artifacts |
|---|---|---|---|---|---|---|---|
| plan_pusht_scaleN | `lewm.plan`, CEM 300/30/30, H = 5, 50 pairs, budget 50 | per-pair | uncommitted (on top of `2b0cdf4`) | ≈ 1 min | — | **Success 30/50 = 60% (95% CI 46–74%)**; plan time 0.35 ± 0.05 s (78 plans); 0 trivial starts | `outputs/eval/plan_results.json`, `outputs/eval/gifs/` |
| pusht_scaleN (wandb `12n66vyd`) | `configs/pusht.yaml` (stride 5, ×N), 28,240 steps | 0 | uncommitted (on top of `63cd295`) | ≈ 1 h 36 min (4.94 it/s) | ≈ 11.2 GiB | Final train: pred 0.018, SIGReg (×N) 0.98, emb_std 0.99, effective rank 64. Final val (eval mode): pred 0.031, SIGReg 2.52, emb_std 0.95. Probe r (lin/MLP), random split: agent 0.987/0.999, block 0.982/0.999, angle 0.880/0.984; group split: agent 0.964/0.988, block 0.915/0.929, angle 0.593/0.467 | `outputs/runs/pusht_scaleN/`, `outputs/eval/pusht_scaleN/probes.json` |
| ablate_base | pusht.yaml, 1 epoch (2,824 steps), no ×N | 0 | uncommitted (on top of `63cd295`) | ≈ 10 min | 11.2 GiB | Effective rank 1.0; probe r (lin/MLP): agent 0.17/0.06, block 0.53/0.54, angle 0.39/0.46 | `outputs/runs/ablate_base/`, `outputs/eval/ablate_base/` |
| ablate_scaleN | pusht.yaml, 1 epoch (2,824 steps), ×N | 0 | uncommitted (on top of `63cd295`) | ≈ 10 min | 11.2 GiB | Effective rank 25.5; probe r (lin/MLP): agent 0.56/0.43, block 0.87/0.92, angle 0.38/0.69 | `outputs/runs/ablate_scaleN/`, `outputs/eval/ablate_scaleN/` |
| pusht (wandb `oayb7cn3`) | `configs/pusht.yaml`, stride 5, 28,240 steps | 0 | `26d624b` | ≈ 1 h 36 min (4.95 it/s; estimated from checkpoint timestamps) | 11.21 GiB | **Collapsed (superseded):** effective rank ≈ 3/192, probe linear r agent 0.59 / block 0.64 / angle 0.60. Final train: pred 0.0028, SIGReg 0.103, emb_std 0.79. Final val (eval mode, 20 batches): pred 0.0024, SIGReg 0.114, emb_std 0.78 | `outputs/runs/pusht/epoch_{0..9}.pt`, `last.pt`, `outputs/runs/pusht_full.log` |

## Deviations from the paper
- **Dataset size (known, accepted by owner 2026-11-23).** The released `pusht_expert_train.h5` has **18,685 episodes with a mean length of 125.06 steps** (2,336,736 steps). Appendix E, p. 18, describes 20,000 episodes averaging 196 steps. The cause is unknown: 198 × 101 = 19,998 ≈ 20,000 suggests some starting-state groups may have been held out, but this is unverified. We use the file as released, with no resampling or padding to match the paper.

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

### 2026-11-24 — Phase 4b-1: diagnosis, model vs planner
- **Setup:** 50 dev pairs (seed 1, val groups, episodes disjoint from the 50 eval pairs, which stay untouched). One plan per pair (25 steps, no replanning), comparing the expert's own 25 actions (clipped) with CEM's plan.
- **Sim success:** expert 50/50, CEM 29/50 (in line with the 60% on the eval pairs).
- **The optimizer is not the limit:** CEM finds a lower *predicted* latent cost than the expert in 49/50 pairs, including all 21 CEM failures (median predicted cost 4.3 vs 19.5).
- **The model is:**
  - Its 5-block autoregressive prediction error ‖ẑ₅ − z(reached frame)‖² is ≈22 even on expert actions (≈24 on CEM actions), about 7% of the median start-to-goal latent distance (292).
  - CEM exploits that error: predicted cost 4.3 vs real cost 24.5 for the frame actually reached.
  - Median CEM final error is 20.9 px / 0.076 rad, right at the 20 px threshold.
- **Implication:** improving planning needs a more accurate world model, not planner tuning. The main known gap from the paper's likely setup is training length (stride 5: 28k steps; stride 1 would be ≈138k).

### 2026-11-24 — Phase 4b-1: CEM planner and first evaluation (60% vs 96%)
- **Code:**
  - `src/lewm/plan.py`: `cem` (Alg. 2), `rollout_cost` (autoregressive latent rollout, context ≤3), MPC loop (plan 5 blocks, execute 25 steps, replan), eval pairs, GIFs, and per-pair final errors.
  - `tests/test_plan.py` (3 tests): CEM converges on a quadratic; CEM respects the bounds; the rollout context window (real-then-predicted entries, never more than 3).
- **Result on the 50 fixed pairs:** **30/50 = 60%** (95% CI 46–74%), against the paper's 96.0 ± 2.8%.
  - Plan time 0.35 ± 0.05 s, faster than the paper's ~1 s. The first plan per process is ~2 s because of CUDA warm-up.
  - No pair succeeds trivially: no start already satisfies the goal.
  - A rerun gave the same 30/50.
- **The 20 failures:**
  - Median final error: 41 px (agent + block xy combined; needs < 20) and 0.15 rad (needs < 0.35).
  - 10 are within 2× of both thresholds.
  - **9 fail only on the agent position**: the block's position and angle are within the thresholds, but the agent isn't.
  - About 5 are large misses (100–220 px).
  - Failures start early in their episodes (median t = 31 vs 68 for successes), when the expert makes bigger moves within the 25 steps.
- **Correction:** looking at a composite of GIF frames, I first called the failures "near misses". Checking against the states showed I had misread the panel layout: ep334 ends 166 px / 148 px away (agent / block). The recorded per-pair errors are the reliable source.
- **Not done:** any tuning on the 50 eval pairs. They are fixed, and tuning on them would be test-set tuning.

### 2026-11-24 — Phase 4b-0: simulator investigation (stop for owner review)
- **BN check** on `pusht_scaleN/last.pt`: eval mode gives *lower* pred_loss than batch stats (train windows 0.0115 vs 0.0158; val 0.0305 vs 0.0391). Eval-mode SIGReg is higher (1.46 vs 1.07) only because batch-stat BN re-standardizes each batch. No recalibration needed.
- **Install:** `uv add stable-worldmodel`, then fix two problems:
  1. torchvision from PyPI broke the import (`torchvision::nms does not exist`). Fixed by making torchvision a direct dependency from the cu128 index (`0.26.0+cu128`).
  2. The env's runtime deps were missing. Added `opencv-python-headless`, `pygame`, `pymunk`, `shapely` and `imageio`; skipped the heavy `[env]` extra.
  - torch is unchanged (2.11.0+cu128), and `swm/PushT-v1` creates fine. Only the env class was read (reset/step/eval_state/render source), not the library's models, planners or data code.
- **Env facts:**
  - Observation `state` has the dataset's 7-d layout.
  - `reset(options={"state", "goal_state"})` sets an exact start and goal (`_set_state` takes one 0.01 s physics step).
  - The action space is declared as `Box(-1, 1)` but is **not enforced**: `step` doesn't clip.
  - Relative control: PD target = agent pos + 100 × action, with 10 physics substeps per env step.
  - Success (`eval_state`): ‖goal[:4] − cur[:4]‖ < 20 (agent + block xy together) and angle diff < π/9 (wrap-aware). `step` returns `terminated = success`.
  - Render: 224×224×3 uint8; the green target is at a fixed `goal_pose`.
- **Match checks** (8 random val windows, `outputs/eval/sim_check/sim_vs_data.png`):
  - Reset render vs dataset frame: mean |diff| 0.34–0.52 (of 255), only along shape edges.
  - Replaying 25 dataset actions: agent 0.3–1.1 px, block 0.0–2.3 px, angle ≤ 0.017 rad from `state[t+25]`. **Oracle success 8/8.**
  - Encoder: ‖z_sim − z_data‖ median 0.89, vs 19.2 between a dataset frame and random frames.
- **Owner decision (resolved):** clip CEM actions to [−1, 1] (PLAN default). Context: PLAN's default clips CEM actions to the env's action space, `[-1, 1]`. But the env doesn't enforce it, and the expert data goes beyond it (−1.5 … 2.0; std 0.21). Only 0.12% of dataset rows have a component outside [−1, 1].

### 2026-11-24 — Phase 4a: ×N retrain and probe results
- **Retrain:** 10 epochs, stride 5, ×N, no errors.
  - Effective rank of z: 31 after epoch 0, **64 at the end** (probe frames: 56.7), against ≈3 without ×N.
  - Val pred_loss fell every epoch: 0.136 → 0.031.
  - Eval-mode val SIGReg stays above train (2.5 vs ≈1.0), so the running-stat lag hasn't fully gone. Worth rechecking before planning.
- **Probes, random frame split, matching Table 1:**

  | | Ours, linear / MLP r | Paper, linear / MLP r |
  |---|---|---|
  | Agent | 0.987 / 0.999 | 0.974 / 0.998 |
  | Block | 0.982 / 0.999 | 0.986 / 0.999 |
  | Angle | 0.880 / 0.984 | 0.902 / 0.990 |

- **Probes, group split, lower:** agent 0.964 / 0.988, block 0.915 / 0.929, angle 0.593 / 0.467.
- **Why the group split is lower for block and angle:**
  - It isn't the angle wrap. 17% of frames are within 0.5 rad of 0/2π, but (cos, sin) labels score just as low (0.44 / 0.49), and frames away from the wrap only reach 0.64.
  - It's coverage. Each group shares one starting block pose, so 14 groups cover only part of the block-pose space and the probe must extrapolate to 5 unseen ones.
  - Under the random split the same embedding gives 0.98 / 0.999, so the information is present. By contrast, the collapsed run could only reach linear r 0.82 on the agent even when scored on its own training frames.
- **Owner decision:** `probe.py` reports both splits.
- **Process slip:** a background wait-then-probe job never ran, because its `pgrep -f` pattern matched the job's own command line. I killed it and ran the probes directly.

### 2026-11-24 — Phase 4a: probes revealed representational collapse; retrain with SIGReg ×N
- **What happened:**
  - By its own metrics, training looked correct: no NaNs, val pred 0.0024 ≈ train, emb_std 0.79, SIGReg falling.
  - The first probes (`src/lewm/eval/probe.py`) on `outputs/runs/pusht/last.pt` were far below Table 1. Linear r: agent 0.587, block 0.641, angle 0.595. The MLP was *worse* than linear (0.45–0.55).
- **Diagnosis:**
  - The probe was ruled out: linear r was only 0.82 even when fit and scored on its own training frames; nearest neighbours in z were closer in agent position than random (56 vs 186 px); and a random-frame split gave the same r.
  - Looking at **effective rank** (exp-entropy of the covariance eigenvalues) of z: **1.1 → 2.6 → 3.0 of 192** at epochs 0 / 4 / 9, with ~100% of the variance in 5 dims. The pre-projector CLS was equally poor (r ≈ 0.57–0.67).
  - So this is **representational (dimensional) collapse**. `emb_std` (per-dim std, averaged) missed it, because all 192 dims were mixtures of ~3 factors.
- **Cause:** SIGReg too weak. Without ×N, it barely penalizes a low-rank embedding whose random projections are roughly Gaussian. It only fell 0.32 → 0.10, against a Gaussian floor of ≈0.008 at B = 128.
- **Ablation evidence:** 1 epoch each, same seed and settings, λ = 0.1.

  | | No ×N | ×N |
  |---|---|---|
  | Effective rank | 1.0 | **25.5** |
  | Agent r (lin/MLP) | 0.17/0.06 | 0.56/0.43 |
  | Block r (lin/MLP) | 0.53/0.54 | **0.87/0.92** |
  | Angle r (lin/MLP) | 0.39/0.46 | 0.38/**0.69** |

  The ×N arm's train pred_loss is higher (0.08 vs 0.006), as expected once the embedding stops being nearly constant. Both arms' eval-mode probes are likely understated after 1 epoch because of BN running-stat lag (×N val SIGReg 4.1 vs train ≈1.9).
- **Fix:**
  - **Owner decision:** SIGReg uses ×N (`sigreg_scale_n: true`).
  - Code: `sigreg(..., scale_n)`, with the default still off so the tests and the paper's formula stay as written. Also new: an `effective_rank` helper, which train.py now logs as `train/emb_rank`, and the probe script prints.
  - Retraining 10 epochs at stride 5 into `outputs/runs/pusht_scaleN`. The old run is kept but superseded.
- **Lesson:** monitor effective rank, not only emb_std. Collapse can be dimensional rather than total.

### 2026-11-24 — Phase 3: full 10-epoch run (Phase 3 done)
- **Run:** `configs/pusht.yaml` at stride 5; 28,240 steps at 4.95 it/s with 0% data wait; peak VRAM 11.21 GiB; no NaNs or errors.
- **Train curve:**
  - pred_loss 0.47 (step 25) → 0.013 (step 500) → about 0.003
  - SIGReg 0.32 → 0.10
  - emb_std 0.23 → 0.79, with no collapse
- **Val per epoch (eval mode):**
  - pred_loss 0.089, 0.331, 0.020, 0.0062, 0.0044, 0.0101, 0.0031, 0.0030, 0.0025, 0.0024
  - SIGReg 0.234 → 0.114
- **Epoch-1 val spike** (0.331): diagnosed on `epoch_1.pt` as eval-mode BatchNorm lag, not overfitting.
  - With batch statistics: val pred 0.0051 vs train 0.0044. In eval mode, train windows were just as bad (0.300).
  - Cause: mostly the encoder projector BN. Its running mean was off by 0.16 batch std and its variance by 12%, because the running stats lag the encoder at peak LR.
- **Final checkpoint has no gap:**
  - eval vs train-mode BN: train windows 0.0018 vs 0.0024; val windows 0.0023 vs 0.0036
  - val ≈ train, so unseen starting states generalize
  - No BN recalibration is needed for Phase 4.

### 2026-11-24 — Phase 3 (in progress): training loop, smoke run, loader fix
- **Code:**
  - `src/lewm/train.py`: AdamW 5e-5 / wd 1e-3, clip 1.0, bf16 autocast, 5% warmup + cosine to 0, wandb (online), per-epoch val loss and checkpoint, `--resume`, `--max_steps`, `--stride`.
  - `configs/pusht.yaml` and `configs/smoke.yaml`.
  - `lewm_loss` now also returns `emb_std`.
  - `wandb` added to the dependencies.
- **Smoke run** (1,024 fixed train windows, 400 steps):
  - Train pred_loss 0.169 → 0.014. emb_std stays at 0.26–0.35, SIGReg ≈ 0.30, and the LR schedule ends at 0.
  - Resume test: interrupted after epoch 19, then `--resume` continued at epoch 20, step 160, with the LR continuing exactly on the cosine.
- **Val emb_std near 0 early on (not a bug):**
  - Val emb_std was 0.001–0.01 for the first ~5 epochs, while train emb_std was 0.28.
  - Root cause: before the projector's BatchNorm, the encoder's CLS output barely varies across images at init (variance ≈ 3e-6). Train-mode BN uses batch stats, which hides this. Eval mode uses running_var, which starts at 1 and decays as 0.9^step, so eval outputs come out ~30× too small (running_var 0.0025 at step 56).
  - It catches up after ~120 steps (val emb_std 0.35 by step 120). Irrelevant for the full run.
- **Loader bottleneck, found and fixed:**
  - The first 50-step check gave 2.22 it/s with ≈50% data wait.
  - Root cause: h5py list indexing of 4 rows on the Blosc-chunked `pixels` (33 windows/s in one process). Reading the contiguous `r0:r0+16` slice and keeping `[::5]` gives identical frames at 909 windows/s. A larger chunk cache made no difference.
  - Measured: DataLoader 2.5 → 15.4 batches/s (12 workers); the GPU-only step on a synthetic batch is 4.47 it/s.
  - After the fix, the 100-step check on full data gives **4.40 it/s, 0% data wait, peak VRAM 11.21 GiB**. The smoke run trains the same (final pred_loss 0.0136 vs 0.0146).
- **Logging bug, fixed:** the it/s and wait timers were reset at every epoch start, so on smoke (8 steps/epoch, logging every 25) they read 12–43 it/s. They now reset only at each logging point.
- **Ctrl-C note:** SIGINT left the process hanging during DataLoader/wandb shutdown; SIGTERM was needed. `last.pt` was intact.
- `uv run pytest -q`: 16 passed.

### 2026-11-24 — Phase 2: SIGReg and loss
- `src/lewm/loss.py`:
  - `sigreg(z)`: input (T, N, D), forced to FP32 with autocast off. It uses 1024 unit directions and the ECF on 17 knots in [0.2, 4], weighted by exp(−t²/2), integrated with the trapezoid rule, then averaged over directions and time steps.
  - `lewm_loss`: `mse(pred, emb[:, 1:]) + 0.1·sigreg(emb)`, with no stop-gradient.
- `tests/test_loss.py`:
  - SIGReg values (N = 1024, full range): N(0, I) 1.08e-3, zeros 0.409, constant 0.778, N(3, I) 1.39. (Half range, before the owner's ×2 change, gave exactly half of these.)
  - The all-zeros value matches the closed-form trapezoid of (1 − e^{−t²/2})²·e^{−t²/2}.
  - The output is FP32 under bf16 autocast.
  - Every parameter gets a finite gradient, and the encoder and adaLN layers get nonzero ones. Predictor attn/MLP and the action encoder correctly get zero gradients at step 0 because of adaLN-Zero gates.
  - Perturbing a3 leaves the loss bit-identical.
- Also added shape comments to `model.py` (no behavior change).
- No problems found. `uv run pytest -q`: 16 passed.

### 2026-11-24 — Phase 1: model
- `src/lewm/model.py`:
  - ViT-Tiny/14 encoder: Conv2d patchify, CLS, 257 learned position embeddings, 12 pre-norm blocks, final LN, CLS → BN projector.
  - Action encoder: MLP 10→192→192.
  - Predictor: 6 adaLN-Zero causal blocks (16×64 heads, MLP 2048, dropout 0.1), 3 learned positions, final LN → projector.
  - `LeWM.encode` normalizes uint8 frames on device; `forward` returns emb (B,4,192) and pred (B,3,192). a3 is not fed to the predictor.
- Parameter counts (`uv run python -m lewm.model`): encoder 6.29M (5.49M ViT + 0.79M projector; the paper's "~5M" is the ViT), predictor 11.60M (incl. 0.79M projector; paper "~10M"), action encoder 0.04M.
- Earlier action-block and data questions settled (2026-11-24): the action file has no NaNs (0 of 2,336,736 rows, including last rows), so no fix is needed. `group = episode // 101` was re-verified directly (identical first state within each block, min 14.3 between blocks, 185 unique). a3 is kept in samples per App. D ("4 frames and 4 blocks of 5 actions") but is unused by the loss.
- Causality test bugs (in the test, not the model):
  1. Perturbing with a constant +1.0 was erased by the final LayerNorm. Switched to random perturbations.
  2. The "position 2 changes" cutoff of 1e-2 was arbitrary; the real change at init is ~3e-3. Lowered it to 1e-4, still well above float noise (~1e-7). Checked by hand that a non-causal predictor leaks ~6e-4 into positions 0–1, which `assert_close` rejects.
- `uv run pytest -q`: 12 passed.

### 2026-11-23 — Stage 0.3: split, window index, Dataset (Phase 0 done)
- `src/lewm/data/pusht.py`: `make_group_split`, `build_window_index(stride)`, and `PushTWindows`, which returns uint8 frames (4,3,224,224), raw actions (4,10), state (4,7) and `r0`. The h5 handle is opened lazily per worker.
- Window counts: stride 1 gives train 1,772,146 / val 209,575; stride 5 gives train 361,479 / val 42,723. The totals match the inspection numbers.
- Grid `outputs/windows/windows_grid.png` (3 train and 3 val windows): the agent xy from `state` (scaled 224/512) lands on the blue agent in every frame, and frames advance 5 steps with consistent motion.
- `tests/test_pusht_data.py`:
  - split sizes, disjointness and determinism, plus a match with the committed JSON
  - the stride-1 total = 1,981,721, no window crosses an episode, and no val episode appears in train
  - element-wise frame/action/state equality against direct h5 reads for the first, last and a middle window in both splits
- No problems found. `uv run pytest -q`: 9 passed.

### 2026-11-23 — Owner decisions after stage 0.2
- Committed stages 0.1–0.2 (`7e96193`). That commit left out `src/lewm/data/` because the unanchored `.gitignore` pattern `data/` also matched it. Fixed by anchoring the patterns to the repo root (`/data/`, `/outputs/`, …) and committing the missing package (`178a5a9`). `git check-ignore` confirms `data/*.h5` and `outputs/` are still ignored.
- Decisions recorded above:
  - 0b → stable-worldmodel `swm/PushT-v1`, env only
  - split by starting-state group
  - dataset size recorded as a deviation
  - decision 9 left open until stage 0.3
- Read only the stable-worldmodel data and env documentation (PyPI page, README, dataset-API and PushT env pages); no source code was read. Facts from the env docs to verify after installation:
  - action space `Box(-1, 1, (2,))`, whereas dataset actions reach −1.50 and +2.04 (relevant to decision 5)
  - success = agent+block position error < 20 px and angle error < π/9 (decision 7)
  - no documented reset-to-exact-state API
  - default render size 224×224

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
