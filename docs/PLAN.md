# LeWM on PushT — implementation plan

## 1. Goal and scope

Reimplement **LeWorldModel (LeWM)** from the paper, train it on PushT using one NVIDIA RTX 5070 Ti (16 GB VRAM), and evaluate three capabilities:

1. Latent-space planning success using CEM/MPC.
2. Linear and MLP probing of physical state information.
3. Violation-of-expectation surprise detection.

The first priority is a correct, testable implementation and one complete paper-faithful 10-epoch training run. The paper's three-seed result is the full replication target; running all three seeds is a stretch goal if time and compute permit.

## 2. Source policy and labels

The original paper is the primary methodological source. Place it at `references/LeWorldModel.pdf` in the project repository.

This plan deliberately records all defaults borrowed from the official implementation so the coding agent does not need to inspect that implementation. **The agent must not access the official implementation or any fork, mirror, or derivative repository.** See `CLAUDE.md` for the strict restriction.

The following labels distinguish provenance:

- **[PAPER]** — explicitly stated in the paper.
- **[OFFICIAL-REFERENCE]** — used in the official implementation, adopted only because the paper does not specify the detail. This value has been transcribed into this plan for the agent; it is not permission to inspect the original code.
- **[PROJECT DECISION]** — a practical choice for this reimplementation where neither the paper nor the supplied reference settings give a complete specification. Record its rationale in `docs/PROGRESS.md`.

If a paper-defined setting conflicts with an official-reference setting, use the paper-defined setting. In particular, this project uses 10 epochs, λ = 0.1, and the paper's SIGReg interval [0.2, 4]. Do not substitute the official implementation's 100-epoch config, λ = 0.09, or interval [0, 3].

## 3. Paper specification and implementation settings

### 3.1 PushT data and sequence construction

- **[PAPER] Dataset:** the PushT expert dataset used by the paper, following the setup/dataset of Zhou et al. [18]. The paper describes 20,000 expert episodes with an average trajectory length of 196 environment steps (Appendix E, p. 18).
- **[OWNER DECISION] Dataset source:** the Hugging Face dataset repo `quentinll/lewm-pusht`, file `pusht_expert_train.h5.zst` (see Section 4, decision 0a). It ships only a "train" file, so the validation episodes must be carved out of it by the episode-level split below. Phase 0 checks that the episode count and mean length match the paper's 20,000 / 196.
- **[PAPER] Frame skip:** 5 environment actions are grouped into one action block (Appendix D, p. 17).
- **[PAPER] Training sub-trajectory:** 4 frames and 4 action blocks of 5 actions each (Appendix D, p. 17).
- **[PAPER] Image resolution:** 224 × 224 (Appendix D, p. 17).
- **[PAPER] Batch size:** 128 (Appendix D, p. 17).
- **[PAPER] Training duration:** 10 epochs (Appendix E, p. 18).
- **[PAPER] PushT predictor history length:** 3 (Appendix D, p. 18).
- **[PAPER] Prediction target:** one-step-ahead latent prediction with teacher forcing (Section 3.1, p. 5; Algorithm 3, p. 17).
- **[OFFICIAL-REFERENCE] `num_preds`:** 1. This matches the paper's one-step-ahead prediction objective.
- **[OFFICIAL-REFERENCE] Train/validation split:** 90% training (`train_split: 0.9`). The paper does not specify this split ratio. Split by episode before creating windows to avoid the same episode leaking across train and validation; this episode-level splitting is a project choice and must be recorded.
- **[OFFICIAL-REFERENCE] Initial seed:** 3072. Use this for the first reproducible run; choose and record additional distinct seeds for the multi-seed stretch goal.

**Sequence alignment must be explicit and tested.**

- **[PAPER] Action blocks:** action block `a_k` groups the 5 consecutive environment actions *between* frame `o_k` and frame `o_{k+1}` (Appendix D, p. 17). A sample is frames `o0..o3` (spaced 5 environment steps apart) and blocks `a0..a3`, so one sample spans 20 environment steps. Confirm this against the dataset's own indexing/timestamps rather than assuming it.
- **[PAPER] Loss alignment (Algorithm 3, p. 17):** `emb = encoder(obs)` is `(B, 4, D)` and `next_emb = predictor(emb, actions)` is `(B, 4, D)`. The loss compares `emb[:, 1:]` with `next_emb[:, :-1]`. In other words, the predictions made at positions 0–2 (conditioned on `a0..a2`) are matched to targets `z1..z3`. The prediction at position 3 (conditioned on `a3`) has no target and is discarded.
- **Consequence:** under causal masking, block `a3` never affects the loss. This resolves the "fourth block" question. It is part of the paper's sample definition, so keep it in the loader, but it carries no training signal.
- **[PROJECT DECISION] Predictor input length:** the PushT history length is 3, so feed the predictor only `(z0, z1, z2)` with `(a0, a1, a2)` and size its learned positional embeddings for 3 positions. Under causal masking this gives the same loss as Algorithm 3 and matches the 3-frame context used at planning time. Record this decision.
- The predictor uses causal masking, so a prediction at position t cannot depend on later positions. Document tensor shapes and time/action indexing in code and tests.

**Loss reductions [PAPER, Algorithm 3, p. 17]:**

- The prediction loss uses `F.mse_loss`, i.e. a **mean** over batch, time, and embedding dimensions, not the summed squared norm in Eq. 1. The reduction sets the scale of `L_pred` relative to `λ · SIGReg`, so use the mean reduction and do not change it.
- SIGReg is applied to the **encoder embeddings `emb` of all 4 frames** (projector outputs), not to predictor outputs. It is computed step-wise: transpose to `(T, B, D)`, compute SIGReg over the batch at each time step, then average over the 4 steps. Do not flatten time and batch together for SIGReg.

**Epoch definition:** the paper does not say how sample windows are strided. If a window starts at every environment step, an epoch has roughly 20,000 × (196 − 19) ≈ 3.5M windows, which is about 27,700 optimizer steps at batch size 128. If windows start every 5 steps, an epoch is about 5× shorter. "10 epochs" means very different compute in the two cases, so this is open decision 9. Resolve it before Phase 3's time estimate.

**Storage estimate:** 20,000 × 196 ≈ 3.92 million frames. If every frame were stored as uncompressed 224 × 224 RGB uint8, that would be approximately 590 GB (decimal). Inspect the actual source resolution, format, compression, and dataset size before choosing storage. Prefer on-the-fly decoding/resizing or a compressed cache over storing this naive raw representation.

### 3.2 Encoder and projector

- **[PAPER] Encoder:** ViT-Tiny, approximately 5M parameters; patch size 14, 12 transformer layers, 3 attention heads, hidden dimension 192; input 224 × 224 (Section 3.1, p. 4; Appendix D, p. 17).
- **[PAPER] Encoder source:** Appendix D (p. 17) says the encoder is the Hugging Face ViT-Tiny model with patch size 14. This project does not use that class (see `CLAUDE.md`). Instead, rebuild the same standard architecture from scratch.
- **[PROJECT DECISION] Standard ViT-Tiny details the paper does not list:**
  - pre-norm transformer blocks
  - GELU MLP with hidden width 768 (ratio 4)
  - learned positional embeddings for 1 CLS token + 256 patch tokens (224/14 = 16 patches per side)
  - a final LayerNorm before taking the CLS token

  Record these, and report the resulting parameter count against the paper's ~5M.
- **[PAPER] Representation:** take the final-layer CLS token and pass it through a projector (Section 3.1, p. 4).
- **[PAPER] Projector normalization:** Batch Normalization is used because the final ViT LayerNorm otherwise prevents the anti-collapse objective from being optimized effectively (Section 3.1, p. 4).
- **[OFFICIAL-REFERENCE] Projector hidden dimension:** 2048. Use an MLP projector with a 192-dimensional input and output and one hidden layer of width 2048, with BatchNorm1d. The official configuration uses GELU in the MLP. Treat this as the explicit fallback where the paper does not give the hidden width/activation.
- **[PAPER] Training mode:** the encoder is trained end-to-end from pixels; do not load pretrained weights, freeze the encoder, or introduce an EMA target encoder.

The predictor also ends with a projector using the same implementation as the encoder projector (Section 3.1, p. 5).

### 3.3 Action encoder and predictor

- **[PAPER] Predictor:** six transformer layers, 16 attention heads, 10% dropout, approximately 10M parameters (Section 3.1, p. 4).
- **[PAPER] Action conditioning:** condition each predictor layer through Adaptive Layer Normalization (AdaLN); initialize the AdaLN modulation parameters to zero (Section 3.1, p. 4). The paper gives the purpose: "ensure that action conditioning impacts the predictor training progressively".
- **Ambiguity:** the paper does not say which form of AdaLN it uses:
  - **Shift/scale only:** zero-initialized shift and scale, so at initialization the predictor is an ordinary transformer that ignores actions.
  - **DiT-style adaLN-Zero:** also includes zero-initialized residual gates, so at initialization every residual branch is switched off and each block is the identity.

  These give different networks at initialization. Present both and ask before implementing (open decision 10).
- **[PAPER] Temporal structure:** causal temporal masking; the predictor autoregressively predicts future embeddings from a history of frame representations (Section 3.1, pp. 4–5).
- **[PAPER] PushT history length:** 3 (Appendix D, p. 18).
- **[PAPER] Predictor backbone description:** ViT-S with learned positional embeddings and causal masking (Appendix D, p. 18).
- **[OFFICIAL-REFERENCE] Predictor dimensions:** input dimension 192, hidden dimension 192, output dimension 192, MLP dimension 2048, 16 heads with per-head dimension 64, embedding dropout 0.0, and dropout 0.1. The attention projection dimension implied by 16 × 64 is 1024. These dimensions are transcribed from the supplied official model configuration because the paper does not list all of them.
- **[OFFICIAL-REFERENCE] Action encoder method:** use a Conv1D-based action-block embedder followed by an MLP, producing a 192-dimensional conditioning embedding. The action-block input dimension is `frameskip × action_dim` (therefore `5 × action_dim` after grouping). The paper specifies action conditioning through AdaLN but does not describe the action encoder architecture; this high-level Conv1D + MLP method is adopted from the official implementation. Any internal Conv1D channel/kernel choices not stated here must be proposed and recorded as project decisions; do not look them up in the original repository.
- **[OFFICIAL-REFERENCE] No mask token:** do not add a ViT mask token; the supplied model configuration has `use_mask_token: false`.

The official configuration labels the predictor backbone ViT-S while its explicit embedding/hidden dimensions are 192. Implement the numeric dimensions above rather than guessing dimensions from the backbone name. Verify the resulting parameter count and report any difference from the paper's approximate 10M predictor count.

### 3.4 Training objective

The LeWM objective has exactly two terms:

`L = L_pred + λ × SIGReg(Z)`

- **[PAPER] Prediction loss:** mean-squared error between each predicted next embedding and the corresponding encoded next-frame embedding, using teacher forcing (Section 3.1, p. 5; Algorithm 3, p. 17).
- **[PAPER] SIGReg:** Sketched-Isotropic-Gaussian Regularizer based on the Cramér–Wold idea and the univariate Epps–Pulley normality statistic (Section 3.1, p. 5; Appendix A, p. 13).
- **[PAPER] Number of random directions:** `M = 1024`, sampled as unit-norm directions uniformly on the sphere (Section 3.1, p. 5; Appendix A, p. 13).
- **[PAPER] SIGReg integration interval:** trapezoidal quadrature with knots uniformly distributed over `[0.2, 4]` (Appendix A, p. 13).
- **[OFFICIAL-REFERENCE] Number of integration knots:** `T = 17`. The paper specifies the interval and quadrature but not the knot count; 17 is used in the official implementation and is adopted here for that reason. 17 is also one of the knot counts in the paper's own ablation (Figure 15, right, p. 24), which reports performance as largely insensitive to it.
- **[PAPER] Epps–Pulley statistic (Appendix A, p. 13):** `T(h) = ∫ w(t) |φ_N(t; h) − φ0(t)|² dt`. Here `φ_N(t; h) = (1/N) Σ_n exp(i t h_n)` is the empirical characteristic function over the batch, `φ0(t) = exp(−t²/2)` is the standard normal's characteristic function, and `w(t)` is a Gaussian weighting function. SIGReg averages `T` over the M directions.
- **Scale details the paper leaves open (open decision 3):**
  - The width of the weighting function `w`.
  - The paper's formula has no sample-size factor N. Many Epps–Pulley implementations multiply by N; this one should not.
  - The integrand is even in t and the knots cover only `[0.2, 4]`, so a factor of 2 may or may not be included.

  Each of these rescales SIGReg and therefore changes the effective λ. Follow the paper's formula literally where it is explicit. Record each remaining choice, and log SIGReg's value on samples from N(0, I) so the scale can be checked.
- **[PAPER] Loss coefficient:** `λ = 0.1` (Section 3.1, p. 5; Algorithm 3, p. 17). This project intentionally follows the paper's value, not the official implementation's 0.09.
- **[PAPER] Step-wise SIGReg:** apply SIGReg independently at each time step across the batch, then average the per-step SIGReg values (Algorithm 3, p. 17). Do not flatten batch and time into a single sample axis.
- **[PAPER] No extra stabilization:** no stop-gradient, EMA, or additional stabilization heuristics; gradients flow through the objective and components are optimized jointly (Section 3.1, p. 5).
- **[PAPER] No reconstruction loss:** the visualization decoder described in Appendix D is diagnostic only and is not part of the LeWM training objective. Do not implement a reconstruction objective for the core project.

For numerical stability, run SIGReg's characteristic-function/statistical calculations and sensitive reductions in FP32 even when the model uses BF16 autocast. This is an engineering precision choice, not a change to the mathematical objective.

### 3.5 Training configuration

Use this as the initial paper-faithful PushT training configuration:

| Setting | Value | Provenance |
|---|---:|---|
| Epochs | **10** | [PAPER], Appendix E, p. 18 |
| Batch size | **128** | [PAPER], Appendix D, p. 17 |
| Image size | **224 × 224** | [PAPER], Appendix D, p. 17 |
| Sub-trajectory | **4 frames, 4 action blocks** | [PAPER], Appendix D, p. 17 |
| Actions per block | **5** | [PAPER], Appendix D, p. 17 |
| Predictor history | **3** | [PAPER], Appendix D, p. 18 |
| Prediction offset (`num_preds`) | **1 step** | [PAPER] next-step objective; value/name [OFFICIAL-REFERENCE] |
| SIGReg weight | **0.1** | [PAPER], Section 3.1, p. 5 |
| SIGReg projections | **1024** | [PAPER], Section 3.1, p. 5 |
| SIGReg integration interval | **[0.2, 4]** | [PAPER], Appendix A, p. 13 |
| SIGReg knots | **17** | [OFFICIAL-REFERENCE], paper does not specify |
| Optimizer | AdamW | [OFFICIAL-REFERENCE], paper does not specify |
| Learning rate | `5e-5` | [OFFICIAL-REFERENCE], paper does not specify |
| Weight decay | `1e-3` | [OFFICIAL-REFERENCE], paper does not specify |
| Scheduler | Linear warmup + cosine annealing | [OFFICIAL-REFERENCE], paper does not specify |
| Gradient clipping | `1.0` | [OFFICIAL-REFERENCE], paper does not specify |
| Precision | BF16 autocast | [OFFICIAL-REFERENCE], paper does not specify |
| Train split | `0.9` (90% train) | [OFFICIAL-REFERENCE], paper does not specify |
| Initial seed | `3072` | [OFFICIAL-REFERENCE], paper does not specify |
| DataLoader workers | `6` | [OFFICIAL-REFERENCE], runtime setting |
| Persistent workers | `true` | [OFFICIAL-REFERENCE], runtime setting |
| Prefetch factor | `3` | [OFFICIAL-REFERENCE], runtime setting |
| Pin memory | `true` | [OFFICIAL-REFERENCE], runtime setting |

The official training configuration specifies 100 epochs, but this project intentionally uses **10 epochs** because Appendix E says the PushT models are trained for 10 epochs. Do not copy the 100-epoch setting.

The supplied reference configuration does not specify all scheduler arguments (for example, warmup duration and minimum learning rate). Do not inspect the official repository to find them. Before full training, propose explicit values, explain their impact, and record the approved choice in `docs/PROGRESS.md`.

### 3.6 CEM planning and MPC

- **[PAPER] Cost:** squared L2 distance between the final predicted embedding and the goal embedding (Section 3.2, p. 6).
- **[PAPER] CEM initial distribution:** mean `μ₀ = 0`, covariance `Σ₀ = I` (Appendix B, p. 13; Algorithm 2, p. 14).
- **[PAPER] Candidate action sequences:** 300 per CEM iteration (Appendix B, p. 13; Appendix D, p. 18).
- **[PAPER] Elite sequences:** retain the 30 candidates with lowest cost per iteration (Appendix B, p. 13; Appendix D, p. 18).
- **[PAPER] Iterations:** 30 for PushT (Appendix B, p. 13; Appendix D, p. 18).
- **[PAPER] Planning horizon:** 5 action blocks, equivalent to 25 environment steps with frame skip 5 (Appendix D, p. 18).
- **[PAPER] Distribution update:** update mean and covariance using the elite set; use the final sampling-distribution mean as the final plan (Appendix B, pp. 13–14).
- **[PAPER, interpreted] Diagonal covariance:** Algorithm 2 (p. 14) updates `Σ_t ← Var_{i∈E}(a_{1:H})`, the per-element variance of the elites. Read this as a diagonal Gaussian over the flattened `(H × 5 × action_dim)` action sequence. Record this interpretation.
- **[PAPER] MPC execution for this PushT setup:** execute the full 5-block optimized sequence, then replan from the updated observation (Appendix D, p. 18; this is `K = H = 5`).
- **[PROJECT DECISION]** The paper does not spell out action normalization or clipping/bounds. Inspect the PushT action space, propose a safe and explicit choice for the CEM implementation, and record it. Note that `μ₀ = 0, Σ₀ = I` only makes sense in normalized action units. Do not silently clip samples in a way that changes the distribution without documenting it.

Keep a rolling context of the three most recent observations/latent embeddings required by the PushT predictor. At each replanning point, initialize the predictor context from actual observed frames, encode the goal observation, and roll out candidate action blocks autoregressively.

### 3.7 Evaluation protocol and targets

**Planning evaluation [PAPER]:**

- Sample the initial state from an offline dataset trajectory.
- Sample the goal state 25 environment steps after the initial state in that same trajectory.
- Give the controller a budget of 50 environment steps.
- Evaluate on the same set of 50 trajectories when comparing across training seeds (Appendix F.1, p. 19; Table 5, p. 24).
- **[PROJECT DECISION] Fixed evaluation set:** the paper does not say which episodes the 50 start/goal pairs come from. Sample them once, with a fixed seed, from **validation** episodes. Save the list of `(episode, start step)` pairs to a versioned file (e.g. `configs/eval/pusht_eval_50.json`) and reuse it for every checkpoint and seed. Using held-out episodes may make the task slightly harder than the paper's setup if the paper sampled from training data; note this when comparing.
- Report success rate and save representative successful and failed rollouts.
- The paper reports PushT LeWM success of **96.0 ± 2.83%** over three training seeds; DINO-WM is 92.0 and PLDM is 78.0 in the paper's comparison (Table 5, Appendix G, p. 24 and Figure 6, p. 7).
- The paper reports approximately **0.98 seconds** for full planning in its planning-time comparison (Figure 3, p. 3). Treat this as a reference result, not a guaranteed hardware-independent acceptance threshold. Measure and report this machine's actual latency.

**Probing [PAPER]:**

- Freeze the trained encoder and extract embeddings.
- Train both linear and MLP probes.
- Predict agent location, block location, and block angle.
- Report MSE and Pearson correlation `r` (Section 5.1 and Table 1, p. 8).
- Paper's LeWM linear-probe correlation reference values:

  | Quantity | Pearson r |
  |---|---:|
  | Agent location | 0.974 |
  | Block location | 0.986 |
  | Block angle | 0.902 |

- Keep train/validation/test separation at the episode level so adjacent frames from one trajectory do not leak across splits. This is a project evaluation safeguard; document the precise split because the paper does not fully specify it in the main text.

**Violation-of-expectation / surprise [PAPER]:**

- Quantify surprise using next-embedding prediction error (MSE).
- For PushT, use a random policy biased toward interacting with the block (Appendix F.3, p. 22).
- Generate three trajectory conditions: (1) unperturbed; (2) abrupt block color change during the episode; (3) both agent and block teleported to random positions at the perturbation timestep (Appendix F.3, p. 22).
- **[PROJECT DECISION]** For PushT the paper does not give the perturbation timestep, the episode length, or how the random policy is biased toward the block. Choose these, record them, and use the same random seeds across the three conditions so the trajectories are paired.
- Plot prediction error over environment time and identify the perturbation timestep.
- The paper reports a significant increase for physical teleportation (paired t-test, `p < 0.01`) and a weaker, non-significant increase for color changes (Figure 8, p. 9). Treat these as comparison targets, not values to force through analysis choices.

## 4. Open implementation decisions

The paper and the owner-supplied reference configuration do not settle every low-level detail. Resolve these deliberately and record the choice and rationale in `docs/PROGRESS.md` before the relevant full run:

0. **PushT dataset and simulator source.**
   - **0a. Dataset: DECIDED by the owner.** Use the Hugging Face **dataset** repo `quentinll/lewm-pusht`, the authors' official PushT dataset release (MIT license).
     - It holds a single file, `pusht_expert_train.h5.zst`: about 13.1 GB, a zstd-compressed HDF5 file.
     - The repo page does not document the keys, image resolution, or episode count. Phase 0 must find these out by inspecting the file.
     - **Warning:** a *model* repo with the same name (`quentinll/lewm-pusht`) holds the official pretrained checkpoint. Always download with the dataset repo type, e.g. `--repo-type dataset`, and never download or load the model repo.
   - **0b. Evaluation simulator: OPEN. Owner decision needed before Phase 6** (and before the Phase 0 rendering comparison). The simulator must render frames that match this dataset. The options are:
     - **(i)** The PushT environment from the LeWM authors' benchmark library. This most likely matches the dataset, but it ships alongside the LeWM code, so it falls under the data/environment-only exception in `CLAUDE.md`.
     - **(ii)** Another PushT implementation, such as DINO-WM's environment. You would have to check that its rendering, action scaling, and state layout match the dataset.

     Record the choice here: `Decision 0b: ________`. Claude must not pick this itself.
1. Exact scheduler arguments: warmup duration, minimum learning rate, and whether the schedule is parameterized by epochs or optimizer steps.
2. Exact Conv1D action-encoder channel widths, kernel size, padding, and MLP details. The adopted high-level method is Conv1D + MLP; do not look up the official code for missing details.
3. Epps–Pulley scale details: the width of the weighting function, whether a factor of 2 is included for integrating only over positive t, and confirmation that no sample-size factor N is applied (see Section 3.4).
4. Whether SIGReg random directions are regenerated per call or managed with a persistent seeded sampler; make the choice reproducible.
5. Action normalization and CEM action bounds/clipping for the PushT environment.
6. CEM covariance: diagonal, per Algorithm 2's per-element variance (Section 3.6). Only revisit with owner approval.
7. Exact environment success predicate and any simulator-specific details needed to reproduce the PushT evaluation.
8. Exact probe MLP architecture and probe optimization settings, which are not specified in the main paper description.
9. Window stride for sampling training windows (every environment step or every 5). This defines what "10 epochs" costs (Section 3.1).
10. AdaLN form: shift/scale only, or DiT-style adaLN-Zero with residual gates (Section 3.3).
11. Source split and fixed file for the 50 evaluation start/goal pairs (Section 3.7).
12. Surprise-experiment details: perturbation timestep, episode length, and how the random policy is biased toward the block (Section 3.7).

Do not turn these into broad hyperparameter searches. Choose the simplest defensible implementation, record it, and keep paper-defined values fixed.

## 5. Phased implementation plan

### Phase 0 — Environment, data, and evaluation simulator

**Tasks**

- The dataset source is decided (decision 0a). The simulator source (decision 0b) is still open: ask the owner before installing any simulator.
- Create the repository structure, Python environment, dependencies, and CUDA-enabled PyTorch installation.
- **Download the dataset:** get `pusht_expert_train.h5.zst` from the **dataset** repo `quentinll/lewm-pusht` (repo type `dataset`, never the model repo of the same name) into a gitignored `data/` directory.
- **Check disk space before decompressing:** check free disk space, and get the decompressed size from the zstd frame header if it is recorded there. Report both to the owner before decompressing, because a raw-pixel HDF5 file could be many times larger than 13.1 GB. If the decompressed file won't fit, stop and present the options.
- **Inspect the HDF5 file:**
  - its groups, keys, shapes, and dtypes
  - its chunking and internal compression
  - how it marks episode boundaries
  - what proprioception and state fields it contains
  - its image resolution

  Write this up as a short data-format note in `docs/PROGRESS.md`.
- Verify the RTX 5070 Ti is visible to PyTorch and run a minimal CUDA/BF16 smoke test.
- Place the paper at `references/LeWorldModel.pdf`.
- Download the PushT expert dataset and inspect its actual format, source image resolution, action shape, episode boundaries, timestamps, and metadata/labels.
- Implement a dataset inspection tool that reports episode count, trajectory lengths, image dimensions, action dimensions, and storage size.
- Implement an episode-level 90/10 train/validation split using the [OFFICIAL-REFERENCE] ratio, then construct 4-frame/4-action-block samples with frameskip 5.
- Verify temporal alignment between frames, action blocks, and state labels. Save visualized sample trajectories and tensor-shape reports.
- Set up the PushT simulator for evaluation. Compare a simulator-rendered frame with the dataset frame for the same environment state; investigate rendering differences rather than assuming exact pixel identity.
- Resolve the open settings needed before training (Section 4, decisions 1–5, 9, 10): scheduler arguments, action encoder internals, SIGReg numerical details, SIGReg direction sampling, action normalization, window stride, AdaLN form, and data preprocessing.
- Create initial configs: `configs/smoke.yaml` and `configs/pusht.yaml`. Keep the full config paper-faithful.

**Done when**

- PyTorch sees the target GPU and the CUDA/BF16 smoke test passes.
- Dataset inspection output is saved and its format is understood.
- Dataset tests pass for shapes, dtypes, episode boundaries, frame/action alignment, and label alignment.
- A saved image grid of sampled trajectories looks correct.
- Loader throughput is measured and is adequate for training; bottlenecks are documented.
- Simulator and dataset observations for matched states are visually compared, with any renderer differences documented.
- All open decisions needed for Phases 1–4 are recorded in `docs/PROGRESS.md`.

### Phase 1 — Encoder, projectors, action encoder, and predictor

**Tasks**

- Implement ViT-Tiny from scratch using the paper's patch size, depth, heads, and width.
- Implement the CLS-token projection head with BatchNorm1d and the adopted 2048 hidden width.
- Implement the Conv1D + MLP action-block embedder with output dimension 192; document chosen internal dimensions.
- Implement the six-layer causal predictor with the paper's head count, dropout, AdaLN conditioning, zero initialization, and learned positional embeddings.
- Use the adopted predictor dimensions from Section 3.3 of this plan.
- Implement the predictor output projector matching the encoder projector.
- Add shape, parameter-count, initialization, and causal-leakage tests.

**Done when**

- Encoder output shape is `(B, T, 192)`.
- Predictor output is correctly aligned for one-step-ahead targets.
- The encoder and predictor parameter counts are reported and compared with the paper's approximate 5M and 10M targets; explain any material difference.
- Changing a future input token cannot change predictions at earlier positions (causality test).
- AdaLN modulation parameters are exactly zero at initialization.
- An initialization test confirms the action-conditioned modulation path starts at its zero-initialized state.
- Projector tests verify BatchNorm behaviour, output dimension, and train/eval mode handling.

### Phase 2 — SIGReg

**Tasks**

- Implement random unit-norm directions sampled uniformly on the unit sphere.
- Implement projection of batch embeddings onto `M = 1024` directions.
- Implement the univariate Epps–Pulley statistic and trapezoidal quadrature over 17 evenly spaced knots in `[0.2, 4]`.
- Implement step-wise SIGReg over the batch dimension and average over time.
- Keep sensitive statistical operations in FP32 and preserve gradient flow to embeddings.

**Done when**

- Unit tests show a low statistic for samples from an approximately standard normal distribution and larger statistics for collapsed, low-rank, shifted, and badly scaled distributions.
- Tests verify projected directions have unit norm and the SIGReg result is finite.
- Gradients flow to the input embeddings and are finite.
- A toy optimization experiment demonstrates movement toward an isotropic Gaussian-like distribution; save plots of the initial and final distributions and report mean/std/effective rank. Do not claim exact Gaussian convergence from a small toy run.
- The implementation's interval, knot count, projection count, and normalization are printed in the run configuration/log.

### Phase 3 — Training loop and smoke training

**Tasks**

- Implement the two-term LeWM objective with teacher forcing and no stop-gradient/EMA.
- Implement the paper's shifted next-embedding loss and step-wise SIGReg aggregation.
- Implement the adopted AdamW optimizer, learning rate, weight decay, scheduler, gradient clipping, and BF16 autocast.
- Implement training/validation loops, checkpointing, resume, deterministic seeding, and experiment logging.
- Log prediction loss, SIGReg loss, total loss, per-dimension embedding standard deviation, effective rank, learning rate, throughput, and VRAM usage.
- Implement a small smoke configuration that is clearly separate from the full 10-epoch config.

**Done when**

- A short smoke run completes and tests the complete data → encoder → predictor → losses → backward → optimizer path.
- Loss-alignment tests pass:
  - the prediction loss compares predictions from positions 0–2 with targets `z1..z3`
  - changing action block `a3` does not change the loss
  - `L_pred` uses the mean reduction
  - SIGReg is computed on the encoder embeddings of all 4 frames, step by step
- A tiny fixed subset can be overfit as a debugging check; report prediction loss and representation statistics rather than weakening SIGReg or acceptance criteria to force a particular outcome.
- No NaN/Inf appears in embeddings, losses, or gradients.
- Checkpoint resume restores the training state and produces a consistent continuation under deterministic settings.
- Peak VRAM and step/epoch time are measured at the intended batch size of 128.
- Estimated time for the full 10-epoch run is recorded before Phase 4 begins.

### Phase 4 — Paper-faithful full training

**Tasks**

- Train on PushT for exactly 10 epochs using the configuration in Section 3.5.
- Use `λ = 0.1`, `M = 1024`, 17 knots over `[0.2, 4]`, and batch size 128.
- Save checkpoints and export loss/representation-health curves.
- Do not run the official implementation's 100-epoch configuration or substitute λ = 0.09 / interval `[0, 3]`.

**Done when**

- The 10-epoch run completes and the final checkpoint is saved.
- Prediction loss and SIGReg curves are exported.
- Embedding standard deviation and effective rank show no obvious representation collapse; report the actual values/curves rather than relying on a vague “healthy” label.
- Configuration, seed, Git hash, runtime, peak VRAM, and any deviations are recorded.

An optional `λ = 0` ablation may be run after the primary model is complete, but it is not part of the paper-faithful main run and must be labelled as an ablation.

### Phase 5 — Physical-state probing

**Tasks**

- Freeze the trained encoder and extract embeddings for labelled PushT observations.
- Train linear probes and MLP probes for agent location, block location, and block angle.
- Keep splits episode-disjoint and record preprocessing/normalization.
- Compute MSE and Pearson correlation for each target and probe type.

**Done when**

- A results table comparable to Table 1 (paper p. 8) is saved.
- Linear and MLP results are reported separately.
- The comparison with paper targets is discussed, including likely reasons for gaps and any differences in data split or label scaling.

### Phase 6 — CEM planner and MPC evaluation

**Tasks**

- Implement CEM with initial mean zero and covariance identity, 300 candidates, 30 elites, and 30 iterations.
- Use a 5-action-block horizon (25 environment steps) and terminal squared L2 latent-goal cost.
- Implement autoregressive latent rollouts using the three-frame context.
- Execute all five planned action blocks before replanning, as in the PushT setup in Appendix D.
- Implement the 50-step evaluation budget and sample the goal 25 environment steps after the initial state in the same dataset trajectory.
- Create the fixed 50-pair evaluation file (Section 3.7, decision 11) once, and use it for every evaluation.
- Save rollout videos/GIFs for representative successes and failures.

**Done when**

- CEM finds the optimum of a toy quadratic objective within a defined tolerance.
- Planner tests verify action-sequence shapes, elite selection, distribution updates, and deterministic seeded behaviour.
- At least one end-to-end planning rollout runs in the PushT simulator.
- Success over the 50 evaluation trajectories is reported.
- Planning latency is measured and compared with the paper's approximately 0.98-second reference, without assuming identical hardware/runtime.
- The exact success predicate, action clipping/bounds, and any deviations are documented.

### Phase 7 — Violation-of-expectation surprise evaluation

**Tasks**

- Implement prediction-error surprise using next-embedding MSE.
- Generate PushT trajectories using a random policy biased toward interacting with the block.
- Generate clean, block-color-change, and joint agent/block teleport perturbation conditions. Apply the perturbation at a fixed, recorded timestep (decision 12), and use the same seeds across the three conditions so the trajectories are paired.
- Plot surprise over time and mark the perturbation point.
- Perform paired statistical comparisons between perturbed and unperturbed conditions using a clearly documented pairing protocol.

**Done when**

- Surprise-over-time plots are saved for all three conditions.
- Teleportation produces a clear prediction-error spike and statistical results are reported.
- Color change is compared separately; do not force it to be non-significant by changing the analysis.
- The results are compared honestly with Figure 8 and Appendix F.3 of the paper.

### Phase 8 — Additional seeds and final write-up (stretch)

**Tasks**

- Train two additional seeds using the same fixed 50 evaluation trajectories.
- Report mean ± standard deviation across three training seeds for planning success, matching the paper's reporting protocol as closely as practical.
- Consolidate figures, tables, commands, configuration, hardware details, limitations, and all paper/implementation deviations into `README.md`.
- Summarize the project history and important technical lessons from `docs/PROGRESS.md`.

**Done when**

- Three-seed results and variance are reported where compute permits.
- The README clearly distinguishes reproduced results from partial results and project-specific deviations.

## 6. 48-hour prioritization

The 48-hour goal is an engineering target, not a guarantee. Prioritize in this order:

1. **Must complete:** Phase 0; Phases 1–3; one paper-faithful 10-epoch run in Phase 4.
2. **Primary evaluation target:** Phase 6 planning evaluation. If time permits, complete Phases 5 and 7 as well.
3. **Stretch:** Phase 8 three-seed replication and polished write-up.

Do not silently remove a method or weaken an acceptance criterion to meet the time target. If runtime or hardware prevents a phase from completing, report the blocker and the evidence collected.

## 7. Progress tracking

Maintain `docs/PROGRESS.md` as the current status and concise development log. At minimum, keep:

- Current phase and next task.
- Completed acceptance checks with evidence.
- Important implementation decisions and their provenance.
- Bugs/problems encountered, root causes, and verified fixes.
- Experiment configs, seeds, Git hashes, runtime, peak VRAM, and artifact paths.
- Deviations from the paper and why they were necessary.
- Open blockers and prerequisites for the next phase.

Do not record every trivial edit. Record meaningful development events, and never invent or retrospectively guess results.
