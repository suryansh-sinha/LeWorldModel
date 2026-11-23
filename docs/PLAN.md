# LeWM on PushT: plan

**Goal:** implement LeWM, train it on PushT for 10 epochs, and evaluate it with probes and planning (the surprise test is optional).
Running three seeds is a stretch goal.

## 1. Spec

### Data (App. D p. 17; file details are in the PROGRESS.md data-format note)
- **The file:** `data/pusht_expert_train.h5` is flat, with one row per env step:
  - `pixels`: (N, 224, 224, 3) uint8, HWC
  - `action`: (N, 2)
  - `state`: (N, 7)
  - `proprio`: (N, 4)

  Episode `e` covers rows `ep_offset[e] : ep_offset[e] + ep_len[e]`.
  The file has 18,685 episodes averaging 125 steps; the paper says about 20,000 episodes averaging 196. We accept that difference.
- **`action[t]`** is the action applied between frame t and frame t+1 (already verified).
- **A sample** is 4 frames, 5 steps apart, plus 4 action blocks.
  Starting at row r0, frame k is row `r0 + 5k`, and block k is rows `r0 + 5k … r0 + 5k + 4`, flattened to 10 numbers.
  The state label for frame k comes from the same row as the frame.
  The start t0 runs from 0 to L − 20 within an episode, so a window never crosses an episode boundary.
- **Split:** 90/10 by starting-state group (185 groups of 101 episodes; seed 3072, giving 166 train and 19 val groups).
  Save the group lists to `configs/splits/pusht_group_split.json`.
  Everything held out (validation, probe test, evaluation pairs) comes from the validation groups.
- Images are 224 × 224. Batch size is 128, and we train for 10 epochs.

### Encoder (Sec. 3.1 p. 4, App. D p. 17)
- A standard ViT-Tiny: patch size 14, 12 layers, 3 heads, width 192, MLP width 768, pre-norm.
  It has a CLS token, learned position embeddings (257 tokens), and a final LayerNorm.
- The CLS token goes through a projector, then becomes the embedding `z` (192-d).
- **Projector:** Linear(192 → 2048), then BatchNorm1d, then GELU, then Linear(2048 → 192).

### Action encoder
- A small MLP: 10 → 192 → 192, with GELU.
  (A Conv1d with kernel size 1 over the blocks does the same job.)
  The paper doesn't describe the action encoder, so this is our choice.

### Predictor (Sec. 3.1 p. 4, App. D p. 18)
- 6 transformer layers: 16 heads × 64 dims per head, width 192, MLP width 2048, dropout 0.1.
- Causal mask, with learned position embeddings for 3 positions (history length 3).
- Conditioned on actions with adaLN-Zero (DiT-style shift, scale and gate per layer), with the modulation layer zero-initialized.
  The paper only says "AdaLN, initialized to zero"; the adaLN-Zero form is our choice.
- Followed by its own projector, built the same way as the encoder's.

### Loss (Algorithm 3 p. 17)
- Encode all 4 frames to get `emb` of shape (B, 4, 192).
- Feed the predictor `emb[:, :3]` with action blocks 0–2 to get `pred` of shape (B, 3, 192).
- `L_pred = F.mse_loss(pred, emb[:, 1:])`, using the mean reduction. Block `a3` isn't used.
- **SIGReg (App. A p. 13):** for each of the 4 time steps:
  - take the batch of embeddings (B, 192) and project it onto 1024 random unit directions
  - compute the Epps–Pulley statistic: the weighted squared distance between the empirical characteristic function and exp(−t²/2)
  - integrate it with the trapezoid rule on 17 knots over [0.2, 4], using weight exp(−t²/2) (the paper doesn't give the weight's width, so this is our default), with no ×N factor

  Average over the directions and the 4 time steps.
- `L = L_pred + 0.1 · SIGReg`.

### Optimizer
- AdamW with learning rate 5e-5, weight decay 1e-3, and gradient clipping at 1.0. Train in BF16.
- Linear warmup over the first 5% of steps, then cosine decay to 0.

### Planning (App. B pp. 13–14, App. D p. 18)
- CEM with 300 samples, the top 30 kept, and 30 iterations, over a horizon of 5 blocks.
- The sampling distribution starts at mean 0 with diagonal variance 1, in normalized action units.
- The cost is the squared L2 distance between the last predicted embedding and the goal embedding.
- Execute all 5 blocks, then replan.
- **Evaluation:** start and goal come from the same dataset episode, with the goal 25 steps later, and a budget of 50 steps.
  Use 50 fixed (episode, start) pairs from validation groups, saved once to a file. Count success with the environment's own success check.

### Targets
- Planning success: LeWM 96.0 ± 2.8 %, DINO-WM 92, PLDM 78 (Table 5 p. 24). Planning takes about 1 s (Fig. 3 p. 3).
- Linear probe r: agent 0.974, block 0.986, angle 0.902 (Table 1 p. 8).
- Surprise: teleporting gives a clear spike; a color change gives a weak one (Fig. 8 p. 9).

### Defaults where the paper is silent (use these, and log them in PROGRESS.md)
- Images: divide by 255, then normalize with mean 0.5 and std 0.5. Do this on the GPU.
- Actions: standardize with the training set's mean and std.
- SIGReg directions: draw new ones every step.
- Window stride: decided at the start of Phase 3 (see below).
  Stride 1 gives about 15.5k steps per epoch; stride 5 gives about 3.2k.
- CEM: convert sampled actions back to raw units, then clip them to the environment's action space.
- Probe labels: agent `state[0:2]`, block `state[2:4]`, angle `state[4]`.
  Probe the raw angle, and keep in mind that it wraps around at 0/2π.
- Probe MLP: one hidden layer of width 256.

## 2. Phases

### Phase 0: Data
- [x] Environment and GPU check
- [x] Dataset download and inspection (see the PROGRESS.md data-format note)
- [x] Group split, a window index, and a Dataset returning uint8 frames (4, 3, 224, 224), actions (4, 10), and state (4, 7)

**Done when:**
- A test on a few real windows checks the row indexing above for frames, action blocks, and state.
- A saved image grid of a few windows looks right.

### Phase 1: Model
- [x] Build the encoder, projectors, action encoder, and predictor.

**Done when:**
- The shapes are right.
- The parameter counts are printed: about 5M for the encoder and about 10M for the predictor.
- Changing a later input does not change earlier predictions (causality).

### Phase 2: SIGReg and loss
**Done when:**
- SIGReg is low for N(0, I) samples and high for collapsed or shifted ones.
- Gradients flow.
- Changing block a3 doesn't change the loss.

### Phase 3: Training
- Write a training loop with wandb logging (prediction loss, SIGReg, embedding std), a checkpoint every epoch, and resume from the last checkpoint.
- Before the full run, train 50 steps on the full data and read the it/s and peak VRAM.
  Then the owner picks stride 1 or 5 so that 10 epochs fit the time available.

**Done when:**
- The smoke config overfits a small subset without the embedding std collapsing to 0.
- The full 10-epoch run finishes.

### Phase 4: Evaluation
- **4a. Probes:** freeze the encoder, then train linear and MLP probes on the validation-group frames. Report MSE and r.
- **4b. Planning:**
  - Install `swm/PushT-v1` and check one simulator frame against a dataset frame by eye.
  - Settle two known questions:
    - The env's docs say the action space is Box(−1, 1), but dataset actions reach −1.5 and +2.0. Check whether they use the same units.
    - The docs show no way to reset to an exact dataset state. Find out how to start an episode from a dataset state.
  - Then implement CEM and run the 50 evaluation pairs. Report the success rate and planning time, and save a few GIFs.
- **4c. Surprise (optional):** compute the prediction error over time on clean, teleported, and recolored rollouts.

### Stretch
- A λ = 0 run to show collapse, more seeds, and a README with the results.
