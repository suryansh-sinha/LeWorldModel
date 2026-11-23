# LeWM-from-scratch

A from-scratch PyTorch reimplementation of **LeWorldModel (LeWM)** on the PushT environment, trained on one NVIDIA RTX 5070 Ti (16 GB VRAM).

The goal is to implement the method described in the paper, train for the paper's 10 epochs, and evaluate planning, latent probing, and violation-of-expectation surprise detection.

## 1. Sources of truth

Read these project files when relevant:

- `references/LeWorldModel.pdf` — the original paper. This is the primary authority for the method and every setting explicitly specified by the paper. The PDF is 28 pages: read only the pages `docs/PLAN.md` cites for the task at hand (for example pp. 17–18), not the whole document.
- `docs/PLAN.md` — the project-specific specification. It records paper-specified settings and explicitly labels any defaults borrowed from the official implementation because the paper leaves them unspecified.
- `docs/PROGRESS.md` — current project state, decisions, experiment results, blockers, and development history.

### Strict restriction: do not access the official implementation

**Do not access, inspect, clone, download, execute, install, or copy from the official LeWorldModel implementation or any of its forks, mirrors, branches, or derivative repositories.** This restriction includes source files, configuration files, notebooks, issues, pull requests, commit history, and implementation documentation belonging to those repositories.

In particular:

- Do not browse to or inspect the official LeWorldModel GitHub repository or any fork.
- Do not use Git, GitHub, a browser, a web search, package installation, or another tool to retrieve its code or configuration.
- Do not search for an implementation of LeWM to copy or adapt.
- Do not infer that a missing value can be obtained by inspecting the official repository.

Some defaults from the official implementation have already been transcribed into `docs/PLAN.md` from reference configuration information supplied by the project owner. Those values are intentionally documented there and labelled **“used in the official implementation”**. Use the values written in `PLAN.md`; do not look up their original source.

You may consult the paper in `references/LeWorldModel.pdf`, general PyTorch documentation, and documentation for permitted environment, dataset, logging, and plotting libraries. Do not use other LeWM implementations as a source of code or design.

**Dataset and simulator exception.** The PushT dataset and simulator source is chosen by the owner and recorded in `docs/PLAN.md` (Section 4, decisions 0a and 0b). Those chosen sources may be downloaded, installed, and used **only** for loading data and running the environment. Do not read, import, or copy any model, loss, training, planning, or policy code from them. If decision 0b (the simulator) is still unresolved, ask before installing any simulator.

The dataset is the Hugging Face **dataset** repo `quentinll/lewm-pusht`. A **model** repo with the same name holds the official pretrained checkpoint. Always download with the dataset repo type, and never download, load, or inspect the model repo or any other repo in that collection.

## 2. Specification precedence

Apply the following precedence whenever settings appear to conflict:

1. **The paper wins** for every method or value it explicitly specifies.
2. **The owner-approved choices in `docs/PLAN.md` win** where the paper is silent. This includes clearly labelled defaults borrowed from the official implementation and explicitly recorded project decisions.
3. If neither source specifies a necessary detail, identify it as an open decision, explain the options and consequences, and ask before making a methodological choice.

Important owner-approved choices already recorded in `docs/PLAN.md`:

- Train for **10 epochs**, not 100.
- Use the paper's SIGReg weight **λ = 0.1**, not 0.09.
- Use **M = 1024** SIGReg projections.
- Use the paper's SIGReg integration interval **[0.2, 4]**, not [0, 3].
- Use **17 integration knots** because the paper does not specify the knot count; this value is labelled as used in the official implementation.

Do not silently change these choices for convenience or to improve a metric.

## 3. How we work

- At the start of a session, read `docs/PROGRESS.md` and only the active phase (plus directly relevant sections) of `docs/PLAN.md`. This file is loaded automatically.
- Consult the relevant paper section or appendix before implementing a paper-defined component. The paper page references are recorded in `docs/PLAN.md`.
- Work on one phase or clearly bounded sub-step at a time. Do not begin later phases automatically.
- For a substantial phase or multi-file task, use Claude Code Plan Mode: inspect the current codebase, prepare an implementation plan, identify files and tests, and wait for my approval before editing. Plan Mode is not necessary for a small, unambiguous bug fix or trivial change.
- Do not implement a phase until I approve the proposed plan.
- For the core components — ViT encoder/projectors, action-conditioned AdaLN predictor, SIGReg, training objective, and CEM planner — explain the proposed tensor shapes, equations, design, and tests before implementation, then wait for my approval.
- Implement only the approved scope. Do not add unrelated features, optimize prematurely, or expand into future phases.
- If an ambiguity affects the method, data alignment, evaluation protocol, or a result, stop and ask rather than silently choosing an interpretation.

### Failure handling

- When a test or check fails, investigate the root cause and attempt reasonable fixes within the approved scope.
- Never weaken, delete, skip, or rewrite an acceptance criterion merely to make a phase pass.
- Never claim that code, a test, or an experiment succeeded unless it was actually run and the result was observed.
- If a fix requires changing a paper-defined method, an owner-approved setting, an acceptance criterion, or the phase scope, stop and explain the issue before proceeding.

### End-of-phase handoff

At the end of each phase:

1. Run the phase's required checks and relevant regression tests.
2. Show the exact commands and concise outputs, metrics, plots, or other evidence.
3. Report files created/changed, decisions made, deviations, and unresolved issues.
4. Update `docs/PROGRESS.md` with verified status, meaningful changes, decisions, problems and resolutions, experiment evidence, and the next step.
5. Update the phase status in `docs/PLAN.md` only when its acceptance criteria are met.
6. Show me the Git diff and suggest a commit message.
7. **Wait for my approval before committing.** Never push to a remote without explicit approval.

Do not record speculative or unverified events in `PROGRESS.md`. Distinguish attempted approaches from successful fixes.

## 4. From-scratch implementation rules

Implement the following components in this repository using native PyTorch building blocks:

- ViT-Tiny image encoder.
- Encoder and predictor projectors.
- Action-block encoder.
- Causal, action-conditioned autoregressive predictor with AdaLN.
- SIGReg and the Epps–Pulley statistic.
- LeWM prediction and regularization objective.
- Training, validation, checkpointing, and resume logic.
- Linear and MLP probes.
- CEM planner and MPC evaluation.

Do not use pretrained model weights. Do not use model classes from `timm`, `transformers`, Hugging Face, or similar model libraries for the encoder or predictor. Do not import model components from another LeWM implementation. Use ordinary PyTorch layers and tensor operations; PyTorch primitives such as `nn.Linear`, `nn.LayerNorm`, `nn.BatchNorm1d`, `nn.MultiheadAttention` or scaled dot-product attention, `nn.Conv1d`, and `torch.nn.functional` are allowed.

Permitted external components include the PushT simulator/environment, dataset download and decoding tools, PyTorch itself, experiment logging (for example, Weights & Biases), and plotting libraries. These are dependencies, not sources of LeWM model code.

The paper's method is end-to-end: do not add stop-gradient, EMA target encoders, a reconstruction loss, reward prediction, privileged state inputs, or additional anti-collapse losses.

## 5. Engineering conventions

- Use a clear, modular PyTorch project layout.
- Store experiment settings in YAML files under `configs/`; do not hardcode tunable hyperparameters inside model modules.
- Add tests for each important module and test tensor shapes, dtypes, devices, and numerical behaviour.
- Put shape comments on tensor reshaping/permutation lines, for example `# (B, T, D)`.
- Use explicit, consistent tensor naming and document the meaning of each time/action index.
- Seed Python, NumPy, PyTorch, CUDA, dataset splitting, and evaluation sampling where applicable. Log the full config, seed, Git commit hash, and run identifier with each experiment.
- Save checkpoints atomically and verify that checkpoint resume restores model, optimizer, scheduler, epoch/step, and RNG state where practical.
- Never commit datasets, downloaded archives, checkpoints, generated rollout media, or experiment-tracking directories. Keep them out of Git with `.gitignore`.

## 6. Hardware and numerical considerations

- Target hardware: RTX 5070 Ti with 16 GB VRAM. Verify the actual GPU, driver, CUDA-enabled PyTorch build, and `torch.cuda.is_available()` at setup time; do not assume a CUDA runtime is working merely because a package installed.
- Use BF16 autocast for model training as specified in `docs/PLAN.md`, while computing SIGReg's characteristic-function/statistical calculations and sensitive reductions in FP32.
- Check for NaN/Inf in embeddings, both loss terms, and gradients. Log prediction loss, SIGReg, per-dimension embedding standard deviation, and effective rank.
- Measure peak allocated VRAM and throughput before starting the full run.
- SIGReg and BatchNorm projectors depend on batch statistics. Gradient accumulation is not equivalent to processing a true batch of 128 for these operations. Do not silently replace the paper's batch size with gradient accumulation or a smaller batch. If memory prevents batch size 128, investigate memory use and present the trade-off for approval before changing the effective batch/statistics.
- Do not change precision, batch size, SIGReg settings, or model dimensions to make a run fit without recording and getting approval for the deviation.

## 7. Expected commands

Use the project's actual package/module layout when implementing these commands. Keep them documented and working:

```bash
# Tests
uv run pytest -q

# Fast smoke test (small, explicitly labelled smoke configuration)
uv run python -m lewm.train --config configs/smoke.yaml

# Paper-faithful PushT training: 10 epochs
uv run python -m lewm.train --config configs/pusht.yaml
```

The smoke configuration is for debugging only. It must not silently alter the paper-faithful full-training configuration.
