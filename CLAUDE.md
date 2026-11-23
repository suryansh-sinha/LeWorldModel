# LeWM-from-scratch

A from-scratch PyTorch reimplementation of LeWorldModel (LeWM) on PushT, trained on one RTX 5070 Ti (16 GB).
Keep it simple: get a correct model trained and evaluated. This is a learning project, not a production system.

- Paper: `references/LeWorldModel.pdf`. Read only the pages that `docs/PLAN.md` cites.
- Plan and settings: `docs/PLAN.md`. Status, decisions and the data-format note: `docs/PROGRESS.md`.

## Rules
- **No official code.** Do not look at, clone, install, or search for the official LeWM repo, its forks, or any other LeWM implementation.
  The values we need are already in PLAN.md.
- **Dataset:** already downloaded and decompressed at `data/pusht_expert_train.h5`, so don't download it again.
  If it ever needs re-downloading, use the HF *dataset* repo `quentinll/lewm-pusht`.
  A *model* repo with the same name holds pretrained weights. Never touch it.
- **Simulator:** `swm/PushT-v1` from `stable-worldmodel`, for the environment only. Install it in Phase 4, not before.
  Don't read or use the library's models, training code, data loaders, planners, or CEM.
- **From scratch:** write the data loader, encoder, predictor, SIGReg, loss, training loop, probes, and CEM planner from torch primitives
  (`nn.Linear`, `nn.LayerNorm`, `F.scaled_dot_product_attention`, etc.). No timm or transformers model classes, and no pretrained weights.
- **Fixed settings, never change them:** 10 epochs, batch size 128, λ = 0.1, 1024 SIGReg projections, 17 knots on [0.2, 4].
  Also no stop-gradient, EMA, or reconstruction loss.

## How we work
- Do one phase at a time, and don't start the next phase on your own.
- Use the simplest thing that works. Don't add infrastructure, benchmarks, or checks that the phase doesn't list.
- Where the paper is silent, use the default in PLAN.md and note it in one line in PROGRESS.md. Don't stop to ask.
  Only ask if something would change a fixed setting or the paper's method.
- Core pieces (encoder, predictor, SIGReg, loss, CEM): explain your approach briefly before writing code.
- Never weaken a test to make it pass, and never claim something ran unless you ran it.
- At the end of a phase: run the tests, show me the output, and add a few lines to PROGRESS.md (what was done, problems and fixes).
  Then ask before committing.

## Commands
- `uv run pytest -q`
- `uv run python -m lewm.train --config configs/smoke.yaml` (tiny subset, a few minutes)
- `uv run python -m lewm.train --config configs/pusht.yaml` (full run)

## Environment
- Python 3.12 managed by uv. torch 2.11.0+cu128, which supports the Blackwell sm_120 GPU.

## Gotchas
- SIGReg and the BatchNorm projectors use batch statistics, so gradient accumulation is not the same as a real batch of 128.
  Ask before using it.
- Train in BF16 autocast, but compute SIGReg in FP32.
- `.gitignore` patterns are anchored to the repo root (`/data/`, `/outputs/`), because an unanchored `data/` once hid `src/lewm/data/`.
  Never commit data, checkpoints, or wandb runs.
