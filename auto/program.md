# Auto Research: repro-maa

Autonomous research loop for reproducing Meta-Ability-Alignment experiments.

## Objective

**Maximize mean reward** on MAA reasoning tasks (deduction, induction, abduction)
via GRPO training of Qwen3.5-9B. Higher is better. The reward combines continuous
format compliance (0-1) with MAA content scoring (-2 to +3).

## Files

| File | Role |
|------|------|
| `auto/bench.py` | **Benchmark** — 3-round diagnostic, exit 0 = healthy |
| `auto/results.tsv` | **Experiment log** — tab-separated results |
| `src/repro_maa/train.py` | **MODIFIABLE** — training loop, GRPO config, round logic |
| `src/repro_maa/prompt_reward_bridge.py` | **MODIFIABLE** — system prompt, reward function, format scoring |
| `src/repro_maa/compare_training.py` | **MODIFIABLE** — comparison harness, CLI args |
| `src/repro_maa/stream.py` | Read-only — curriculum selection (CuriosityStream) |
| `src/repro_maa/task_cell.py` | Read-only — problem generation |
| `src/repro_maa/mdl_scorer.py` | Read-only — MDL curiosity scorer |
| `src/repro_maa/maa_compat.py` | Read-only — MAA submodule bridge |
| `Meta-Ability-Alignment/` | Read-only — upstream MAA scorers (submodule) |

## Diagnostic Ladder

Before optimizing the metric, climb this ladder. Each rung must PASS in
`auto/bench.py` before moving to the next. This prevents wasting hours of
GPU time on a broken setup.

### Rung 1: Environment
- [ ] `uv run auto/bench.py` runs without import/CUDA errors
- [ ] Model loads onto GPU in bf16
- [ ] `HF_HUB_DISABLE_XET=1` is set if HuggingFace XET errors occur

### Rung 2: Signal
- [ ] `reward_variance: PASS` — reward_std > 0 across generations
- [ ] `gradient_signal: PASS` — loss != 0, grad_norm > 0
- [ ] If reward_std = 0: check problem diversity, reward granularity, num_generations

### Rung 3: Learning
- [ ] Rewards improve over 10+ rounds (run a longer training)
- [ ] `completion_quality: PASS` — clipped_ratio < 0.95
- [ ] `memory_stability: PASS` — no memory growth between rounds

### Rung 4: Optimization
All bench checks pass. Now optimize val_metric via:
- Prompt engineering (system prompt in prompt_reward_bridge.py)
- Reward shaping (format weights, content scoring granularity)
- Training hyperparameters (learning_rate, num_generations, temperature)
- Curriculum policy (epsilon, window_size)
- Completion length vs generation count tradeoff

## Experiment Loop

**Environment**: Set `HF_HUB_DISABLE_XET=1` before all runs.

LOOP FOREVER:

1. Run bench: `HF_HUB_DISABLE_XET=1 uv run auto/bench.py > auto/run.log 2>&1`
2. Check result: `tail -10 auto/run.log`
3. If bench FAILS:
   - Read diagnostic output to identify which check failed
   - Consult the Debugging Playbook below
   - Fix the root cause in the modifiable files
   - `git add <files> && git commit -m "fix: <what you fixed>"`
   - Re-run bench
4. If bench PASSES:
   - Record in `auto/results.tsv`
   - Propose an improvement to increase val_metric
   - `git commit` the change
   - Re-run bench
   - If val_metric improved AND all checks still pass: keep
   - Otherwise: `git reset --hard HEAD~1`
5. **NEVER STOP.** The user may be asleep. Keep iterating until interrupted.

## Debugging Playbook

Ordered by frequency. Check in order.

### reward_std = 0 (all generations score identically)

This is the #1 killer. GRPO normalizes rewards within each prompt's
generations. If all N generations get the same score, normalized rewards
are all zero → zero loss → zero gradient → no learning.

**Root causes and fixes:**

1. **Same problem every round** — `TaskCell.generate()` uses a fixed seed
   that doesn't advance. Fix: ensure `self._seed += n` after each call.

2. **Binary reward function** — format is pass/fail, all completions fail
   the same way. Fix: use continuous format scoring
   (`tags_present / tags_expected * weight`) so 3/4 tags scores higher than 0/4.

3. **Too few generations** — with `num_generations=4`, all 4 often score
   identically. Fix: increase to 16+ for more diversity.

4. **Reward function ignores answer content when format fails** — the MAA
   scorer skips content validation on format failure. Fix: extract `<answer>`
   independently and score content regardless of format.

### loss = 0 but reward_std > 0

- Check `per_device_train_batch_size` divides `num_generations`
- Check `generation_batch_size` equals `num_generations` (TRL constraint)

### completions all max length (clipped_ratio = 1.0)

- `max_completion_length` too short. The model rambles through reasoning
  and runs out of tokens before writing `<answer>`.
- Increase to 2048. Watch memory: `16 gens × 2048 tokens × 9B model ≈ 48GB`.
- 4096 tokens with 16 gens will OOM on DGX Spark (128GB unified).

### Model produces `<answer>...</answer>` with literal ellipsis

- System prompt says "inside `<think>...</think>` tags" and the model
  interprets `...` literally. Fix: rewrite prompt with explicit example
  showing actual content in the tags.

### OOM / process killed silently

- `num_generations × max_completion_length` is the memory multiplier.
  Safe budget for 9B bf16 on 128GB: ~50K total tokens
  (e.g., 16 × 2048 or 64 × 768).
- Check for memory leak: new GRPOTrainer per round without cleanup.
  Fix: `del trainer; gc.collect(); torch.cuda.empty_cache()` after each round.

### Memory grows between rounds

- GRPOTrainer creates optimizer/accelerator state each round. Without
  explicit cleanup, GPU memory grows until OOM.
- Fix: delete trainer, dataset, train_output after each round + gc + empty_cache.

### HuggingFace model download fails (XET errors)

- Set `HF_HUB_DISABLE_XET=1` environment variable.
- If `.locks` directory has wrong permissions: `sudo chown -R $USER ~/.cache/huggingface/hub/.locks/`

### PyTorch is CPU-only (torch+cpu)

- `pyproject.toml` needs `[tool.uv] extra-index-url = ["https://download.pytorch.org/whl/cu130"]`
- After changing, run: `uv pip install torch --index-url https://download.pytorch.org/whl/cu130 --reinstall`

## Hardware Notes (DGX Spark)

- NVIDIA GB10, 128GB unified memory (shared CPU/GPU)
- `nvidia-smi` shows "Not Supported" for memory — use `free -h` instead
- CUDA capability 12.1, PyTorch warns about max supported 12.0 (works fine)
- vLLM incompatible (requires CUDA 12 libs, we have CUDA 13)
- Throughput-optimized: prefer larger batch sizes (8+)

## Results Tracking

Log to `auto/results.tsv` (tab-separated). DO NOT commit this file. It SHOULD NOT be tracked by git:

```
commit	val_metric	status	description
```

- `commit`: short git hash (7 chars)
- `val_metric`: mean reward from bench.py (higher is better)
- `status`: `keep`, `discard`, or `crash`
- `description`: one-line summary of what changed
