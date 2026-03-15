---
description: Set up an autonomous research loop (auto/ directory with program.md and bench.py)
---

## Context

The user wants to create an autonomous research loop inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted for
paper reproduction and RL training projects. The key innovation over vanilla
autoresearch: instead of optimizing a single metric on a single file, this loop
must first **get to productive gradients** — diagnosing and fixing the chain of
environmental, configuration, and reward-design issues that prevent learning
before optimizing the actual research objective.

## Instructions

Create an `auto/` directory at the project root with two files:

### 1. `auto/bench.py`

A diagnostic benchmark script that runs a **short training loop** (3-5 rounds)
and checks for the failure modes we've learned matter most. It should:

**Import and run the project's actual training code** (not a mock), then check:

1. **Environment health**: CUDA available, model loads, no import errors
2. **Problem diversity**: ground truths differ between rounds (detect fixed-seed bug)
3. **Reward variance**: `reward_std > 0` across generations (GRPO needs this for gradients)
4. **Non-zero gradients**: `loss != 0`, `grad_norm != 0`
5. **Completion quality**: clipped_ratio < 1.0 (completions aren't all truncated)
6. **Answer extraction**: at least some completions have parseable answers
7. **Memory stability**: memory after round N ≈ memory after round 1 (no leak)

Output a structured results block:

```
--- bench results ---
environment:     PASS/FAIL  (details)
problem_diversity: PASS/FAIL  (N unique ground truths out of M)
reward_variance:   PASS/FAIL  (mean std=X across rounds)
gradient_signal:   PASS/FAIL  (mean loss=X, grad=X)
completion_quality: PASS/FAIL (clipped_ratio=X, answers_parsed=Y/Z)
memory_stability:  PASS/FAIL  (round1=XGB, roundN=YGB, delta=ZGB)
---
overall: PASS/FAIL
val_metric: <primary metric value, e.g. mean reward>
```

The bench should:
- Use minimal settings (few rounds, small batch, few generations) for speed
- Complete in under 10 minutes
- Exit 0 on PASS, exit 1 on FAIL
- Print actionable diagnostics on failure (not just "FAIL" but WHY)

**IMPORTANT**: Read the project's existing training code to understand:
- How to invoke training programmatically (import `run_training` or equivalent)
- What config dataclass/args to use
- Where metrics are logged (trainer.state.log_history, log files, etc.)
- What the reward/loss function signature looks like

Tailor bench.py to the specific project, don't write a generic template.

### 2. `auto/program.md`

Agent instructions for the autonomous research loop. Structure:

```markdown
# Auto Research: <project name>

## Objective
<one-line goal, e.g. "minimize validation loss" or "maximize reward on MAA tasks">

## Invariants
- What files/modules are READ-ONLY (evaluation harness, data loaders)
- What files/modules are MODIFIABLE (training config, reward functions, prompts)
- What the bench metric is and which direction is better

## Diagnostic Ladder
Before optimizing the metric, the agent MUST climb this ladder. Each rung
must PASS before moving to the next. This prevents wasting experiments on
a broken setup.

### Rung 1: Environment
- [ ] Training script runs without crashes
- [ ] GPU is utilized, CUDA tensors are on device
- [ ] Model loads and generates text

### Rung 2: Signal
- [ ] Reward function produces variance across generations (std > 0)
- [ ] Loss is non-zero
- [ ] Gradients flow (grad_norm > 0)

### Rung 3: Learning
- [ ] Rewards improve over first 10 rounds
- [ ] Model outputs change between early and late rounds
- [ ] No memory leak (can complete full run)

### Rung 4: Optimization
Only now start optimizing the actual research metric.

## Experiment Loop

LOOP FOREVER:

1. Run `uv run auto/bench.py > auto/run.log 2>&1`
2. Read results: `grep "^overall:\|^val_metric:" auto/run.log`
3. If bench FAILS:
   - Read the diagnostic output to identify which check failed
   - Fix the issue (this is the hardest part — see Debugging Playbook)
   - git commit the fix
   - Re-run bench
4. If bench PASSES:
   - Record result in auto/results.tsv
   - Propose a modification to improve val_metric
   - git commit the change
   - Re-run bench
   - If val_metric improved: keep (advance branch)
   - If val_metric regressed: `git reset --hard HEAD~1`
5. NEVER STOP. If stuck, consult the Debugging Playbook.

## Debugging Playbook

These are ordered by frequency. Check them in order.

### All rewards identical (reward_std = 0)
- **Cause**: All generations score the same → GRPO normalizes to zero → no gradient
- **Fixes**:
  - Check if problems are identical each round (fixed seed bug)
  - Make reward function more granular (continuous vs binary)
  - Increase num_generations for more diversity
  - Check if reward function is actually being called with model output

### Loss is zero but rewards vary
- **Cause**: Reward normalization is collapsing signal
- **Fixes**:
  - Check GRPO's `scale_rewards` setting
  - Ensure per_device_train_batch_size divides num_generations

### Completions all max length (clipped_ratio = 1.0)
- **Cause**: max_completion_length too short for the task
- **Fixes**:
  - Increase max_completion_length (but watch memory)
  - Simplify the prompt to encourage shorter responses
  - Check if model is in a repetition loop

### Correct format but wrong answers
- **Cause**: Model learned structure but not content
- **Fixes**:
  - Verify answer extraction parses correctly
  - Check if ground truth format matches expected parser format
  - Try easier problems first (lower difficulty levels)

### OOM / Process killed
- **Cause**: num_generations × max_completion_length × model_size > memory
- **Fixes**:
  - Reduce num_generations
  - Reduce max_completion_length
  - Enable gradient_checkpointing
  - Check for memory leak (new Trainer per round without cleanup)

### Memory grows between rounds
- **Cause**: Resources not freed between training rounds
- **Fixes**:
  - Delete trainer after each round
  - Call gc.collect() and torch.cuda.empty_cache()
  - Check for accumulating lists/caches

## Results Tracking

Log experiments to `auto/results.tsv` (tab-separated):

```
commit	val_metric	status	description
a1b2c3d	-0.094	keep	baseline
b2c3d4e	0.150	keep	increase num_generations to 16
c3d4e5f	-0.094	discard	switch to binary reward (no variance)
```
```

## What to create

1. Read the project's training code thoroughly to understand the API
2. Create `auto/bench.py` tailored to this specific project
3. Create `auto/program.md` with project-specific details filled in
4. Create `auto/results.tsv` with just the header row
5. Verify bench.py runs: `uv run auto/bench.py`

Do NOT create generic templates. The bench.py must actually import and run
this project's training pipeline. The program.md must reference this project's
actual files, metrics, and configuration.
