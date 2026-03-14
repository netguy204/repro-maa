<!--
This document captures HOW you'll achieve the chunk's GOAL.
It should be specific enough that each step is a reasonable unit of work
to hand to an agent.
-->

# Implementation Plan

## Approach

Create a `compare_training` module (`src/repro_maa/compare_training.py`) that orchestrates two training runs (curiosity vs. fixed curriculum) and produces a unified analysis. The module:

1. **Launches training runs** — calls `train.run_training()` twice with different `TrainConfig` settings (one with `curriculum="curiosity"`, one with `curriculum="fixed"`), or resumes from existing JSONL logs if they already exist.
2. **Generates comparison plots** — delegates to `visualize.generate_all()` which already supports side-by-side curiosity vs. baseline plots for all four visualization types.
3. **Produces a summary report** — a new `generate_report()` function that reads both JSONL logs and computes quantitative comparisons: cumulative reward curves, steps-to-threshold, per-ability reward breakdown, cell allocation differences, and plateau detection. Output is a Markdown file.
4. **Provides data for selection_granularity investigation** — the report includes per-ability vs. per-level variance decomposition of cell selections, directly feeding hypotheses H1–H4.

The module builds entirely on existing infrastructure:
- `train.py` for the training loop (both curriculum modes)
- `simulation.py` for `StepRecord`, `read_log`, `compare_runs`
- `visualize.py` for all four plot types
- `mdl_scorer.py` indirectly via the logged MDL scores

The entry point is `python -m repro_maa.compare_training` with CLI flags forwarded to `TrainConfig` plus comparison-specific options (`--skip-training` to use existing logs, `--output-dir` for all artifacts).

Testing follows the project's philosophy: test the measurement apparatus, not the research outcome. Tests verify that the report generator produces well-formed output from fixture logs, that plateau detection logic is correct, and that the CLI parses arguments properly. No tests assert that curiosity beats fixed curriculum.

## Subsystem Considerations

No subsystems documented in `docs/subsystems/`. This chunk is a pure orchestration and analysis layer over existing modules — no new cross-cutting patterns introduced.

## Sequence

### Step 1: Define ComparisonConfig and report data structures

Create `src/repro_maa/compare_training.py` with a `ComparisonConfig` dataclass that wraps the shared training parameters plus comparison-specific settings:

- `num_rounds: int` — number of training rounds per run (forwarded to `TrainConfig`)
- `batch_size: int` — problems per round
- `seed: int` — shared seed for reproducibility
- `output_dir: str` — root directory for all artifacts (default: `"comparison_output"`)
- `skip_training: bool` — if True, expect existing JSONL logs and skip training
- `curiosity_log: str | None` — path to existing curiosity log (for `skip_training` mode)
- `baseline_log: str | None` — path to existing baseline log (for `skip_training` mode)
- Fields forwarded to `TrainConfig`: `model_name`, `num_generations`, `per_device_train_batch_size`, `max_prompt_length`, `max_completion_length`, `learning_rate`, `gradient_checkpointing`, `bf16`, `epsilon`, `window_size`, `checkpoint_interval`

Also define a `ReportData` dataclass to hold computed analysis results:
- `curiosity_records: list[StepRecord]`
- `baseline_records: list[StepRecord]`
- `reward_thresholds: dict[str, dict[str, int | None]]` — per-ability steps to reach reward thresholds
- `plateau_rounds: dict[str, list[int]]` — per-ability rounds where reward improvement stalled
- `cell_allocation: dict[str, dict[str, int]]` — per-run cell selection frequency
- `per_ability_cumulative: dict[str, dict[str, list[float]]]` — per-ability cumulative reward curves
- `selection_variance: dict[str, float]` — ability vs. difficulty variance decomposition (for selection_granularity investigation)

Location: `src/repro_maa/compare_training.py`

### Step 2: Implement `run_comparison()` — the training orchestrator

Write `run_comparison(config: ComparisonConfig) -> tuple[list[StepRecord], list[StepRecord]]` that:

1. If `skip_training` is False:
   - Build a `TrainConfig` with `curriculum="curiosity"` and `output_dir=<output_dir>/curiosity/`, call `train.run_training()`.
   - Build a `TrainConfig` with `curriculum="fixed"` and `output_dir=<output_dir>/fixed/`, call `train.run_training()`.
2. If `skip_training` is True:
   - Read existing JSONL logs using `simulation.read_log()` from the specified paths (or default locations `<output_dir>/curiosity/stream_log.jsonl` and `<output_dir>/fixed/stream_log.jsonl`).
3. Return both log lists.

This is thin glue — the complexity is in `train.py` (already implemented) and the analysis functions below.

Location: `src/repro_maa/compare_training.py`

### Step 3: Implement analysis functions

Write the analytical core — a set of pure functions that take `list[StepRecord]` inputs and return computed metrics:

**`compute_steps_to_threshold(records, thresholds)`** — For each ability, find the first round where cumulative reward per ability exceeds each threshold. Thresholds default to `[0.0, 10.0, 50.0, 100.0]`. Returns `dict[ability, dict[threshold_str, step_or_None]]`.

**`detect_plateaus(records, window=10, min_improvement=0.5)`** — For each ability, identify rounds where the rolling mean reward over the last `window` rounds has not improved by more than `min_improvement` from the window before it. Returns `dict[ability, list[plateau_start_round]]`. This directly addresses the goal's "plateau detection" requirement and the MAA paper's observation that "the 7B model converges by Level 2."

**`compute_per_ability_cumulative(records)`** — Break the cumulative reward curve down by ability, yielding three separate curves showing where each ability's learning happens. Returns `dict[ability, list[float]]` (cumulative sum of batch rewards for rounds selecting that ability).

**`compute_selection_variance(records)`** — ANOVA-style decomposition: compute what fraction of MDL score variance is explained by the ability factor vs. the difficulty factor. Returns `dict["ability_fraction", "difficulty_fraction", "residual_fraction"]`. This feeds the selection_granularity investigation H1.

**`compute_cell_allocation(records)`** — Count cell selection frequencies, same as `compare_runs` but preserved separately for report formatting. Returns `dict[cell_key, int]`.

Location: `src/repro_maa/compare_training.py`

### Step 4: Implement `generate_report()` — Markdown report writer

Write `generate_report(curiosity_records, baseline_records, output_path)` that:

1. Calls all analysis functions from Step 3 on both record lists.
2. Formats a Markdown report with sections:
   - **Overview**: Run parameters (rounds, seed, curriculum type, model name).
   - **Cumulative Reward Comparison**: Final cumulative reward for each run, advantage.
   - **Steps to Reward Thresholds**: Table comparing how many rounds each run took to reach cumulative reward milestones, per ability.
   - **Plateau Detection**: For each run, list rounds where each ability's learning plateaued. Highlight whether curiosity reallocated away from plateaued cells faster.
   - **Cell Allocation**: Side-by-side table of selection counts per cell. Note which cells curiosity favored that fixed curriculum didn't, and vice versa.
   - **Per-Ability Reward Curves**: Summary statistics for each ability's contribution to total reward under each curriculum.
   - **Selection Granularity Analysis**: Ability vs. difficulty variance decomposition for the curiosity run. References hypotheses H1–H4 from `docs/investigations/selection_granularity/`.
3. Writes the report to `<output_dir>/comparison_report.md`.

Location: `src/repro_maa/compare_training.py`

### Step 5: Implement `generate_plots()` wrapper

Write `generate_plots(curiosity_log_path, baseline_log_path, output_dir)` that delegates to `visualize.generate_all()` with the correct paths. This is a thin wrapper that exists so the comparison module can call it in the orchestration pipeline without the caller needing to know about `visualize`.

Location: `src/repro_maa/compare_training.py`

### Step 6: Implement CLI entry point

Write `parse_args()` and `main()` functions for the CLI:

```
python -m repro_maa.compare_training [options]
```

CLI flags:
- `--num-rounds N` (default 100)
- `--batch-size N` (default 4)
- `--seed N` (default 42)
- `--output-dir PATH` (default `comparison_output`)
- `--skip-training` (flag; use existing logs)
- `--curiosity-log PATH` (for `--skip-training`)
- `--baseline-log PATH` (for `--skip-training`)
- All `TrainConfig` forwarding flags: `--model-name`, `--num-generations`, `--per-device-train-batch-size`, `--max-prompt-length`, `--max-completion-length`, `--learning-rate`, `--gradient-checkpointing/--no-gradient-checkpointing`, `--bf16/--no-bf16`, `--epsilon`, `--window-size`, `--checkpoint-interval`

The `main()` function:
1. Parse args into `ComparisonConfig`
2. Call `run_comparison()` to get both log lists
3. Call `generate_plots()` to produce the four PNGs
4. Call `generate_report()` to produce the Markdown summary
5. Print paths to all generated artifacts

Location: `src/repro_maa/compare_training.py`

### Step 7: Write tests

Create `tests/test_compare_training.py` with tests that verify the measurement apparatus:

**Fixture**: Build two synthetic `list[StepRecord]` lists — one simulating curiosity behavior (varied cell selections, non-zero MDL scores) and one simulating fixed curriculum (sequential ability/level assignments, zero MDL scores). ~20 records each with realistic reward distributions.

**Test: `test_compute_steps_to_threshold`** — Given fixture records where deduction reaches cumulative +10.0 at step 5, verify `compute_steps_to_threshold` returns step 5 for that threshold. Verify None returned for unreached thresholds.

**Test: `test_detect_plateaus`** — Construct a record sequence where one ability's reward is flat for 15 rounds (plateau), then jumps. Verify the plateau is detected at the correct starting round.

**Test: `test_compute_per_ability_cumulative`** — Verify that the per-ability cumulative sums are correct: only rounds selecting that ability contribute, and the sum matches the total cumulative for that ability.

**Test: `test_compute_selection_variance`** — Use records where ability fully determines MDL score (deduction=high, induction=medium, abduction=low, regardless of level). Verify ability_fraction is close to 1.0. Then use records where level determines score. Verify difficulty_fraction dominates.

**Test: `test_generate_report_structure`** — Call `generate_report()` with fixture data, verify the output is a string containing expected section headings ("## Cumulative Reward Comparison", "## Plateau Detection", "## Selection Granularity Analysis") and is valid Markdown (no unclosed formatting).

**Test: `test_parse_args_defaults`** — Verify `parse_args([])` returns a `ComparisonConfig` with expected defaults.

**Test: `test_parse_args_skip_training`** — Verify `parse_args(["--skip-training", "--curiosity-log", "a.jsonl", "--baseline-log", "b.jsonl"])` populates the correct fields.

**Test: `test_run_comparison_skip_training_reads_logs`** — Write two temporary JSONL log files, call `run_comparison()` with `skip_training=True`, verify it returns the expected records.

Location: `tests/test_compare_training.py`

### Step 8: Update `__init__.py` exports and `__main__.py`

- Add `compare_training` imports to `src/repro_maa/__init__.py` — export `ComparisonConfig`, `run_comparison`, `generate_report`.
- No change to `__main__.py` needed (the module is invoked as `python -m repro_maa.compare_training` directly).

Location: `src/repro_maa/__init__.py`

All steps add a module-level backreference comment:
```python
# Chunk: docs/chunks/training_comparison - Training comparison and analysis
```

## Dependencies

- **curiosity_grpo_loop** (ACTIVE): Provides `train.py` with `run_training()`, `TrainConfig`, `build_dataset`, `make_dispatching_reward_func`, `append_step_record`. This chunk's `run_comparison()` calls `run_training()` directly.
- **stream_visualization** (ACTIVE): Provides `visualize.generate_all()` for all four plot types. Already supports optional baseline comparison.
- **simulation_harness** (ACTIVE): Provides `StepRecord`, `read_log`, `write_log`, `compare_runs`, `FixedCurriculumBaseline`.
- No new external libraries required. All analysis uses numpy (already a dependency) and the standard library.

## Risks and Open Questions

- **Training duration**: Two sequential training runs may take several hours on DGX Spark. The `--skip-training` flag mitigates this for iterating on the analysis without re-training. If a run crashes partway through, append-mode JSONL logging (from `train.py`) preserves partial logs.
- **ANOVA decomposition with sparse data**: The selection variance analysis assumes enough rounds selecting each cell to compute meaningful variance. If epsilon is low and the curiosity stream strongly favors a few cells, many cells will have zero selections and the ANOVA will be degenerate. Mitigation: handle cells with <2 selections gracefully (exclude from variance computation, note in report).
- **`_extract_rewards` precision**: The current `train.py` uses mean-based reward estimates rather than true per-problem rewards. This affects per-ability analysis precision but not the overall comparison direction. Document this limitation in the report.
- **Plateau detection sensitivity**: The window size and min_improvement threshold for plateau detection are heuristic. Expose them as parameters so the operator can tune them.

## Deviations

<!--
POPULATE DURING IMPLEMENTATION, not at planning time.

When reality diverges from the plan, document it here:
- What changed?
- Why?
- What was the impact?

Minor deviations (renamed a function, used a different helper) don't need
documentation. Significant deviations (changed the approach, skipped a step,
added steps) do.

Example:
- Step 4: Originally planned to use std::fs::rename for atomic swap.
  Testing revealed this isn't atomic across filesystems. Changed to
  write-fsync-rename-fsync sequence per platform best practices.
-->