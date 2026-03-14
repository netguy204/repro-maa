<!--
This document captures HOW you'll achieve the chunk's GOAL.
It should be specific enough that each step is a reasonable unit of work
to hand to an agent.
-->

# Implementation Plan

## Approach

Create a new `src/repro_maa/visualize.py` module that reads JSONL simulation logs (via the existing `read_log()` / `StepRecord` from `simulation.py`) and produces four matplotlib figures as PNG files. The module exposes both a programmatic API (individual plot functions + a `generate_all` orchestrator) and a CLI entry point via `python -m repro_maa.visualize`.

**Strategy:**

- **Pure consumer of existing data**: The module imports only `read_log` and `StepRecord` from `simulation.py`. It performs no simulation — it reads finished logs and plots them.
- **One function per plot**: Each of the four success-criteria plots is an independent function that takes a list of `StepRecord` and returns a `matplotlib.figure.Figure`. This makes each plot unit-testable in isolation.
- **`generate_all` orchestrator**: Accepts one or two log paths (curiosity + optional baseline), calls each plot function, saves PNGs to an output directory.
- **Matplotlib with Agg backend**: Use the non-interactive `Agg` backend so the module works in headless environments (CI, remote servers) without display dependencies.
- **Tests follow TESTING_PHILOSOPHY.md**: Structural tests verify that output files are created, figures have the expected number of axes, axes have correct labels/titles, and data arrays have expected shapes. No aesthetic assertions.

**Existing code leveraged:**

- `simulation.read_log(path) -> list[StepRecord]` — log deserialization
- `simulation.StepRecord` — the data schema (step, ability, level, mdl_score, selection_reason, batch_rewards, batch_mean_reward, cumulative_reward, reward_history_summary)
- `simulation.compare_runs()` — for reference on cell frequency key format (`"{ability}_L{level}"`)
- The 3×5 cell grid convention: abilities = `["deduction", "induction", "abduction"]`, levels = `[1, 2, 3, 4, 5]`

**New dependency:** `matplotlib` added to `pyproject.toml` dependencies.

## Subsystem Considerations

No existing subsystems are relevant. This chunk is a pure leaf consumer of the simulation log data.

## Sequence

Add `# Chunk: docs/chunks/stream_visualization` backreference at module level in `visualize.py` and on the CLI entry point in `__main__.py`.

### Step 1: Add matplotlib dependency

Add `matplotlib` to the `dependencies` list in `pyproject.toml`.

Location: `pyproject.toml`

### Step 2: Write structural tests for all four plots and the CLI

Write `tests/test_visualize.py` with tests that create synthetic `StepRecord` lists (reusing patterns from `test_simulation.py`) and assert structural properties of each plot. Tests are written first per TDD — they will fail until the implementation exists.

**Test fixtures (in conftest or local):**
- `make_records(n, strategy)` — helper that produces a list of `n` `StepRecord` objects with plausible data for either `"curiosity"` or `"fixed_schedule"` selection_reason. Vary ability/level across steps so plots are non-trivial.
- Use `matplotlib.pyplot.close("all")` in teardown to avoid resource leaks.

**Tests to write:**

1. `test_cell_selection_timeline_axes` — calls `plot_cell_selection_timeline(records)`, asserts: returns a `Figure`, has 1 `Axes`, x-axis spans `[0, n-1]` (steps), y-axis has tick labels for the cell identifiers, title contains "selection".
2. `test_cell_selection_timeline_two_streams` — calls with both curiosity and baseline records, asserts: figure has 2 subplots (one per stream), or a single axes with two distinct series.
3. `test_mdl_score_evolution_axes` — calls `plot_mdl_evolution(records)`, asserts: returns a `Figure`, has at least 1 line plotted per cell that had non-zero MDL scores, x-axis is steps, y-axis label references "MDL".
4. `test_cumulative_reward_comparison` — calls `plot_cumulative_reward(curiosity_records, baseline_records)`, asserts: returns a `Figure`, has exactly 2 lines, y-axis label references "reward", legend has 2 entries.
5. `test_selection_heatmap_shape` — calls `plot_selection_heatmap(curiosity_records, baseline_records)`, asserts: returns a `Figure`, has 2 subplots (side by side), each subplot shows a 3×5 grid (abilities × levels).
6. `test_generate_all_creates_png_files` — calls `generate_all(curiosity_log_path, baseline_log_path, output_dir)` with temp paths, asserts: 4 PNG files are created in `output_dir` with expected filenames.
7. `test_generate_all_single_log` — calls `generate_all` with only a curiosity log (no baseline), asserts: the plots that don't need a baseline still produce files; cumulative reward plot shows one line; heatmap shows one panel.
8. `test_cli_runs_successfully` — invokes `python -m repro_maa.visualize <log_path> --output-dir <tmpdir>` via `subprocess.run`, asserts: exit code 0, PNG files exist in output dir.

Location: `tests/test_visualize.py`

### Step 3: Implement data extraction helpers

Create `src/repro_maa/visualize.py` with helper functions that transform `list[StepRecord]` into plot-ready data structures.

Functions:

- `_cell_key(ability: str, level: int) -> str` — returns `"{ability} L{level}"` display label.
- `_extract_selections(records: list[StepRecord]) -> tuple[list[int], list[str]]` — returns `(steps, cell_labels)` where each entry is the step number and the selected cell's display label.
- `_extract_mdl_timeseries(records: list[StepRecord]) -> dict[str, list[tuple[int, float]]]` — returns `{cell_label: [(step, mdl_score), ...]}` by scanning `reward_history_summary` for each step. Only the *selected* cell's MDL score is directly available (`record.mdl_score`), so this function records `(step, mdl_score)` for the selected cell at each step. For the MDL evolution plot, we track per-cell scores from the `reward_history_summary` — since full per-cell MDL isn't logged, we plot the selected cell's MDL at each step as the primary series, and optionally reconstruct approximate MDL trajectories from the reward history counts.
- `_extract_cumulative_rewards(records: list[StepRecord]) -> tuple[list[int], list[float]]` — returns `(steps, cumulative_rewards)`.
- `_extract_selection_counts(records: list[StepRecord]) -> np.ndarray` — returns a 3×5 numpy array (rows = abilities in order `["deduction", "induction", "abduction"]`, cols = levels 1–5) with selection counts.

Location: `src/repro_maa/visualize.py`

### Step 4: Implement `plot_cell_selection_timeline`

```python
def plot_cell_selection_timeline(
    curiosity_records: list[StepRecord],
    baseline_records: list[StepRecord] | None = None,
) -> Figure:
```

Creates a figure with 1 or 2 subplots (stacked vertically). Each subplot is a scatter/strip plot with step on x-axis and cell label on y-axis. Points are color-coded by ability (3 colors: one for deduction, one for induction, one for abduction). Y-axis tick labels are the 15 cell names sorted by ability then level.

Subplot titles: "Curiosity-Driven" and "Fixed Curriculum".

Location: `src/repro_maa/visualize.py`

### Step 5: Implement `plot_mdl_evolution`

```python
def plot_mdl_evolution(records: list[StepRecord]) -> Figure:
```

Line plot showing MDL score over time. Since only the selected cell's MDL score is recorded per step, plot one line per cell showing the MDL score at steps when that cell was selected (connected with lines). Cells that are never selected won't have data and are omitted from the legend.

X-axis: step. Y-axis: MDL score. Legend: cell labels. Lines color-coded by ability, with different line styles per level (solid for L1, dashed for L2, etc.).

Location: `src/repro_maa/visualize.py`

### Step 6: Implement `plot_cumulative_reward`

```python
def plot_cumulative_reward(
    curiosity_records: list[StepRecord],
    baseline_records: list[StepRecord] | None = None,
) -> Figure:
```

Simple two-line (or one-line) plot. X-axis: step. Y-axis: cumulative reward. One line per strategy with distinct colors and a legend ("Curiosity-Driven", "Fixed Curriculum"). Grid enabled for readability.

Location: `src/repro_maa/visualize.py`

### Step 7: Implement `plot_selection_heatmap`

```python
def plot_selection_heatmap(
    curiosity_records: list[StepRecord],
    baseline_records: list[StepRecord] | None = None,
) -> Figure:
```

Creates 1 or 2 subplots side by side. Each subplot uses `imshow` (or `pcolormesh`) to render a 3×5 heatmap where rows are abilities and columns are levels 1–5. Cell values are selection counts, annotated with the count number. Use a sequential colormap (e.g., `"YlOrRd"`). Subplot titles: "Curiosity-Driven" / "Fixed Curriculum". Axis labels: abilities on y-axis, levels on x-axis.

Location: `src/repro_maa/visualize.py`

### Step 8: Implement `generate_all` orchestrator

```python
def generate_all(
    curiosity_log_path: Path,
    baseline_log_path: Path | None = None,
    output_dir: Path = Path("plots"),
) -> list[Path]:
```

1. Read logs via `read_log()`.
2. Set matplotlib backend to `"Agg"`.
3. Call each plot function.
4. Save each figure to `output_dir/` with filenames:
   - `cell_selection_timeline.png`
   - `mdl_evolution.png`
   - `cumulative_reward.png`
   - `selection_heatmap.png`
5. Close all figures.
6. Return list of output paths.

Creates `output_dir` if it doesn't exist.

Location: `src/repro_maa/visualize.py`

### Step 9: Implement CLI entry point

Create `src/repro_maa/__main__.py` (or add a block to `visualize.py`) so that `python -m repro_maa.visualize` works.

CLI interface using `argparse`:
```
python -m repro_maa.visualize <curiosity_log> [--baseline <baseline_log>] [--output-dir <dir>]
```

Arguments:
- `curiosity_log` (positional, required) — path to the curiosity-driven JSONL log
- `--baseline` (optional) — path to the fixed-curriculum baseline JSONL log
- `--output-dir` (optional, default `"plots"`) — directory for output PNGs

The entry point calls `generate_all()` and prints the paths of created files.

Location: `src/repro_maa/visualize.py` (with `if __name__ == "__main__"` block) — invocable as `python -m repro_maa.visualize` via a thin `__main__.py` that imports and calls the CLI function.

### Step 10: Run tests and iterate

Run `pytest tests/test_visualize.py -v` and fix any failures. Verify all 8 tests pass. Then run the full suite (`pytest tests/ -m "not slow"`) to ensure no regressions.

## Dependencies

- **simulation_harness chunk** (complete): Provides `StepRecord`, `read_log()`, `write_log()`, `compare_runs()`, and the JSONL log format that this chunk consumes.
- **matplotlib** (new dependency): Must be added to `pyproject.toml`. Used for all plot generation.
- **numpy** (already in dependencies): Used for heatmap array construction.

## Risks and Open Questions

- **MDL evolution per non-selected cells**: The JSONL log only records the MDL score for the *selected* cell at each step. Full per-cell MDL trajectories would require re-computing MDL from `reward_history_summary`, but that summary only contains mean/count (not the full reward window needed for MDL). The plan addresses this by plotting MDL only at steps where a cell was selected — this still shows the trajectory, just with gaps. If this proves unsatisfying, we could reconstruct approximate scores from mean/count, but that's a stretch goal.
- **Large logs**: With thousands of steps, scatter plots may become dense. Consider downsampling or alpha transparency. This is an aesthetic concern and should be addressed during implementation if needed, not over-engineered upfront.
- **CLI entry point naming**: The GOAL says `python -m curiosity_stream.visualize` but the package is `repro_maa`. We'll implement `python -m repro_maa.visualize` to match the actual package name. The GOAL wording appears to use an older name for the package.

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