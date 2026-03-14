<!--
This document captures HOW you'll achieve the chunk's GOAL.
It should be specific enough that each step is a reasonable unit of work
to hand to an agent.
-->

# Implementation Plan

## Approach

Build a `CuriosityStream` class in a new `src/repro_maa/stream.py` module that
orchestrates the existing `TaskCell` and `MDLScorer` to implement the
curiosity-driven curriculum policy. The class follows the same patterns as the
existing codebase: pure Python with numpy, explicit seed-based determinism,
clean public API with private helpers.

**Key design choices:**

- **Composition over inheritance**: `CuriosityStream` takes a list of `TaskCell`
  instances and an `MDLScorer` instance, keeping those layers decoupled.
- **Internal reward history**: A `dict[tuple[str,int], collections.deque]`
  mapping `(ability, level)` to a rolling window of reward floats. Using
  `deque(maxlen=window_size)` gives O(1) append with automatic eviction.
- **numpy.random.Generator for selection RNG**: Separate from TaskCell seeds.
  The stream's seed controls cell selection order; each TaskCell's own seed
  controls problem generation. This preserves determinism at both layers.
- **Epsilon-greedy as a simple branch**: With probability ε, pick a uniform
  random cell; otherwise pick the argmax MDL cell. Ties broken by RNG for
  determinism.
- **Rich batch metadata dataclass**: A `BatchResult` dataclass carries all
  required metadata fields. This gives type safety and makes downstream
  consumers (simulation harness, visualization) easy to build.

**Testing strategy (per TESTING_PHILOSOPHY.md):**

TDD applies cleanly here — the stream generator has deterministic, well-defined
behavior: given a seed and reward history, the selection sequence is fixed.
Tests will be written first for each success criterion using synthetic reward
histories (no MAA generator calls needed). Tests use semantic assertions
(greedy picks highest MDL, epsilon=1 is uniform, metadata fields present and
correct) rather than structural ones.

## Subsystem Considerations

No subsystems are documented yet. This chunk does not introduce cross-cutting
patterns that warrant subsystem documentation — it is a single module composing
the existing TaskCell and MDLScorer layers.

## Sequence

### Step 1: Define the BatchResult dataclass and CuriosityStream skeleton

Create `src/repro_maa/stream.py` with:

- A `BatchResult` dataclass holding all required metadata fields:
  - `ability: str` — the meta-ability type ("deduction", "induction", "abduction")
  - `level: int` — difficulty level (1–5)
  - `mdl_score: float` — the MDL curiosity score that drove selection
  - `selection_reason: str` — either `"curiosity"` (greedy) or `"exploration"` (epsilon)
  - `batch_size: int` — number of problems in this batch
  - `step: int` — monotonically increasing step counter
  - `reward_history_summary: dict` — per-cell summary (mean, count) at time of selection
  - `problems: list[dict]` — the actual problem dicts from `TaskCell.generate()`

- The `CuriosityStream.__init__` signature:
  - `cells: list[TaskCell]` — the 15 task cells
  - `scorer: MDLScorer` — the MDL scorer instance
  - `batch_size: int = 8`
  - `epsilon: float = 0.0` — exploration probability (0 = pure greedy)
  - `window_size: int = 20` — rolling reward history window per cell
  - `seed: int = 42`

- Internal state:
  - `_history: dict[tuple[str,int], deque[float]]` — rolling reward window per cell
  - `_rng: numpy.random.Generator` — seeded RNG for selection
  - `_step: int` — step counter starting at 0

Location: `src/repro_maa/stream.py`

### Step 2: Write tests for select_cell (TDD red phase)

Create `tests/test_stream_generator.py` with failing tests:

1. **test_greedy_selects_highest_mdl**: Seed reward histories so one cell has
   a clearly higher MDL score (mixed rewards) while others are uniform. With
   `epsilon=0`, `select_cell()` must return that cell.

2. **test_epsilon_zero_is_purely_greedy**: With `epsilon=0`, run `select_cell()`
   many times — it must never select a non-maximal cell.

3. **test_epsilon_one_is_purely_random**: With `epsilon=1`, run `select_cell()`
   many times with a fixed seed — verify the selection covers multiple cells
   (not always the max MDL cell).

4. **test_select_cell_deterministic**: Same seed and reward history produces the
   same selection sequence across two independent runs.

Tests use synthetic reward histories injected directly into `_history` (no
generator calls needed).

Location: `tests/test_stream_generator.py`

### Step 3: Implement select_cell

Add the `select_cell()` method to `CuriosityStream`:

1. Compute `MDLScorer.score(list(self._history[key]))` for every cell.
2. Draw from `self._rng.random()`: if < ε, pick a uniform random cell index
   via `self._rng.integers(len(self._cells))` → return `("exploration", cell)`.
3. Otherwise, find the cell with the max MDL score. Break ties using
   `self._rng.integers()` over the tied subset → return `("curiosity", cell)`.
4. Return a `(reason, cell, mdl_score)` tuple.

Verify Step 2 tests pass (TDD green phase).

Location: `src/repro_maa/stream.py`

### Step 4: Write tests for emit_batch (TDD red phase)

Add tests to `tests/test_stream_generator.py`:

1. **test_emit_batch_returns_batch_result**: Call `emit_batch()`, assert it
   returns a `BatchResult` with all required fields populated.

2. **test_emit_batch_metadata_complete**: Verify every field in `BatchResult`
   is present and has the correct type: ability is a string, level is int 1–5,
   mdl_score is non-negative float, selection_reason is "curiosity" or
   "exploration", batch_size matches the configured value, step increments
   monotonically, problems is a list of the right length.

3. **test_emit_batch_step_increments**: Call `emit_batch()` three times, verify
   step values are 0, 1, 2.

Location: `tests/test_stream_generator.py`

### Step 5: Implement emit_batch

Add `emit_batch()` to `CuriosityStream`:

1. Call `self.select_cell()` to get `(reason, cell, mdl_score)`.
2. Call `cell.generate(self._batch_size)` to get the problem batch.
3. Build `reward_history_summary`: for each cell key, compute
   `{"mean": mean(history), "count": len(history)}`.
4. Construct and return a `BatchResult` with all metadata fields.
5. Increment `self._step`.

Verify Step 4 tests pass.

Location: `src/repro_maa/stream.py`

### Step 6: Write tests for update (TDD red phase)

Add tests to `tests/test_stream_generator.py`:

1. **test_update_records_rewards**: Call `update(cell, [1.0, 2.0])`, verify the
   cell's history contains those values.

2. **test_update_rolling_window**: Set `window_size=5`, push 7 rewards — verify
   only the last 5 are retained.

3. **test_update_affects_selection**: Start with all cells having uniform
   histories (low MDL). Update one cell with mixed rewards. Verify
   `select_cell()` now picks that cell (with epsilon=0).

Location: `tests/test_stream_generator.py`

### Step 7: Implement update

Add `update(cell: TaskCell, rewards: list[float])` to `CuriosityStream`:

1. Look up `key = (cell.ability, cell.level)`.
2. Extend `self._history[key]` with the new rewards. The `deque(maxlen=...)`
   automatically evicts old entries beyond `window_size`.

Verify Step 6 tests pass.

Location: `src/repro_maa/stream.py`

### Step 8: Write determinism test (TDD red phase)

Add test to `tests/test_stream_generator.py`:

1. **test_full_sequence_deterministic**: Create two `CuriosityStream` instances
   with the same seed, same cells, same initial reward histories. Run a
   sequence of `emit_batch()` + `update()` cycles. Assert that both produce
   identical `BatchResult` sequences (same cell selections, same step numbers,
   same MDL scores).

Location: `tests/test_stream_generator.py`

### Step 9: Implement determinism and verify

This should already work from the seeded RNG in Step 3. Run the test from
Step 8 and verify it passes. If not, debug seed propagation.

### Step 10: Export from package and update GOAL.md code_paths

1. Add `CuriosityStream` and `BatchResult` to `src/repro_maa/__init__.py`
   exports.
2. Update `docs/chunks/stream_generator/GOAL.md` frontmatter `code_paths` with:
   - `src/repro_maa/stream.py`
   - `tests/test_stream_generator.py`
   - `src/repro_maa/__init__.py`

Location: `src/repro_maa/__init__.py`, `docs/chunks/stream_generator/GOAL.md`

### Step 11: Run full test suite

Run `pytest tests/ -v` to verify all existing tests still pass and all new
tests pass. Fix any issues.

---

**BACKREFERENCE COMMENTS**

Add the following backreference at the module level of `src/repro_maa/stream.py`:

```python
# Chunk: docs/chunks/stream_generator - Curiosity-driven curriculum stream generator
```

## Dependencies

- **taskcell_abstraction** (ACTIVE): Provides `TaskCell` class with `generate(n)`
  and `score(response, ground_truth)` API. Located at `src/repro_maa/task_cell.py`.
- **mdl_curiosity_scorer** (ACTIVE): Provides `MDLScorer` class with
  `score(rewards) -> float` API. Located at `src/repro_maa/mdl_scorer.py`.
- **numpy**: Already a project dependency (used by MDLScorer). Used here for
  `numpy.random.Generator` seeded RNG.
- **No new external dependencies required.**

## Risks and Open Questions

- **Cold start**: When reward history is empty for all cells, MDLScorer returns
  0.0 for every cell. The initial selection will be effectively random (tie-
  breaking via RNG). This is acceptable — the simulation harness will seed
  initial histories. Document this behavior in the docstring.
- **All cells at zero MDL**: If all cells are mastered or unreachable, every
  MDL score is 0.0. Greedy selection degenerates to random tie-breaking. This
  is correct behavior (no cell has signal), but worth documenting.
- **TaskCell.generate() speed**: Some cells (especially high-level abduction)
  may be slow. The stream generator itself adds negligible overhead — the
  bottleneck is the underlying MAA generator. Not a risk for this chunk (tests
  use synthetic histories), but the simulation_harness chunk will need to
  account for this.

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