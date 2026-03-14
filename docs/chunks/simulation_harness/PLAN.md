<!--
This document captures HOW you'll achieve the chunk's GOAL.
It should be specific enough that each step is a reasonable unit of work
to hand to an agent.
-->

# Implementation Plan

## Approach

Build the simulation harness as a new module `src/repro_maa/simulation.py` that
composes the existing `CuriosityStream`, `TaskCell`, `MDLScorer`, and `BatchResult`
to run complete curriculum simulations with synthetic (and optionally live) agents.

**Key design choices:**

- **Agent as a protocol**: Define a minimal `Agent` protocol with a single method
  `respond(problem: dict) -> float` that returns a reward. `SyntheticAgent` implements
  this with a configurable solve-probability matrix; `LiveAgent` implements it by
  calling the local LLM endpoint. This lets `run_simulation` be agent-agnostic.

- **SyntheticAgent uses the MAA reward scheme directly**: Given a 3×5 probability
  matrix, for each problem it draws a Bernoulli trial and returns +3.0 (correct) or
  −3.0 (wrong). This matches the dominant reward signal from the MAA reward functions
  (format+answer correct vs. wrong). The solve probability can optionally improve
  after successful solves to simulate learning.

- **FixedCurriculumBaseline is a stream substitute**: Rather than subclassing
  `CuriosityStream`, implement `FixedCurriculumBaseline` as a simple iterator that
  yields batches from a predetermined ability/level schedule (level 1 for N steps,
  then level 2 for M steps). It produces `BatchResult`-compatible output so the
  same `run_simulation` loop works for both strategies.

- **JSONL stream log**: Each simulation step appends one JSON line containing all
  metadata from `BatchResult` plus the agent's rewards and cumulative reward. The
  log file is the primary output artifact — visualization (next chunk) reads it.

- **Comparison output**: A summary function reads two JSONL logs (curiosity vs.
  baseline) and computes per-step cumulative reward and cell selection frequency,
  printing a text summary.

- **Deterministic via explicit seeds**: `SyntheticAgent` takes its own seed for
  Bernoulli draws. The `CuriosityStream` seed controls cell selection. Combined
  with `TaskCell` seeds, the full simulation is reproducible.

**Testing strategy (per TESTING_PHILOSOPHY.md):**

TDD for the structural/mechanical parts: SyntheticAgent reward distribution,
`run_simulation` log schema, cumulative reward tallying, fixed baseline schedule
correctness, determinism. The research question ("does curiosity beat fixed?") is
NOT tested — it's verified by inspecting the output log. Tests use mock/synthetic
problem generation to avoid MAA generator overhead (per TESTING_PHILOSOPHY.md:
"Synthetic/fixture data for MAA generator outputs").

## Sequence

### Step 1: Define Agent protocol and SyntheticAgent

Create `src/repro_maa/simulation.py` with:

- Module-level backreference comment:
  `# Chunk: docs/chunks/simulation_harness - Simulation harness for curiosity stream`

- `Agent` typing protocol with method `respond(problem: dict) -> float`

- `SyntheticAgent` class:
  - Constructor: `(solve_matrix: dict[tuple[str,int], float], seed: int = 42,
    learning_rate: float = 0.0)` where `solve_matrix` maps `(ability, level)` to
    solve probability (0.0–1.0). `learning_rate` controls how much the probability
    increases after a successful solve (0.0 = no learning).
  - `respond(problem: dict) -> float`: Extracts ability and level from problem
    metadata, looks up solve probability, draws Bernoulli, returns +3.0 or −3.0.
    If learning is enabled and the solve succeeds, increments the probability by
    `learning_rate` (capped at 1.0).
  - `_rng: np.random.Generator` seeded for determinism.

Problems from `TaskCell.generate()` need ability/level metadata to route correctly.
Since `BatchResult` already carries ability and level, `run_simulation` will pass
this context when calling `agent.respond()`. Alternatively, augment the agent's
`respond` method to accept ability and level directly:
`respond(problem: dict, ability: str, level: int) -> float`.

Location: `src/repro_maa/simulation.py`

### Step 2: Write tests for SyntheticAgent (TDD red phase)

Create `tests/test_simulation.py` with failing tests:

1. **test_synthetic_agent_correct_reward_values**: With solve_prob=1.0, agent always
   returns +3.0. With solve_prob=0.0, always −3.0.

2. **test_synthetic_agent_stochastic**: With solve_prob=0.5 and seed, run 100 draws.
   Verify mean is near 0.0 (mix of +3 and −3) and both values appear.

3. **test_synthetic_agent_deterministic**: Same seed produces same reward sequence.

4. **test_synthetic_agent_learning**: With learning_rate=0.1 and initial
   solve_prob=0.5, after several successful solves the probability increases.
   Verify by running many trials — the fraction of +3.0 should increase over time.

5. **test_synthetic_agent_learning_caps_at_one**: With high learning_rate and many
   solves, probability never exceeds 1.0.

Location: `tests/test_simulation.py`

### Step 3: Implement SyntheticAgent (TDD green phase)

Implement the `SyntheticAgent` class to pass the Step 2 tests.

Location: `src/repro_maa/simulation.py`

### Step 4: Define stream log format and SimulationLog dataclass

Add to `src/repro_maa/simulation.py`:

- `StepRecord` dataclass containing:
  - `step: int`
  - `ability: str`
  - `level: int`
  - `mdl_score: float`
  - `selection_reason: str`
  - `batch_rewards: list[float]` — per-problem rewards from agent
  - `batch_mean_reward: float`
  - `cumulative_reward: float` — running total
  - `reward_history_summary: dict`

- `to_jsonl_line(record: StepRecord) -> str` — serialize one step to JSON

- `write_log(records: list[StepRecord], path: Path) -> None` — write JSONL file

- `read_log(path: Path) -> list[StepRecord]` — read JSONL file back

Location: `src/repro_maa/simulation.py`

### Step 5: Write tests for log serialization (TDD red phase)

Add to `tests/test_simulation.py`:

1. **test_step_record_roundtrip**: Create a `StepRecord`, write to JSONL, read back,
   verify all fields match.

2. **test_log_schema_completeness**: A serialized `StepRecord` JSON must contain
   every required key: step, ability, level, mdl_score, selection_reason,
   batch_rewards, batch_mean_reward, cumulative_reward.

3. **test_cumulative_reward_correctness**: Given a sequence of step records with
   known batch rewards, verify cumulative_reward is the running sum of
   batch_mean_reward × batch_size (or sum of batch_rewards).

Location: `tests/test_simulation.py`

### Step 6: Implement log serialization (TDD green phase)

Implement `StepRecord`, `to_jsonl_line`, `write_log`, `read_log` to pass Step 5 tests.

Location: `src/repro_maa/simulation.py`

### Step 7: Implement run_simulation

Add `run_simulation` function:

```python
def run_simulation(
    stream: CuriosityStream,
    agent: Agent,
    n_steps: int,
    log_path: Path | None = None,
) -> list[StepRecord]:
```

Loop for `n_steps`:
1. Call `stream.emit_batch()` → `BatchResult`
2. For each problem in the batch, call `agent.respond(problem, ability, level)` → reward
3. Call `stream.update(cell, rewards)` to feed rewards back to the curiosity scorer
4. Build a `StepRecord` with cumulative reward running total
5. Append to log

After loop, optionally write JSONL to `log_path`.

To update the stream with the correct cell, we need to look up the cell from
the batch result's ability and level. Add a helper `_find_cell(cells, ability, level)`.

Location: `src/repro_maa/simulation.py`

### Step 8: Write tests for run_simulation (TDD red phase)

Add to `tests/test_simulation.py`:

1. **test_run_simulation_returns_correct_count**: With n_steps=10, returns 10
   `StepRecord`s.

2. **test_run_simulation_steps_sequential**: Step numbers are 0, 1, ..., n-1.

3. **test_run_simulation_cumulative_monotonic_or_correct**: Cumulative reward at
   step k equals sum of all batch rewards from steps 0..k.

4. **test_run_simulation_deterministic**: Same seeds produce identical log.

5. **test_run_simulation_writes_log_file**: When `log_path` is provided, the JSONL
   file exists and contains `n_steps` lines.

These tests use `SyntheticAgent` with a simple solve matrix and mock out
`TaskCell.generate()` to avoid MAA generator calls (following TESTING_PHILOSOPHY.md
pattern of synthetic fixtures).

Location: `tests/test_simulation.py`

### Step 9: Implement run_simulation (TDD green phase)

Implement to pass Step 8 tests. Use monkeypatching or a thin wrapper to mock
`TaskCell.generate()` in tests — the simulation loop just needs dicts with
any content since `SyntheticAgent` only cares about ability/level.

Location: `src/repro_maa/simulation.py`

### Step 10: Implement FixedCurriculumBaseline

Add `FixedCurriculumBaseline` class:

- Constructor: `(cells: list[TaskCell], schedule: list[tuple[str, int, int]],
  batch_size: int = 8, seed: int = 42)` where `schedule` is a list of
  `(ability, level, n_steps)` tuples defining the curriculum.
  Default schedule: `[("deduction", 1, N//2), ("deduction", 2, N//2)]` etc.

- Provide a class method `maa_default(cells, n_steps)` that creates the MAA
  paper's schedule: each ability trains level 1 for `n_steps//2`, then level 2
  for the remainder (matching the 7B model's "level 1→2" approach).

- `emit_batch() -> BatchResult`: Emits the next batch from the current schedule
  position. Returns `BatchResult` with `selection_reason="fixed_schedule"` and
  `mdl_score=0.0` (no curiosity signal).

- Internal step counter and schedule pointer.

This class is compatible with `run_simulation` — it duck-types enough of the
`CuriosityStream` interface (`emit_batch()` and `update()`). `update()` is a
no-op since the fixed curriculum ignores reward feedback.

Location: `src/repro_maa/simulation.py`

### Step 11: Write tests for FixedCurriculumBaseline (TDD red phase)

Add to `tests/test_simulation.py`:

1. **test_fixed_baseline_follows_schedule**: With schedule `[("deduction", 1, 3),
   ("deduction", 2, 2)]`, the first 3 batches select deduction L1, the next 2
   select deduction L2.

2. **test_fixed_baseline_maa_default**: `maa_default(cells, 20)` produces a
   schedule covering all 3 abilities, levels 1 and 2.

3. **test_fixed_baseline_with_run_simulation**: `run_simulation` works with
   `FixedCurriculumBaseline` and produces a valid log.

Location: `tests/test_simulation.py`

### Step 12: Implement FixedCurriculumBaseline (TDD green phase)

Implement to pass Step 11 tests.

Location: `src/repro_maa/simulation.py`

### Step 13: Implement comparison summary

Add `compare_runs` function:

```python
def compare_runs(
    curiosity_log: list[StepRecord],
    baseline_log: list[StepRecord],
) -> dict:
```

Returns a dict with:
- `cumulative_reward_curiosity: list[float]` — per-step cumulative reward
- `cumulative_reward_baseline: list[float]`
- `cell_frequency_curiosity: dict[str, int]` — count of times each cell was selected
- `cell_frequency_baseline: dict[str, int]`
- `final_advantage: float` — curiosity final reward minus baseline final reward
- `summary_text: str` — human-readable summary

Location: `src/repro_maa/simulation.py`

### Step 14: Write tests for compare_runs (TDD red phase)

1. **test_compare_runs_structure**: Output dict has all required keys.

2. **test_compare_runs_correct_tallies**: With known logs, verify cell frequency
   counts and cumulative rewards match expectations.

Location: `tests/test_simulation.py`

### Step 15: Implement compare_runs (TDD green phase)

Location: `src/repro_maa/simulation.py`

### Step 16: Implement LiveAgent (optional/stretch)

Add `LiveAgent` class:

- Constructor: `(endpoint: str = "http://100.88.102.33:8000/v1", model: str = "default")`
- `respond(problem: dict, ability: str, level: int) -> float`:
  - Format the problem's `puzzle_text` as a prompt
  - Send to OpenAI-compatible completions API via the `openai` library
    (already a project dependency)
  - Extract the response text
  - Use `TaskCell.score(response, ground_truth)` to compute the reward
  - Requires `ground_truth` from the problem dict

Since `LiveAgent` needs both the problem and a way to score, it takes a
`cells` reference or a scoring function at construction time. Alternative:
accept the `TaskCell` in `respond` for scoring.

Location: `src/repro_maa/simulation.py`

### Step 17: Export from package and update GOAL.md code_paths

1. Add `SyntheticAgent`, `FixedCurriculumBaseline`, `run_simulation`,
   `compare_runs`, `StepRecord` to `src/repro_maa/__init__.py` exports.

2. Update `docs/chunks/simulation_harness/GOAL.md` frontmatter `code_paths`:
   - `src/repro_maa/simulation.py`
   - `src/repro_maa/__init__.py`
   - `tests/test_simulation.py`

Location: `src/repro_maa/__init__.py`, `docs/chunks/simulation_harness/GOAL.md`

### Step 18: Run full test suite

Run `pytest tests/ -v -m "not slow"` to verify all new and existing tests pass.
Fix any failures.

---

**BACKREFERENCE COMMENTS**

Add the following backreference at the module level of `src/repro_maa/simulation.py`:

```python
# Chunk: docs/chunks/simulation_harness - Simulation harness for curiosity stream
```

## Dependencies

- **stream_generator** (ACTIVE): Provides `CuriosityStream` and `BatchResult`
  classes at `src/repro_maa/stream.py`. The simulation loop calls `emit_batch()`
  and `update()`.
- **taskcell_abstraction** (ACTIVE): Provides `TaskCell` with `generate()` and
  `score()` methods. Used by `FixedCurriculumBaseline` and `LiveAgent`.
- **mdl_curiosity_scorer** (ACTIVE): Provides `MDLScorer`, required to construct
  `CuriosityStream`.
- **numpy**: Already a project dependency. Used for `SyntheticAgent` RNG and
  reward statistics.
- **openai**: Already a project dependency. Used by `LiveAgent` to call the
  local LLM endpoint.
- **json** (stdlib): For JSONL serialization.
- **No new external dependencies required.**

## Risks and Open Questions

- **Mock strategy for TaskCell.generate()**: Tests need to avoid calling real MAA
  generators (slow, requires submodule). Plan is to monkeypatch `generate()` to
  return synthetic problem dicts. If this is fragile, may need a dedicated test
  fixture or a `DummyTaskCell` subclass.

- **LiveAgent reliability**: The local LLM endpoint at `http://100.88.102.33:8000/v1`
  may not always be available. `LiveAgent` should handle connection errors
  gracefully (retry with backoff, or return a sentinel reward). Mark LiveAgent
  tests as `@pytest.mark.slow` and skip if endpoint unreachable.

- **FixedCurriculumBaseline schedule design**: The MAA paper's exact schedule for
  the 7B model is "level 1, then level 2" but doesn't specify exact step counts
  per ability. The `maa_default` class method will split steps evenly across
  abilities, which is an approximation. Document this assumption.

- **Cumulative reward interpretation**: The comparison is cumulative total reward,
  which favors strategies that pick easy problems. The more interesting metric may
  be reward-per-step on frontier cells. The comparison summary should include both.
  However, the core success criterion (#5) asks for cumulative reward comparison,
  so that's the primary metric.

## Deviations

<!--
POPULATE DURING IMPLEMENTATION, not at planning time.
-->