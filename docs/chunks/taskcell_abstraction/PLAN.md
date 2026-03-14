# Implementation Plan

## Approach

Build a `TaskCell` class in `src/repro_maa/task_cell.py` that wraps the three MAA generator/scorer pairs behind a uniform interface. The key challenge is that each ability has a structurally different generator API and output format:

- **Deduction**: `DeductionSampler(difficulty, seed).sample_unique(n)` → list of `(formulas, assignment)` tuples; requires `DeductionFormatter` to produce human-readable text and solution text.
- **Induction**: `InductionGenerator(seed).generate_puzzles(num, level)` → list of dicts with `puzzle_text`, `solution_text` (int), `complete_sequence`.
- **Abduction**: `generate_abduction_problem(problem_id, num_goals, ...)` → dict with `premises`, `goals`, `reachable_goals`, `unreachable_goals`; requires formatting to produce puzzle text and solution text.

The strategy is to use an **adapter pattern**: a private `_generate_one_*` method per ability normalizes each generator's output into a common problem dict `{"puzzle_text": str, "ground_truth": dict}`, and `score()` dispatches to the correct MAA reward function. The `ground_truth` dict always contains the key `"solution_text_format"` expected by the MAA scorers.

For determinism (success criterion #5), all random state flows through an explicit seed parameter on `TaskCell.__init__`. The seed is used to construct generators and to derive per-call seeds for `generate(n)` calls.

Tests follow TDD per `docs/trunk/TESTING_PHILOSOPHY.md`:
- **Unit tests** (`tests/test_task_cell.py`) verify the TaskCell contract with synthetic/fixture data where possible, and real generators for integration coverage.
- **Integration tests** are marked `@pytest.mark.slow` since generators at higher levels can be slow.

The existing `maa_compat.py` shim (from `scaffold_project`) provides all the imports we need. TaskCell builds on top of it — it does not duplicate any import-path manipulation.

## Sequence

### Step 1: Write failing unit tests for TaskCell

Create `tests/test_task_cell.py` with tests that express the success criteria before writing implementation code. Tests to write:

1. **Construction**: `TaskCell("deduction", 1)` constructs without error; invalid ability or level raises `ValueError`.
2. **Generate contract**: `cell.generate(n)` returns a list of exactly `n` dicts, each with `"puzzle_text"` (str, non-empty) and `"ground_truth"` (dict with `"solution_text_format"` key).
3. **Score contract**: `cell.score(response_str, ground_truth_dict)` returns a float. A correct response scores positive; an incorrect response scores negative.
4. **All 15 cells**: Parametrized test over all 3 abilities × 5 levels verifying `generate(1)` succeeds for each. Mark `@pytest.mark.slow`.
5. **Determinism**: Two `TaskCell` instances with the same `(ability, level, seed)` produce identical problems from `generate(n)`.

Location: `tests/test_task_cell.py`

### Step 2: Define the TaskCell class skeleton

Create `src/repro_maa/task_cell.py` with:

- `TaskCell.__init__(self, ability: str, level: int, seed: int = 42)` — validates ability is one of `("deduction", "induction", "abduction")` and level is in `range(1, 6)`. Stores ability, level, seed.
- `TaskCell.generate(self, n: int) -> list[dict]` — stub that raises `NotImplementedError`.
- `TaskCell.score(self, response: str, ground_truth: dict) -> float` — stub that raises `NotImplementedError`.
- `TaskCell.__repr__` for debugging: `TaskCell(ability='deduction', level=1)`.

Add module-level backreference: `# Chunk: docs/chunks/taskcell_abstraction - Unified TaskCell abstraction`

Location: `src/repro_maa/task_cell.py`

### Step 3: Implement the difficulty parameter mapping for Abduction

Define a module-level constant `_ABDUCTION_LEVEL_PARAMS` mapping level (1–5) to the `generate_abduction_problem` keyword arguments. The smoke test uses `num_goals=max(1, level)`, `reachable_k=1`, `chain_depth=level+1`, `distractors=3`, `cycle_prob=0.1` — we follow that pattern but extend to levels 4–5:

```python
_ABDUCTION_LEVEL_PARAMS = {
    1: dict(num_goals=1, reachable_k=1, chain_depth=2, distractors=3, cycle_prob=0.1),
    2: dict(num_goals=2, reachable_k=1, chain_depth=3, distractors=3, cycle_prob=0.1),
    3: dict(num_goals=3, reachable_k=1, chain_depth=4, distractors=3, cycle_prob=0.1),
    4: dict(num_goals=4, reachable_k=1, chain_depth=5, distractors=3, cycle_prob=0.1),
    5: dict(num_goals=5, reachable_k=1, chain_depth=6, distractors=3, cycle_prob=0.1),
}
```

Location: `src/repro_maa/task_cell.py`

### Step 4: Implement generate() for each ability

Implement three private methods that normalize each generator's output to the common format `{"puzzle_text": str, "ground_truth": {"solution_text_format": ...}}`:

**`_generate_deduction(self, n: int) -> list[dict]`**:
- Create `DeductionSampler(difficulty=self.level, seed=self._seed)`.
- Call `sampler.sample_unique(n)` to get `(formulas, assignment)` tuples.
- For each tuple, create a `DeductionFormatter(formulas, assignment)`.
- Return `{"puzzle_text": fmt.puzzle_text(), "ground_truth": {"solution_text_format": fmt.solution_text()}}`.

**`_generate_induction(self, n: int) -> list[dict]`**:
- Create `InductionGenerator(seed=self._seed)`.
- Call `gen.generate_puzzles(num=n, level=self.level)`.
- Return `{"puzzle_text": p["puzzle_text"], "ground_truth": {"solution_text_format": p["solution_text"]}}` for each puzzle.

**`_generate_abduction(self, n: int) -> list[dict]`**:
- For `i` in `range(n)`, call `generate_abduction_problem(problem_id=seed_offset+i, **_ABDUCTION_LEVEL_PARAMS[self.level])`.
- Format `puzzle_text` from premises, known_atoms, and goals (similar to how the MAA training data formats it — a readable prompt listing premises and asking which goals are reachable).
- Format `ground_truth["solution_text_format"]` as numbered lines: `"(1) GoalA is reachable\n(2) GoalB is unreachable\n..."` — matching the format expected by `backward_reasoning.compute_score`.
- Return the list of problem dicts.

Wire `generate()` to dispatch to the correct private method based on `self.ability`.

Location: `src/repro_maa/task_cell.py`

### Step 5: Implement score()

Implement `TaskCell.score(self, response: str, ground_truth: dict) -> float`:

- Dispatch based on `self.ability`:
  - `"deduction"` → `deduction_score(response, ground_truth)`
  - `"induction"` → `induction_score(response, ground_truth)`
  - `"abduction"` → `abduction_score(response, ground_truth)`
- Return the float result.

The `response` parameter is expected to already be in the MAA format: `"Assistant: <think>...</think><answer>...</answer>"`. TaskCell does not wrap responses — that is the caller's responsibility (consistent with how the existing integration tests work).

Location: `src/repro_maa/task_cell.py`

### Step 6: Export TaskCell from the package

Add `TaskCell` to `src/repro_maa/__init__.py` exports so downstream code can `from repro_maa import TaskCell`.

Location: `src/repro_maa/__init__.py`

### Step 7: Run tests and iterate

Run `pytest tests/test_task_cell.py -v` to verify all unit tests pass. Then run the full suite with `pytest tests/ -v` to ensure no regressions. Fix any failures.

For the `@pytest.mark.slow` parametrized tests across all 15 cells, verify with `pytest tests/test_task_cell.py -v -m slow`. These tests exercise the real MAA generators at levels 1–5, confirming that every cell in the 3×5 grid produces valid output.

### Step 8: Verify determinism

Run the determinism test explicitly: create two `TaskCell` instances with identical `(ability, level, seed)`, call `generate(3)` on each, and assert the returned problem lists are identical. This is already covered in Step 1's test #5, but verify it passes for all three abilities.

## Dependencies

- **scaffold_project** (ACTIVE): Provides the `maa_compat.py` shim with all generator/scorer imports. Must be complete before this chunk can run.
- **No new external libraries**: TaskCell uses only the existing `repro_maa.maa_compat` exports and Python stdlib.

## Risks and Open Questions

- **Abduction puzzle_text formatting**: The MAA codebase doesn't have a standard formatter for Abduction problems the way Deduction has `PuzzleFormatter`. We need to construct the puzzle text ourselves from `premises`, `known_atoms`, and `goals`. The exact format doesn't matter for this abstraction layer (downstream consumers use `puzzle_text` as a prompt), but it should be human-readable and complete enough to solve.
- **Abduction difficulty scaling at levels 4–5**: The smoke test only exercises levels 1–3. Levels 4–5 with more goals and deeper chain depths may be slow or produce degenerate problems. If generation fails or is prohibitively slow at high levels, we may need to adjust the parameter mapping. The `@pytest.mark.slow` marker on high-level tests protects the fast test suite.
- **Abduction seeding**: Unlike Deduction and Induction generators, `generate_abduction_problem` doesn't take a `seed` parameter. Determinism may require setting `random.seed()` before each call. Need to verify whether the function uses `random` module internally.
- **Deduction `sample_unique(n)` for large n**: May fail if there aren't enough unique puzzles at a given difficulty. We should document this limitation but not over-engineer around it — the typical use case is small batches (1–32).

## Deviations

<!--
POPULATE DURING IMPLEMENTATION, not at planning time.
-->
