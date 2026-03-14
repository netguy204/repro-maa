<!--
This document captures HOW you'll achieve the chunk's GOAL.
It should be specific enough that each step is a reasonable unit of work
to hand to an agent.
-->

# Implementation Plan

## Approach

This chunk creates the foundational Python project scaffolding that every subsequent chunk in the `curiosity_stream_mvp` narrative depends on. The strategy is:

1. **uv-managed project with `pyproject.toml`** — Use uv as the package manager (consistent with modern Python workflows). The project's own source lives in `src/repro_maa/` using src-layout. The MAA submodule is not installed as a package; instead we add its paths to `sys.path` at import time via a thin compatibility shim.

2. **Compatibility shim for MAA imports** — The MAA codebase is not packaged (no `__init__.py` files, no setup.py). Rather than forking and restructuring it, we create a `src/repro_maa/maa_compat.py` module that handles `sys.path` manipulation and re-exports the generator classes and reward functions under clean names. This isolates the path-hacking to one file and gives downstream chunks stable import paths.

3. **Integration tests over the MAA boundary** — Per TESTING_PHILOSOPHY.md, "we test that we can import and invoke it, not that it's correct. Our boundary is the wrapper, not the internals." Tests will verify that each generator produces structurally valid output and each reward function accepts that output without error.

4. **Smoke test script** — A standalone script in `scripts/smoke_test.py` that exercises all generators and scorers end-to-end and prints a summary table. This is the human-readable proof that the scaffold works.

No new architectural decisions are introduced. If the decision to use uv + src-layout proves significant enough to warrant a DECISIONS.md entry, we'll propose it during implementation.

## Sequence

### Step 1: Create `pyproject.toml` with project metadata and dependencies

Create `pyproject.toml` at the repo root with:
- Project name: `repro-maa`
- Python requirement: `>=3.10`
- Dependencies:
  - `openai` — for the local LLM endpoint at `http://100.88.102.33:8000/v1` (downstream chunks)
  - `numpy` — used by MAA generators and our MDL scorer (future chunks)
  - `pytest` — test runner (dev dependency)
- Build system: hatchling (standard for src-layout)
- Package discovery pointing to `src/`

Run `uv sync` to create the virtual environment and lock file.

Location: `pyproject.toml`

### Step 2: Create the source package structure

Create the directory layout:

```
src/
  repro_maa/
    __init__.py        # Package marker, version string
    maa_compat.py      # MAA import compatibility shim
scripts/
  smoke_test.py        # (placeholder, filled in Step 6)
tests/
  __init__.py
  conftest.py          # Shared fixtures
```

Location: `src/repro_maa/`, `tests/`, `scripts/`

### Step 3: Implement the MAA compatibility shim

Create `src/repro_maa/maa_compat.py` that:

1. Computes the absolute paths to `Meta-Ability-Alignment/Data_Synthesis/` and `Meta-Ability-Alignment/Training/verl/utils/reward_score/` relative to the repo root.
2. Temporarily adds those paths to `sys.path` during import (using a context manager or guarded insertion).
3. Re-exports the key classes and functions:
   - `DeductionSampler` — alias for `Deduction.NestedLogicPuzzleSampler`
   - `DeductionFormatter` — alias for `Deduction.PuzzleFormatter`
   - `InductionGenerator` — alias for `Induction.SequencePuzzleGenerator`
   - `generate_abduction_problem` — from `Abduction`
   - `deduction_score` — alias for `formula.compute_score`
   - `induction_score` — alias for `squence.compute_score`
   - `abduction_score` — alias for `backward_reasoning.compute_score`
   - `mixed_score` — alias for `mix.compute_score`
4. Provides a `REPO_ROOT` constant for other modules to reference.

The shim must handle the case where MAA code has its own implicit dependencies (e.g., `random`, `re`, `itertools` from stdlib — these should already be available). If MAA code imports third-party libraries not in our deps, we discover and add them in this step.

Location: `src/repro_maa/maa_compat.py`

### Step 4: Write integration tests for data generators

Create `tests/test_integration_maa.py` with tests that verify success criteria #2:

- **`test_deduction_generator_produces_valid_problem`**: Instantiate `DeductionSampler(difficulty=1, seed=42)`, call `sample_unique(1)`, verify the result is a non-empty list of `(formulas, assignment)` tuples where formulas is a list of strings and assignment is a dict mapping variable names to booleans.
- **`test_induction_generator_produces_valid_problem`**: Instantiate `InductionGenerator(seed=42)`, call `generate_puzzles(num=1, level=1)`, verify the result is a list of dicts with keys `"question"`, `"answer"`, `"complete_sequence"`.
- **`test_abduction_generator_produces_valid_problem`**: Call `generate_abduction_problem(problem_id=1, num_goals=1, reachable_k=1, chain_depth=2, distractors=3, cycle_prob=0.1)`, verify the result is a dict with keys `"premises"`, `"known_atoms"`, `"goals"`, `"reachable_goals"`, `"unreachable_goals"`.
- **`test_each_generator_at_multiple_levels`**: Parametrized test across generators and levels 1–3 (levels 4–5 are slow for deduction). Marked `@pytest.mark.slow`.

Location: `tests/test_integration_maa.py`

### Step 5: Write integration tests for reward functions

Add tests to `tests/test_integration_maa.py` that verify success criteria #3:

- **`test_deduction_score_with_correct_answer`**: Build a synthetic `solution_str` with proper `<think>...</think><answer>...</answer>` tags containing a correct assignment, and a `ground_truth` dict with `solution_text_format` in the expected `"(1) A is True\n(2) B is False"` format. Call `deduction_score()` and assert score > 0 (expected: +3).
- **`test_deduction_score_with_wrong_answer`**: Same format but incorrect assignment. Assert score < 0.
- **`test_induction_score_with_correct_answer`**: Build solution_str with `<answer>42</answer>` and ground_truth with `solution_text_format: "42"`. Assert score > 0.
- **`test_abduction_score_with_correct_answer`**: Build solution_str with correct reachable/unreachable classification. Assert score > 0.
- **`test_mixed_score_routes_correctly`**: Call `mixed_score` with each of the three ground_truth formats and verify it returns the same score as the direct scorer.

These tests use synthetic inputs (not generated problems) to keep them fast and deterministic.

Location: `tests/test_integration_maa.py`

### Step 6: Create the smoke test script

Create `scripts/smoke_test.py` that:

1. Imports all generators and scorers via `repro_maa.maa_compat`.
2. For each ability (Deduction, Induction, Abduction) and a subset of levels (1–3 for speed):
   a. Generates one problem using the appropriate generator.
   b. Constructs a dummy response string (a plausible but likely wrong answer in the `<think>...</think><answer>...</answer>` format).
   c. Formats the ground truth into the dict format expected by the scorer.
   d. Calls the scorer with the dummy response and ground truth.
   e. Records the ability, level, and score.
3. Prints a summary table to stdout showing ability, level, and score for each cell.
4. Exits 0 if all cells produced a score (regardless of whether the dummy answer was correct — we're testing importability and invocation, not answer quality).

The script should be runnable as `python scripts/smoke_test.py` from the repo root.

Location: `scripts/smoke_test.py`

### Step 7: Run tests and validate

1. Run `python -m pytest tests/test_integration_maa.py -v` — all tests must pass.
2. Run `python scripts/smoke_test.py` — must exit 0 and print the summary table.
3. Run `ve validate` — must pass.

Fix any issues discovered during this step (missing dependencies, import errors, path issues).

Location: N/A (validation step)

## Dependencies

- **Meta-Ability-Alignment submodule** must be checked out and present at `Meta-Ability-Alignment/` in the repo root.
- **uv** must be installed on the system for `uv sync`.
- No other chunks need to be complete first — this is chunk 0 in the narrative.

## Risks and Open Questions

- **MAA implicit dependencies**: The MAA generator code may import third-party packages we haven't pinned (e.g., `sympy`, `tqdm`). We'll discover these during Step 3 when we first attempt the imports and add them to `pyproject.toml` as needed.
- **Abduction generator reliability**: `generate_abduction_problem` includes a consistency check (`check_consistency`) that brute-force enumerates 2^N valuations. At higher difficulty levels (N up to ~20 atoms), this can be slow. The smoke test should stick to levels 1–2 for abduction to avoid timeouts. If consistency checks fail (the function returns invalid puzzles and retries), we may need to add retry logic in the shim.
- **MAA code uses `print()` for debug output**: The reward scoring functions print debugging info to stdout. This is noisy but not harmful. We may want to capture/suppress this output in tests, but it's not a blocker.
- **`squence.py` filename typo**: The MAA codebase misspells "sequence" as "squence" in the filename. Our shim re-exports it as `induction_score` so downstream code never sees this quirk.

## Deviations

<!--
POPULATE DURING IMPLEMENTATION, not at planning time.
-->
