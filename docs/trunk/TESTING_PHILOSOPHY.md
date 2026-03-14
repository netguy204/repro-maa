# Testing Philosophy

This document establishes how we think about verification in this project.
It informs every chunk's testing strategy but doesn't prescribe specific tests.

## The Research Context

This is a research project exploring curiosity-driven curriculum selection for RL training. Unlike production software where the expected behavior is known before implementation, research code often discovers its correct behavior *through* implementation. This shapes our testing approach in specific ways:

- **Algorithmic building blocks have known behavior.** The MDL scorer, task cell wrappers, and stream generator all have well-defined input/output contracts. These get full TDD treatment.
- **Experimental outcomes don't.** Whether the curiosity-driven stream actually outperforms fixed curriculum is a research question, not a specification. We don't write tests asserting the answer — we write tests that ensure the *measurement apparatus* is correct.
- **The MAA submodule is not our code.** We test that we can import and invoke it, not that it's correct. Our boundary is the wrapper, not the internals.

## Testing Principles

### Test-Driven Development

We practice TDD for code with meaningful behavior:

1. **Write failing tests first** — Before writing implementation code, write tests that express what the code should do. These tests must fail initially.
2. **Write the implementation** — Write the minimum code necessary to make the tests pass.
3. **See previously failing tests succeed** — The same tests that failed now pass.

**When TDD applies cleanly:** Deterministic computations, data transformations, API contracts, reward scoring, stream selection logic, data format compliance.

**When TDD requires adaptation:** Exploratory analysis, visualization, simulation runs where the "correct" output is discovered through the work. In these cases:

- Write **structural tests** first: the function runs without error, returns the expected type/shape, respects invariants (e.g., probabilities sum to 1, selected cell is always from the valid set).
- Add **characterization tests** after: once you understand the behavior, pin it with a test that captures the now-known-correct output. This prevents regressions while acknowledging the answer wasn't known upfront.
- Document the gap: if a success criterion can't be TDD'd, note what you *did* test and what remains a manual/visual verification.

**When TDD doesn't apply:** Simple scaffolding code (dataclasses, enums, config structs) has no meaningful behavior to test. If the only failing test you can write is trivial (see Anti-Patterns below), skip the red phase entirely.

### Goal-Driven Test Design

Tests must assert semantically meaningful properties with respect to the goal. There must always be a clear relationship between:

- The success criteria in a chunk's GOAL.md
- The tests that verify those criteria

Each test should answer: "What success criterion does this test verify?" If the answer isn't clear, the test may not be valuable.

### Semantic Assertions Over Structural Assertions

**Avoid superficial assertions.** Tests that check types, property existence, or implementation details provide false confidence.

Bad:
```python
def test_mdl_scorer():
    scorer = MDLScorer()
    result = scorer.score(rewards)
    assert result is not None
    assert isinstance(result, float)
```

Good:
```python
def test_mdl_scorer_mastered_cell_scores_low():
    """A cell where the agent always succeeds should have low MDL."""
    scorer = MDLScorer()
    all_success = [3.0] * 20  # format_reward + answer_reward = +3
    score = scorer.score(all_success)
    assert score < scorer.score([3.0, -3.0] * 10)  # mixed should score higher
```

### Test Behavior at Boundaries

Prioritize testing:

- Empty states (no reward history yet, no problems generated)
- Edge cases from the domain (all 15 cells mastered, single cell available, reward window smaller than minimum)
- Error conditions (invalid difficulty level, generator produces no valid problems)

## Test Categories

### Unit Tests

Unit tests verify individual functions and classes in isolation.

- **Boundary**: A single function, method, or class
- **Dependencies**: Real implementations for pure computations. Synthetic/fixture data for MAA generator outputs (avoid calling generators in unit tests when a pre-built fixture suffices).
- **Location**: `tests/test_<module>.py`
- **Speed**: Each test under 1 second. No GPU, no model loading.

Examples: MDL computation correctness, stream selection logic, metadata serialization, reward aggregation.

### Integration Tests

Integration tests verify that our code correctly wraps and invokes the MAA submodule.

- **Boundary**: Our wrapper code + the MAA generators/reward functions
- **Dependencies**: Real MAA code, real filesystem for any generated data
- **Location**: `tests/test_integration_<feature>.py`
- **Speed**: May take several seconds per test (generators can be slow at high difficulty). Mark slow tests with `@pytest.mark.slow`.

Examples: TaskCell generates valid problems at each difficulty, reward functions accept generator output, smoke test end-to-end.

### Simulation Tests

Simulation tests verify that the stream generator and simulation harness produce well-formed output — not that the research hypothesis is correct.

- **Boundary**: The full pipeline from stream generator through simulation
- **Dependencies**: Synthetic agent model (configurable solve probability), real generators optional
- **Location**: `tests/test_simulation.py`
- **What they assert**: Stream log has expected schema, every emitted batch has valid metadata, cumulative rewards are monotonically computed, both curiosity and baseline strategies run to completion.
- **What they don't assert**: That curiosity beats baseline. That is a research finding, verified by inspection of the output artifacts, not by a pytest assertion.

### What We Don't Have

- **Property tests**: May be added later if the MDL scorer or stream selection develops complex invariants worth fuzzing
- **Performance tests**: No performance requirements — stream selection should be fast but we don't specify a bound
- **GPU tests**: All tests run on CPU. The MAA training pipeline requires GPU but our code doesn't.

## Hard-to-Test Properties

### "Does curiosity-driven selection actually work better?"

This is the central research question and is explicitly **not** tested by automated tests. Instead:

- The simulation harness produces structured logs
- The visualization chunk produces comparison plots
- A human inspects the output and draws conclusions

What *is* tested: that the measurement apparatus faithfully records what happened (correct reward tallying, correct cell attribution, no off-by-one in the comparison).

### Stochastic Behavior

The generators and stream selection involve randomness. To test deterministically:

- All random state flows through explicit seeds
- Tests that exercise stochastic code pass a fixed seed and assert on the deterministic output
- The `Reproducibility` required property (GOAL.md) is tested by running the same seed twice and asserting identical output

## What We Don't Test

- **MAA generator correctness**: We verify our wrappers work, not that the upstream puzzles are mathematically valid
- **Visualization aesthetics**: We verify plots are produced as files, not that they look good
- **Template prose**: Smoke test output format is tested structurally, not for exact wording
- **VeRL training integration**: Out of scope per GOAL.md — we produce the stream, not a trained model

## Anti-Pattern: Trivial Tests

A **trivial test** verifies something that cannot meaningfully fail. It tests the language or framework rather than the system's behavior.

A test is trivial if:

1. **It asserts that a value equals what was just assigned.** Setting `cell.ability = "deduction"` and asserting `cell.ability == "deduction"` tests Python's assignment operator.
2. **It cannot fail unless the runtime is broken.**
3. **It tests no transformation, computation, side effect, or rejection.**

The goal is signal, not coverage. Fewer meaningful tests beat a padded suite.

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures (synthetic rewards, dummy problems, seeds)
├── test_task_cell.py        # TaskCell wrapper unit tests
├── test_mdl_scorer.py       # MDL curiosity scorer unit tests
├── test_stream_generator.py # Stream selection logic unit tests
├── test_simulation.py       # Simulation harness and log format tests
├── test_integration_maa.py  # Integration tests for MAA generator/reward imports
└── test_smoke.py            # End-to-end smoke test
```

**Organization principles:**
- Mirror the source module structure
- Shared fixtures live in `conftest.py` — check before duplicating
- Mark slow tests with `@pytest.mark.slow` so fast iteration stays fast

## CI Requirements

```bash
# Fast suite (default, no slow marker)
pytest tests/ -m "not slow"

# Full suite (including MAA integration)
pytest tests/
```

Fast tests should complete in under 10 seconds. The full suite may take longer due to generator invocations but should stay under 60 seconds.
