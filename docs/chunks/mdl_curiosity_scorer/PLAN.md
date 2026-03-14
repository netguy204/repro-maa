<!--
This document captures HOW you'll achieve the chunk's GOAL.
It should be specific enough that each step is a reasonable unit of work
to hand to an agent.
-->

# Implementation Plan

## Approach

Implement the MDL curiosity scorer as a pure-computation module with no dependency on TaskCell at runtime — it operates on plain `list[float]` reward windows. The core idea: compare the description length of the reward data under a **unimodal Gaussian model** (one cluster) vs. a **bimodal Gaussian model** (two clusters, hard assignment via optimal split). The curiosity score is the *improvement* in description length when switching from unimodal to bimodal — positive when the data genuinely has two modes (the learning frontier), zero when it doesn't (mastered or unreachable).

**Mathematical formulation (BIC-based MDL):**

For n reward values with maximum-likelihood Gaussian parameters:

- **Description length** = negative log-likelihood + model complexity penalty
  - Data cost: `(n/2)(1 + ln(2π σ²))` for n points with ML variance σ²
  - Model cost (BIC): `(k/2) ln(n)` where k = number of free parameters

- **Unimodal model** (k=2: μ, σ²):
  - `L_uni = (n/2)(1 + ln(2π σ²_all)) + ln(n)`

- **Bimodal model** (k=5: μ₁, σ₁², μ₂, σ₂², mixing weight π):
  - Hard clustering: sort rewards, try every possible split point, pick the one minimizing total description length
  - `L_bi = Σ_k [(n_k/2)(1 + ln(2π σ_k²))] + (5/2) ln(n)`
  - The split-point search costs O(n log n) for the sort + O(n) for the scan — lightweight

- **Curiosity score** = `max(0, L_uni − L_bi)`
  - Positive when bimodal explains the data better despite higher model cost
  - Zero when the data is already well-explained by a single Gaussian

**Zero-variance handling:** When all values in a cluster are identical (σ²=0), substitute a small epsilon (1e-10) to avoid `ln(0)`. This naturally yields very low data cost for uniform clusters, which is correct — uniform rewards are easy to describe.

**Edge cases:** Empty window → 0.0. Single element → 0.0 (cannot meaningfully compare models).

This approach is:
- **Fully deterministic** — no iterative EM, no random initialization. Given the same inputs, the output is identical.
- **Pure numpy** — uses only `numpy` (already a project dependency). No new dependencies needed.
- **Consistent with the testing philosophy** — the MDL computation is a deterministic algorithm with known correct behavior for synthetic inputs, making it ideal for TDD.

Tests follow TDD per `docs/trunk/TESTING_PHILOSOPHY.md`: write failing tests for each success criterion first, then implement the scorer to make them pass. Tests use synthetic reward distributions with hand-computable expected behavior (not exact float values, but relational assertions: frontier > mastered, frontier > unreachable, monotonic decrease as cell transitions to mastered).

## Sequence

### Step 1: Write failing tests for MDLScorer

Create `tests/test_mdl_scorer.py` with tests covering all seven success criteria from GOAL.md. Tests should import `MDLScorer` from `repro_maa.mdl_scorer` (which doesn't exist yet — tests will fail on import).

Test cases:

1. **`test_scorer_returns_float`** — `MDLScorer().score([1.0, 2.0, 3.0])` returns a float. *(Success criterion 1)*
2. **`test_mastered_cell_scores_low`** — Uniform high rewards `[3.0] * 20` scores lower than a mixed window `[3.0, -3.0] * 10`. *(Criterion 2)*
3. **`test_unreachable_cell_scores_low`** — Uniform low rewards `[-3.0] * 20` scores lower than a mixed window. *(Criterion 3)*
4. **`test_frontier_cell_scores_highest`** — A 50/50 mix of `+3.0` and `-3.0` scores higher than both uniform-high and uniform-low windows. *(Criterion 4)*
5. **`test_monotonic_decrease_toward_mastery`** — Start with 50% success, then 60%, 70%, 80%, 90%, 100%. Each step's score should be ≤ the previous. *(Criterion 5)*
6. **`test_empty_window_returns_zero`** — `score([])` returns `0.0`. *(Criterion 6)*
7. **`test_single_element_returns_zero`** — `score([3.0])` returns `0.0`. *(Criterion 6)*
8. **`test_two_element_identical_returns_zero`** — `score([3.0, 3.0])` returns `0.0`. *(Edge case: unimodal is optimal)*
9. **`test_two_element_different_returns_positive`** — `score([3.0, -3.0])` returns a positive value. *(Edge case: smallest bimodal window)*
10. **`test_deterministic`** — Calling `score()` twice with the same input returns identical values. *(Implied by deterministic design)*

Location: `tests/test_mdl_scorer.py`

### Step 2: Create MDLScorer class skeleton

Create `src/repro_maa/mdl_scorer.py` with the `MDLScorer` class and a `score()` method that raises `NotImplementedError`. This makes the tests importable (they should now fail on assertions rather than import errors).

Add a module-level backreference comment: `# Chunk: docs/chunks/mdl_curiosity_scorer - MDL-based curiosity signal for curriculum selection`

Location: `src/repro_maa/mdl_scorer.py`

### Step 3: Implement unimodal description length

Add a private method `_unimodal_mdl(self, rewards: np.ndarray) -> float` that computes the description length of the reward array under a single-Gaussian model:

```
n = len(rewards)
var = np.var(rewards)  # ML variance (not Bessel-corrected)
if var < EPSILON:
    var = EPSILON
data_cost = (n / 2) * (1 + np.log(2 * np.pi * var))
model_cost = np.log(n)  # (2/2) * ln(n)
return data_cost + model_cost
```

Location: `src/repro_maa/mdl_scorer.py`

### Step 4: Implement bimodal description length

Add a private method `_bimodal_mdl(self, rewards: np.ndarray) -> float` that:

1. Sorts the reward array
2. Iterates over all valid split points (index 1 through n-1), partitioning into left/right clusters
3. For each split, computes per-cluster data cost using the same Gaussian encoding
4. Adds the bimodal model cost: `(5/2) * ln(n)`
5. Returns the minimum total description length across all splits

Use cumulative sum/sum-of-squares for O(n) variance computation across all splits (avoid recomputing from scratch at each split point).

Location: `src/repro_maa/mdl_scorer.py`

### Step 5: Implement the score() method

Wire everything together in `score(self, rewards: list[float]) -> float`:

1. If `len(rewards) < 2`: return `0.0`
2. Convert to numpy array
3. Compute `l_uni = self._unimodal_mdl(arr)`
4. Compute `l_bi = self._bimodal_mdl(arr)`
5. Return `max(0.0, l_uni - l_bi)`

Run `pytest tests/test_mdl_scorer.py` — all tests should now pass.

Location: `src/repro_maa/mdl_scorer.py`

### Step 6: Export MDLScorer from package

Add `MDLScorer` to `src/repro_maa/__init__.py` exports alongside `TaskCell`.

Location: `src/repro_maa/__init__.py`

### Step 7: Final validation

Run the full test suite (`pytest tests/ -m "not slow"`) to confirm no regressions. Verify the MDL scorer tests all pass with exact expected behavior.

---

**BACKREFERENCE COMMENTS**

Add to `src/repro_maa/mdl_scorer.py` at module level:
```python
# Chunk: docs/chunks/mdl_curiosity_scorer - MDL-based curiosity signal for curriculum selection
```

Add to `tests/test_mdl_scorer.py` at module level:
```python
# Chunk: docs/chunks/mdl_curiosity_scorer - MDL curiosity scorer unit tests
```

## Dependencies

- **taskcell_abstraction** (ACTIVE): This chunk depends on the TaskCell abstraction existing to define the reward-value domain (floats from MAA reward functions). However, the MDLScorer itself has no runtime import dependency on TaskCell — it operates on plain `list[float]`. The dependency is conceptual: the rewards the scorer processes come from `TaskCell.score()`.
- **numpy**: Already in `pyproject.toml` dependencies. Used for efficient array operations in the MDL computation.
- No new external dependencies required.

## Risks and Open Questions

- **Monotonicity criterion (Success Criterion 5)**: The BIC-based MDL comparison is not analytically guaranteed to be monotonic as the success fraction increases continuously. However, for the discrete test case (stepping from 50% to 100% success in 10% increments with ±3.0 rewards), the signal should decrease monotonically because each step makes the data more unimodal. If edge cases arise where tiny floating-point differences break strict monotonicity, we may need to verify with ≤ (non-strict) rather than < (strict) comparisons in the test.
- **Minimum window size for meaningful signal**: With very small windows (2–3 rewards), the BIC model penalty dominates and the bimodal model is penalized heavily. This is actually correct behavior — we shouldn't trust a curiosity signal from 2 observations — but downstream consumers (the stream generator) should be aware that useful signal requires ~10+ rewards.
- **Reward value range assumption**: The scorer makes no assumption about the range of reward values. It works on any floats. The MAA reward functions produce values in roughly [-3, +3], but the scorer doesn't depend on this.

## Deviations

<!--
POPULATE DURING IMPLEMENTATION, not at planning time.
-->