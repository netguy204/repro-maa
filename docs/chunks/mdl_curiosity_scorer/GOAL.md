---
status: ACTIVE
ticket: null
parent_chunk: null
code_paths:
- src/repro_maa/mdl_scorer.py
- src/repro_maa/__init__.py
- tests/test_mdl_scorer.py
code_references:
  - ref: src/repro_maa/mdl_scorer.py#MDLScorer
    implements: "MDL curiosity scorer class — computes curiosity score from reward windows"
  - ref: src/repro_maa/mdl_scorer.py#MDLScorer::score
    implements: "Public API: accepts list[float] rewards, returns non-negative curiosity score"
  - ref: src/repro_maa/mdl_scorer.py#MDLScorer::_unimodal_mdl
    implements: "Description length under single-Gaussian model (BIC-based)"
  - ref: src/repro_maa/mdl_scorer.py#MDLScorer::_bimodal_mdl
    implements: "Description length under bimodal Gaussian model with optimal split-point search"
  - ref: tests/test_mdl_scorer.py
    implements: "Unit tests covering all success criteria with synthetic reward distributions"
narrative: curiosity_stream_mvp
investigation: null
subsystems: []
friction_entries: []
bug_type: null
depends_on:
- taskcell_abstraction
created_after:
- scaffold_project
---

# Chunk Goal

## Minor Goal

Implement the MDL curiosity scorer — the core signal that drives curriculum selection. Given a window of recent reward outcomes for a task cell, compute the Minimum Description Length under a bimodal clustering model. This operationalizes the causal curiosity reward from Sontakke et al. (2021): −L(O|M) where O is the set of reward outcomes and M is a bimodal model.

The scorer should return high values for cells with structured-but-not-trivial reward distributions (the learning frontier — where the agent sometimes succeeds and sometimes fails) and low values for cells that are mastered (uniformly high rewards) or unreachable (uniformly low rewards). This directly implements the "MDL-based curiosity signal" required property from GOAL.md.

## Success Criteria

1. **MDLScorer class** accepts a list of float reward values and returns a float curiosity score.
2. **Mastered cells score low**: A window of uniform high rewards (e.g., all +3.0) produces a lower score than a mixed window.
3. **Unreachable cells score low**: A window of uniform low rewards (e.g., all −3.0) also produces a low score.
4. **Frontier cells score high**: A window with a structured mix of high and low rewards (e.g., 50/50 split) produces the highest score.
5. **Monotonic signal**: As a cell transitions from frontier (mixed) to mastered (all high), the score monotonically decreases.
6. **Handles edge cases**: Empty reward window returns a neutral/zero score. Single-element window returns a defined value.
7. **Unit tests** with synthetic reward distributions covering all the above cases. Tests use exact expected values where possible (the MDL computation is deterministic given inputs).