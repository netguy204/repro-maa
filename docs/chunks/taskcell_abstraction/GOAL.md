---
status: ACTIVE
ticket: null
parent_chunk: null
code_paths:
- src/repro_maa/task_cell.py
- src/repro_maa/__init__.py
- tests/test_task_cell.py
code_references:
- ref: src/repro_maa/task_cell.py#TaskCell
  implements: "Unified abstraction over the 3×5 ability/difficulty grid with generate() and score() interface"
- ref: src/repro_maa/task_cell.py#TaskCell::generate
  implements: "Dispatches to ability-specific generators, returns uniform problem dicts with puzzle_text and ground_truth"
- ref: src/repro_maa/task_cell.py#TaskCell::score
  implements: "Dispatches to correct MAA reward function (deduction_score, induction_score, abduction_score)"
- ref: src/repro_maa/task_cell.py#TaskCell::_generate_deduction
  implements: "Deduction adapter: DeductionSampler + DeductionFormatter → common problem dict format"
- ref: src/repro_maa/task_cell.py#TaskCell::_generate_induction
  implements: "Induction adapter: InductionGenerator.generate_puzzles → common problem dict format"
- ref: src/repro_maa/task_cell.py#TaskCell::_generate_abduction
  implements: "Abduction adapter: generate_abduction_problem with seeded random state → common problem dict format"
- ref: src/repro_maa/task_cell.py#_ABDUCTION_LEVEL_PARAMS
  implements: "Difficulty parameter mapping for abduction levels 1–5 (num_goals, chain_depth, etc.)"
- ref: src/repro_maa/task_cell.py#_format_abduction_puzzle
  implements: "Human-readable puzzle text formatting for abduction problems"
- ref: src/repro_maa/task_cell.py#_format_abduction_solution
  implements: "Ground truth solution formatting matching backward_reasoning.compute_score expected format"
- ref: tests/test_task_cell.py
  implements: "Unit and integration tests for TaskCell construction, generate/score contracts, determinism, and all 15 cells"
narrative: curiosity_stream_mvp
investigation: null
subsystems: []
friction_entries: []
bug_type: null
depends_on:
- scaffold_project
created_after:
- scaffold_project
---

# Chunk Goal

## Minor Goal

Wrap the three MAA data generators (Deduction, Induction, Abduction) and their corresponding reward functions in a unified `TaskCell` abstraction. Each cell represents one slot in the 3×5 ability/difficulty grid and exposes two operations: `generate(n)` to produce n problems at the cell's difficulty level, and `score(response, ground_truth)` to compute the reward using the existing MAA reward functions (`formula.py`, `backward_reasoning.py`, `squence.py`).

This is the foundational data layer for the curiosity stream. Every downstream chunk (MDL scorer, stream generator, simulation) operates on TaskCells rather than calling the raw MAA generators directly. The abstraction isolates the rest of the system from the MAA codebase's internal structure (different generator classes, different reward function signatures, different output formats per ability).

## Success Criteria

1. **TaskCell class exists** with a uniform interface: `TaskCell(ability: str, level: int)` where ability is one of "deduction", "induction", "abduction" and level is 1–5.
2. **`generate(n)`** returns n problem dicts, each containing at minimum `puzzle_text` and `ground_truth` fields, for all 15 cells (3 abilities × 5 levels).
3. **`score(response, ground_truth)`** invokes the correct MAA reward function for the cell's ability and returns a float reward value. Deduction uses `formula.compute_score`, Induction uses `squence.compute_score`, Abduction uses `backward_reasoning.compute_score`.
4. **Unit tests** verify that every cell can generate at least one problem and that scoring a synthetic correct/incorrect response produces expected reward signs (+reward for correct, −reward for incorrect).
5. **Deterministic**: Given the same seed, `generate(n)` produces identical problems.