---
status: ACTIVE
ticket: null
parent_chunk: null
code_paths:
- src/repro_maa/stream.py
- src/repro_maa/__init__.py
- tests/test_stream_generator.py
code_references:
  - ref: src/repro_maa/stream.py#BatchResult
    implements: "Dataclass carrying batch metadata (ability, level, MDL score, selection reason, step, reward history summary, problems)"
  - ref: src/repro_maa/stream.py#CuriosityStream
    implements: "Epsilon-greedy curriculum stream orchestrating TaskCell and MDLScorer"
  - ref: src/repro_maa/stream.py#CuriosityStream::select_cell
    implements: "Epsilon-greedy cell selection: greedy picks highest MDL, exploration picks random"
  - ref: src/repro_maa/stream.py#CuriosityStream::emit_batch
    implements: "Generates problem batch from selected cell with full metadata"
  - ref: src/repro_maa/stream.py#CuriosityStream::update
    implements: "Records new rewards into rolling window history for a cell"
  - ref: src/repro_maa/__init__.py
    implements: "Package exports for BatchResult and CuriosityStream"
  - ref: tests/test_stream_generator.py
    implements: "Unit tests for greedy/epsilon selection, metadata, rolling window, determinism"
narrative: curiosity_stream_mvp
investigation: null
subsystems: []
friction_entries: []
bug_type: null
depends_on:
- taskcell_abstraction
- mdl_curiosity_scorer
created_after:
- scaffold_project
---

# Chunk Goal

## Minor Goal

Build the stream generator that ties the task cells and MDL scorer together into a curiosity-driven curriculum policy. This is the central component of the project — the thing that actually produces the curiosity-driven training stream described in GOAL.md.

Each step of the stream generator: (1) computes the MDL curiosity score for all 15 task cells using their recent reward history, (2) selects the highest-signal cell, (3) samples a batch of problems from it, and (4) emits the batch with full metadata. The generator maintains a rolling reward history per cell and updates it as new rewards come in.

Supports both greedy selection (always pick highest MDL cell) and epsilon-greedy exploration (with probability ε, pick a random cell instead). This allows downstream experiments to compare pure exploitation of the curiosity signal vs. exploration.

## Success Criteria

1. **CuriosityStream class** that takes a list of TaskCells, an MDLScorer, and configuration (batch size, epsilon, reward window size, seed).
2. **`select_cell()`** returns the TaskCell with the highest MDL score, or a random cell with probability ε.
3. **`emit_batch()`** returns a batch of problems from the selected cell along with metadata: ability, level, MDL score, cell reward history summary, selection reason ("curiosity" or "exploration").
4. **`update(cell, rewards)`** records new reward outcomes for a cell, maintaining a rolling window.
5. **Full metadata on every batch**: ability type, difficulty level, MDL curiosity score, batch size, selection method, timestamp/step number.
6. **Deterministic** given seed: the same seed and reward history produce the same cell selection sequence.
7. **Unit tests** verify: greedy selection picks the highest-MDL cell, epsilon=0 is purely greedy, epsilon=1 is purely random, metadata schema is complete, reward history rolls correctly.