---
status: ACTIVE
ticket: null
parent_chunk: null
code_paths:
  - pyproject.toml
  - src/repro_maa/__init__.py
  - src/repro_maa/maa_compat.py
  - tests/conftest.py
  - tests/test_integration_maa.py
  - scripts/smoke_test.py
code_references:
  - ref: pyproject.toml
    implements: "Project metadata, Python >=3.10 requirement, and dependency declarations (openai, numpy, pytest)"
  - ref: src/repro_maa/__init__.py
    implements: "Package public API surface: version string and re-exports of BatchResult, CuriosityStream, MDLScorer, TaskCell"
  - ref: src/repro_maa/maa_compat.py#REPO_ROOT
    implements: "Repository root path constant for locating MAA submodule"
  - ref: src/repro_maa/maa_compat.py#_temporary_sys_path
    implements: "Context manager isolating sys.path manipulation for MAA imports"
  - ref: src/repro_maa/maa_compat.py#_patch_main_for_abduction
    implements: "Stub injection for Abduction.py __main__ dependencies"
  - ref: src/repro_maa/maa_compat.py#_import_generators
    implements: "Lazy import of MAA data generator classes (Deduction, Induction, Abduction)"
  - ref: src/repro_maa/maa_compat.py#_import_scorers
    implements: "Lazy import of MAA reward scoring functions with verl stub modules to avoid torch dependency"
  - ref: src/repro_maa/maa_compat.py#DeductionSampler
    implements: "Public proxy to Deduction.NestedLogicPuzzleSampler"
  - ref: src/repro_maa/maa_compat.py#DeductionFormatter
    implements: "Public proxy to Deduction.PuzzleFormatter"
  - ref: src/repro_maa/maa_compat.py#InductionGenerator
    implements: "Public proxy to Induction.SequencePuzzleGenerator"
  - ref: src/repro_maa/maa_compat.py#generate_abduction_problem
    implements: "Public proxy to Abduction.generate_abduction_problem"
  - ref: src/repro_maa/maa_compat.py#deduction_score
    implements: "Public proxy to formula.compute_score"
  - ref: src/repro_maa/maa_compat.py#induction_score
    implements: "Public proxy to squence.compute_score"
  - ref: src/repro_maa/maa_compat.py#abduction_score
    implements: "Public proxy to backward_reasoning.compute_score"
  - ref: src/repro_maa/maa_compat.py#mixed_score
    implements: "Public proxy to mix.compute_score"
  - ref: tests/test_integration_maa.py#TestDeductionGenerator
    implements: "Integration tests verifying deduction generator produces valid problems"
  - ref: tests/test_integration_maa.py#TestInductionGenerator
    implements: "Integration tests verifying induction generator produces valid problems"
  - ref: tests/test_integration_maa.py#TestAbductionGenerator
    implements: "Integration tests verifying abduction generator produces valid problems"
  - ref: tests/test_integration_maa.py#TestDeductionScore
    implements: "Tests for deduction reward scoring with correct and wrong answers"
  - ref: tests/test_integration_maa.py#TestInductionScore
    implements: "Tests for induction reward scoring with correct and wrong answers"
  - ref: tests/test_integration_maa.py#TestAbductionScore
    implements: "Tests for abduction reward scoring with correct and wrong answers"
  - ref: tests/test_integration_maa.py#TestMixedScore
    implements: "Tests verifying mixed_score routes to correct ability-specific scorer"
  - ref: scripts/smoke_test.py#main
    implements: "Smoke test exercising all generators and scorers with summary table output"
narrative: curiosity_stream_mvp
investigation: null
subsystems: []
friction_entries: []
bug_type: null
depends_on: []
created_after: []
---
# Chunk Goal

## Minor Goal

Set up this repository as a working Python project that can import and exercise critical parts of the Meta-Ability-Alignment submodule. This is foundational infrastructure — every subsequent chunk in the `curiosity_stream_mvp` narrative (task cell abstraction, MDL scorer, stream generator, simulation, visualization) depends on being able to generate problems and compute rewards using the MAA codebase.

Concretely:
- Create a Python package structure (uv and `pyproject.toml`) with the project's own source directory.
- Pin dependencies required by the MAA training/data-synthesis code (and by our own code going forward), including `openai` for the local LLM endpoint at `http://100.88.102.33:8000/v1` (OpenAI-compatible API, no auth) which downstream chunks will use for live agent inference.
- Verify that the three data generators (`Data_Synthesis/Deduction.py`, `Data_Synthesis/Induction.py`, `Data_Synthesis/Abduction.py`) can be imported and produce sample problems.
- Verify that the reward scoring functions (`formula.py`, `backward_reasoning.py`, `squence.py`, `mix.py`) can be imported and invoked.
- Provide a smoke test script that exercises all of the above and exits cleanly.

## Success Criteria

1. **`pyproject.toml` exists** at the repo root with project metadata, Python ≥3.10 requirement, and dependencies declared.
2. **`python -m pytest tests/`** runs at least one passing test that imports each of the three MAA data generators and produces at least one valid problem per generator.
3. **Reward functions importable**: A test imports `formula.compute_score`, `backward_reasoning.compute_score`, and `squence.compute_score` and calls each with a synthetic input without error.
4. **Smoke test script**: `python scripts/smoke_test.py` generates one problem per ability/level, scores a dummy response against each, and prints a summary table — exits 0.
5. **`ve validate` passes** after all changes.