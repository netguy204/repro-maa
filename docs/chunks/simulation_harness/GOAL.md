---
status: ACTIVE
ticket: null
parent_chunk: null
code_paths:
- src/repro_maa/simulation.py
- src/repro_maa/__init__.py
- tests/test_simulation.py
code_references:
  - ref: src/repro_maa/simulation.py#Agent
    implements: "Minimal agent protocol with respond(problem, ability, level) → float"
  - ref: src/repro_maa/simulation.py#SyntheticAgent
    implements: "Configurable synthetic agent with Bernoulli solve probabilities, MAA reward scheme, and optional learning"
  - ref: src/repro_maa/simulation.py#StepRecord
    implements: "Structured JSONL log format for simulation step records"
  - ref: src/repro_maa/simulation.py#to_jsonl_line
    implements: "JSONL serialization for step records"
  - ref: src/repro_maa/simulation.py#write_log
    implements: "Write simulation log to JSONL file"
  - ref: src/repro_maa/simulation.py#read_log
    implements: "Read simulation log from JSONL file"
  - ref: src/repro_maa/simulation.py#FixedCurriculumBaseline
    implements: "MAA paper's fixed ascending curriculum for baseline comparison"
  - ref: src/repro_maa/simulation.py#run_simulation
    implements: "Core simulation loop: emit batch → agent scores → update stream → log"
  - ref: src/repro_maa/simulation.py#compare_runs
    implements: "Comparison output with cumulative rewards, cell frequencies, and summary text"
  - ref: src/repro_maa/simulation.py#LiveAgent
    implements: "Optional agent variant calling local LLM endpoint for real responses"
  - ref: tests/test_simulation.py
    implements: "Tests for simulation harness: agent rewards, log schema, determinism, baseline schedule"
narrative: curiosity_stream_mvp
investigation: null
subsystems: []
friction_entries: []
bug_type: null
depends_on:
- stream_generator
created_after:
- scaffold_project
---

# Chunk Goal

## Minor Goal

Create a simulation harness that runs the stream generator against a synthetic agent model to produce a full stream trace, then compares it against the MAA paper's fixed ascending curriculum. This delivers Success Criteria #3 (stream analysis artifacts) and #4 (baseline comparison) from GOAL.md.

The synthetic agent is a configurable function that maps (ability, difficulty_level) → solve_probability. This lets us simulate what happens when a "learner" of known capability encounters the curiosity-driven stream vs. the fixed curriculum — without requiring a GPU or actual model training. The agent's solve probability can optionally improve over time (simulating learning) to test whether the curiosity stream adapts to a shifting frontier.

A local LLM is available at `http://100.88.102.33:8000/v1` (OpenAI-compatible completions API, no auth required). This can be used to implement a `LiveAgent` variant that sends actual prompts to the model and scores real responses — providing a more realistic alternative to the synthetic agent for validation.

## Success Criteria

1. **SyntheticAgent class** that takes a solve probability matrix (3 abilities × 5 levels) and, given a problem, returns a simulated reward drawn from the correct distribution (using the MAA reward scheme: +3 for correct format+answer, −3 for wrong, intermediate values for partial).
2. **`run_simulation(stream, agent, n_steps)`** executes n steps: emit batch → agent scores → update stream → log. Returns a structured stream log.
3. **FixedCurriculumBaseline** implements the MAA paper's schedule (level 1 for N steps, then level 2) as an alternative stream for comparison.
4. **Stream log format**: JSONL file where each line contains step number, selected cell (ability + level), MDL score, batch rewards, cumulative reward, and selection reason.
5. **Comparison output**: Summary statistics showing cumulative reward over time for curiosity-driven vs. fixed curriculum, plus which cells each strategy spent time on.
6. **Learning agent variant**: An optional mode where the synthetic agent's solve probability improves after successful solves, testing whether the curiosity stream tracks a moving frontier.
7. **LiveAgent variant** (optional/stretch): An agent that sends problem text to the local LLM endpoint (`http://100.88.102.33:8000/v1`, OpenAI API, no auth) and scores the real response using the MAA reward functions. This validates the full pipeline with actual model behavior.
8. **Tests** verify: log schema correctness, cumulative rewards are correctly tallied, fixed baseline follows the expected schedule, simulation is deterministic given seed.