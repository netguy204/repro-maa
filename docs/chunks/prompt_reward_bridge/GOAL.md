---
status: ACTIVE
ticket: null
parent_chunk: null
code_paths:
  - src/repro_maa/prompt_reward_bridge.py
  - tests/test_prompt_reward_bridge.py
  - tests/conftest.py
  - src/repro_maa/__init__.py
code_references:
  - ref: src/repro_maa/prompt_reward_bridge.py#format_chat_prompt
    implements: "Converts TaskCell problem dict into Qwen3.5-9B chat message list with think/answer tag instructions"
  - ref: src/repro_maa/prompt_reward_bridge.py#make_reward_func
    implements: "Factory returning TRL-compatible reward function that dispatches to MAA scorers by ability type"
  - ref: tests/test_prompt_reward_bridge.py#TestFormatChatPrompt
    implements: "Round-trip tests for prompt formatting across all abilities and difficulty levels"
  - ref: tests/test_prompt_reward_bridge.py#TestMakeRewardFunc
    implements: "Reward parity tests and TRL calling convention integration test"
  - ref: tests/test_prompt_reward_bridge.py#TestCompletionFormatHandling
    implements: "Verifies adapter handles both raw and Assistant-prefixed completions"
  - ref: tests/conftest.py#sample_problems
    implements: "Pre-generated problem fixtures for each ability type"
  - ref: tests/conftest.py#correct_completions
    implements: "Correct completion fixtures wrapping solutions in expected format"
  - ref: tests/conftest.py#wrong_completions
    implements: "Wrong completion fixtures for negative score testing"
narrative: curiosity_training_run
investigation: null
subsystems: []
friction_entries: []
bug_type: null
depends_on: []
created_after:
- stream_visualization
---

# Chunk Goal

## Minor Goal

Build the glue layer between the existing MAA task generators and TRL's GRPOTrainer. Adapt TaskCell problems into Qwen3.5-9B chat prompt format (system message + user problem statement, expecting `<think>...</think><answer>...</answer>` responses). Wrap the MAA reward functions (`formula.py`, `backward_reasoning.py`, `squence.py`) into TRL's `reward_funcs` interface — each function receives a list of completions and returns a list of float rewards. This is pure formatting and scoring code with no training logic.

## Success Criteria

1. **Prompt formatter**: A function that takes a TaskCell problem dict and returns a chat message list (`[{"role": "system", ...}, {"role": "user", ...}]`) in the format Qwen3.5-9B expects.
2. **Reward function adapter**: A function compatible with TRL's `reward_funcs` interface that dispatches to the correct MAA reward scorer based on ability type and returns float rewards.
3. **Round-trip test**: Prompt formatting produces valid chat messages for all three ability types (deduction, induction, abduction) across multiple difficulty levels.
4. **Reward parity test**: The adapted reward function produces identical scores to the existing MAA reward functions for the same (completion, ground_truth) pairs.
5. **Integration test**: The reward adapter works when called with the signature TRL's GRPOTrainer expects.