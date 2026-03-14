<!--
This document captures HOW you'll achieve the chunk's GOAL.
It should be specific enough that each step is a reasonable unit of work
to hand to an agent.
-->

# Implementation Plan

## Approach

Build a single new module `src/repro_maa/prompt_reward_bridge.py` containing two
pure functions (no classes needed — these are stateless transformations):

1. **`format_chat_prompt`** — Converts a TaskCell problem dict into a Qwen3.5-9B
   chat message list. The system message instructs the model to reason in
   `<think>...</think>` and answer in `<answer>...</answer>` tags. The user
   message contains the puzzle text. This mirrors what `curiosity_grpo_loop` will
   feed to GRPOTrainer as its prompt dataset.

2. **`make_reward_func`** — Factory that returns a TRL-compatible reward function
   bound to a specific ability type. TRL's GRPOTrainer calls reward functions with
   the signature `(completions, **kwargs) -> list[float]`, where `completions` is
   a list of model-generated strings and `**kwargs` includes any extra columns from
   the dataset (we pass `ground_truth` through). The returned function iterates
   over completions, dispatches to the correct MAA scorer via `TaskCell.score()`,
   and collects float rewards.

**Building on existing code:**

- `TaskCell` (from `taskcell_abstraction` chunk) already provides uniform
  `generate()` → problem dicts and `score(response, ground_truth) -> float`.
  The prompt formatter consumes `generate()` output; the reward adapter calls
  `score()` internally.
- `maa_compat` handles all MAA import path-hacking. This chunk does not touch
  MAA internals.

**Testing strategy (per `docs/trunk/TESTING_PHILOSOPHY.md`):**

- TDD for both functions — write failing tests first, then implement.
- Prompt formatting is a deterministic transformation: assert on message
  structure, role names, tag presence, and round-trip over all three abilities.
- Reward adapter is a deterministic computation: assert that wrapping+dispatching
  produces identical scores to calling `TaskCell.score()` directly.
- Integration test verifying the adapter works with TRL's calling convention
  (the `completions` + `**kwargs` signature).

## Sequence

### Step 1: Write prompt formatter tests

Create `tests/test_prompt_reward_bridge.py` with tests for `format_chat_prompt`:

- **`test_returns_two_message_list`**: Given a problem dict from any ability,
  returns a list of exactly 2 dicts with `role` keys `"system"` and `"user"`.
- **`test_system_message_contains_think_answer_instructions`**: The system
  message content instructs the model to use `<think>` and `<answer>` tags.
- **`test_user_message_contains_puzzle_text`**: The user message content
  contains the problem's `puzzle_text` verbatim.
- **`test_all_abilities_produce_valid_prompts`**: Parametrize across deduction,
  induction, abduction — each produces well-formed chat messages.
- **`test_different_difficulty_levels`**: Parametrize across levels 1-3 (keep
  fast) — prompt structure is consistent regardless of difficulty.

Tests use pre-built problem fixtures (not live generators) to stay fast per
TESTING_PHILOSOPHY.md unit test guidelines.

Location: `tests/test_prompt_reward_bridge.py`

### Step 2: Implement prompt formatter

Create `src/repro_maa/prompt_reward_bridge.py` with:

```python
def format_chat_prompt(problem: dict) -> list[dict[str, str]]:
```

The function takes a problem dict (as returned by `TaskCell.generate()`) and
returns a chat message list:

```python
[
    {"role": "system", "content": "<system prompt with think/answer instructions>"},
    {"role": "user", "content": problem["puzzle_text"]},
]
```

The system message should:
- Instruct the model to reason step-by-step inside `<think>...</think>` tags
- Instruct the model to place its final answer inside `<answer>...</answer>` tags
- Be concise — this is a formatting instruction, not a long prompt

Add module-level backreference comment:
`# Chunk: docs/chunks/prompt_reward_bridge - Prompt formatting and reward adaptation`

Run tests from Step 1 — they should pass.

Location: `src/repro_maa/prompt_reward_bridge.py`

### Step 3: Write reward adapter tests

Add tests for `make_reward_func` to `tests/test_prompt_reward_bridge.py`:

- **`test_reward_func_returns_list_of_floats`**: Given a list of completions
  and matching ground_truths, returns a list of floats with the same length.
- **`test_reward_parity_with_task_cell_score`**: For each ability, generate a
  problem, create a known-correct completion (wrapping the solution in
  `<think>...<answer>...</answer>` format), and verify the adapter returns
  the same score as calling `TaskCell.score()` directly. This is the core
  parity test from Success Criterion #4.
- **`test_reward_parity_wrong_answers`**: Same as above but with wrong answers
  — verify negative scores match.
- **`test_multiple_completions_scored_independently`**: Pass a batch of 3
  completions (some correct, some wrong) — verify each is scored independently
  and the list has the correct length.
- **`test_trl_calling_convention`**: Call the reward function with the exact
  signature TRL uses: `reward_func(completions=..., ground_truth=..., **kwargs)`.
  Verify it doesn't crash and returns the right shape. This is the integration
  test from Success Criterion #5.

Location: `tests/test_prompt_reward_bridge.py`

### Step 4: Implement reward adapter

Add to `src/repro_maa/prompt_reward_bridge.py`:

```python
def make_reward_func(ability: str) -> Callable:
```

This factory returns a function with TRL's `reward_funcs` signature:

```python
def reward_func(completions: list[str], ground_truth: list[dict], **kwargs) -> list[float]:
```

Internally, it:
1. Creates a `TaskCell` for the given ability (level doesn't matter for scoring —
   the MAA scorers only use the `ground_truth` dict, not difficulty parameters).
2. For each `(completion, gt)` pair, calls `cell.score(completion, gt)`.
3. Returns the list of float scores.

The completion strings from TRL are the raw model outputs. The MAA scorers expect
the format `"Assistant: <think>...</think><answer>...</answer>"`. The adapter must
prepend `"Assistant: "` if TRL strips it, or document the expected format clearly.
Investigate what TRL actually passes — if completions are raw generated text
(without the "Assistant: " prefix), the adapter prepends it.

Run tests from Step 3 — they should pass.

Location: `src/repro_maa/prompt_reward_bridge.py`

### Step 5: Add conftest fixtures for prompt_reward_bridge tests

Add shared fixtures to `tests/conftest.py`:

- `sample_problems`: A dict mapping each ability to a pre-generated problem dict
  (with `puzzle_text` and `ground_truth`). Generated once via `TaskCell.generate(1)`
  for each ability at level 1 with a fixed seed.
- `correct_completions`: A dict mapping each ability to a completion string that
  wraps the correct solution in `<think>...<answer>...</answer>` format.
- `wrong_completions`: A dict mapping each ability to a completion string with
  a deliberately wrong answer.

These fixtures keep test code DRY and provide the "pre-built problem fixtures"
referenced in Step 1.

Location: `tests/conftest.py`

### Step 6: Export from package and update GOAL.md code_paths

1. Add `format_chat_prompt` and `make_reward_func` to `src/repro_maa/__init__.py`
   exports in `__all__`.

2. Update the `code_paths` field in `docs/chunks/prompt_reward_bridge/GOAL.md` to:
   ```yaml
   code_paths:
     - src/repro_maa/prompt_reward_bridge.py
     - tests/test_prompt_reward_bridge.py
     - tests/conftest.py
     - src/repro_maa/__init__.py
   ```

Location: `src/repro_maa/__init__.py`, `docs/chunks/prompt_reward_bridge/GOAL.md`

### Step 7: Run full test suite and verify

Run `pytest tests/test_prompt_reward_bridge.py -v` to verify all tests pass.
Run `pytest tests/ -m "not slow"` to verify no regressions in existing tests.

## Dependencies

- **Existing chunks (ACTIVE):** `taskcell_abstraction` provides `TaskCell` with
  `generate()` and `score()`. `scaffold_project` provides `maa_compat` import shim.
- **External libraries:** None new. The bridge uses only `TaskCell` from the
  existing codebase. TRL is not imported here — the adapter merely conforms to
  TRL's calling convention so `curiosity_grpo_loop` can pass it directly.
- **MAA submodule:** Must be initialized (`git submodule update --init`) for
  integration tests that call real generators/scorers.

## Risks and Open Questions

- **TRL completion format**: TRL's GRPOTrainer may pass completions as raw
  generated text (just the model's output tokens) or as full conversation strings.
  The MAA scorers expect `"Assistant: <think>...</think><answer>...</answer>"`.
  The adapter should handle both cases — if the completion already starts with
  `"Assistant:"`, pass through; otherwise prepend it. Verify against TRL source
  during implementation.
- **Ground truth passthrough**: TRL's `reward_funcs` receive extra dataset columns
  via `**kwargs`. The `curiosity_grpo_loop` chunk must ensure `ground_truth` is
  included as a dataset column so it flows through to the reward function. This
  chunk documents the expectation; the downstream chunk implements it.
- **Conversational vs string format**: TRL can pass completions as either plain
  strings or conversation message lists depending on dataset format. Since
  `curiosity_grpo_loop` will construct the dataset, we can control this — design
  for string completions (simpler) and document the assumption.

## Deviations

<!--
POPULATE DURING IMPLEMENTATION, not at planning time.
-->