# Chunk: docs/chunks/prompt_reward_bridge - Prompt formatting and reward adaptation
"""
Bridge between MAA task generators and TRL's GRPOTrainer.

Provides two pure functions:

- ``format_chat_prompt`` — converts a TaskCell problem dict into a Qwen3.5-9B
  chat message list with ``<think>``/``<answer>`` tag instructions.

- ``make_reward_func`` — factory that returns a TRL-compatible reward function
  bound to a specific ability type. The returned function conforms to the
  ``reward_funcs`` signature expected by ``trl.GRPOTrainer``:
  ``(completions, **kwargs) -> list[float]``.

Design assumptions:

- Completions from TRL are **plain strings** (the raw model output tokens),
  not conversation message lists. The ``curiosity_grpo_loop`` chunk controls
  dataset format and will ensure this.
- ``ground_truth`` is passed through as an extra dataset column via
  ``**kwargs`` in the reward function.
"""
from __future__ import annotations

from typing import Callable

import re

from repro_maa.task_cell import TaskCell

# ---------------------------------------------------------------------------
# Format scoring
# ---------------------------------------------------------------------------

_EXPECTED_TAGS = ["<think>", "</think>", "<answer>", "</answer>"]
_FORMAT_WEIGHT = 1.0  # max reward for perfect format
_CONTENT_CORRECT = 2.0  # reward for correct answer content
_CONTENT_WRONG = -1.0  # penalty for wrong but parseable answer


def _format_score(text: str) -> float:
    """Score format as (tags present / tags expected) * weight.

    Returns a value in [0, _FORMAT_WEIGHT].  Each of the four expected
    tags contributes equally.  This gives the model a continuous signal
    for partial format compliance rather than all-or-nothing.
    """
    present = sum(1 for tag in _EXPECTED_TAGS if tag in text)
    return (present / len(_EXPECTED_TAGS)) * _FORMAT_WEIGHT


def _extract_answer(text: str) -> str | None:
    """Extract content from the last <answer>...</answer> block, if any."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Prompt formatter
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a reasoning assistant. "
    "First, reason step-by-step inside <think> and </think> tags. "
    "Then, provide your final answer inside <answer> and </answer> tags. "
    "Example format:\n"
    "<think>\nStep 1: analyze the problem\nStep 2: work out the solution\n</think>\n"
    "<answer>42</answer>"
)


def format_chat_prompt(problem: dict) -> list[dict[str, str]]:
    """Convert a TaskCell problem dict into a chat message list.

    Parameters
    ----------
    problem : dict
        A problem dict as returned by :meth:`TaskCell.generate`, containing
        at least a ``"puzzle_text"`` key.

    Returns
    -------
    list[dict[str, str]]
        A two-element list of chat messages (system + user) suitable for
        feeding to a chat model.
    """
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": problem["puzzle_text"]},
    ]


# ---------------------------------------------------------------------------
# Reward adapter
# ---------------------------------------------------------------------------

def make_reward_func(ability: str) -> Callable[..., list[float]]:
    """Create a TRL-compatible reward function for the given *ability*.

    The returned callable has the signature::

        reward_func(completions: list[str], *, ground_truth: list[dict], **kwargs) -> list[float]

    This matches the ``reward_funcs`` interface that ``trl.GRPOTrainer``
    expects. Each completion is scored against its paired ground-truth dict
    using the MAA scorer for the specified ability.

    Parameters
    ----------
    ability : str
        One of ``"deduction"``, ``"induction"``, ``"abduction"``.

    Returns
    -------
    Callable
        A reward function conforming to TRL's ``reward_funcs`` protocol.
    """
    # Validate ability eagerly so errors surface at construction time.
    cell = TaskCell(ability, level=1)

    def reward_func(
        completions,
        *,
        ground_truth: list[dict],
        **kwargs: object,
    ) -> list[float]:
        scores: list[float] = []
        for completion, gt in zip(completions, ground_truth):
            # TRL may pass completions as list[str] or list[list[dict]]
            # (chat messages). Extract text content in either case.
            if isinstance(completion, list):
                text = " ".join(
                    msg.get("content", "") for msg in completion if isinstance(msg, dict)
                )
            else:
                text = completion

            # Log the raw model output before any mangling.
            print(f"\n[Raw Completion] ({len(text)} chars, "
                  f"fmt={sum(1 for t in _EXPECTED_TAGS if t in text)}/4 tags)")
            print(text[:2000])
            if len(text) > 2000:
                print(f"... ({len(text) - 2000} more chars)")

            # Continuous format score: tags_present / tags_expected * weight.
            # This gives gradient signal for partial format compliance
            # (e.g. 3/4 tags = 0.75) instead of binary pass/fail.
            fmt = _format_score(text)

            # Content score: try to extract and score <answer> content
            # independently of format validation.  The MAA scorer skips
            # content when format fails, so we do our own extraction.
            answer = _extract_answer(text)
            if answer is not None:
                # Build a well-formed string so the MAA scorer can grade
                # the content even if the original had format issues.
                well_formed = f"Assistant: <think>ok</think><answer>{answer}</answer>"
                maa_score = float(cell.score(well_formed, gt))
            else:
                # No <answer> tags at all — no content to score.
                maa_score = -2.0

            scores.append(fmt + maa_score)
        return scores

    return reward_func
