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

from repro_maa.task_cell import TaskCell


# ---------------------------------------------------------------------------
# Prompt formatter
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a reasoning assistant. "
    "Think step-by-step inside <think>...</think> tags. "
    "Then give your final answer inside <answer>...</answer> tags."
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
        completions: list[str],
        *,
        ground_truth: list[dict],
        **kwargs: object,
    ) -> list[float]:
        scores: list[float] = []
        for completion, gt in zip(completions, ground_truth):
            # TRL passes raw model output tokens. MAA scorers expect the
            # "Assistant: <think>...</think><answer>...</answer>" format.
            # Prepend the prefix when it is absent.
            text = completion if completion.startswith("Assistant:") else f"Assistant: {completion}"
            scores.append(float(cell.score(text, gt)))
        return scores

    return reward_func
