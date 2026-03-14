# Chunk: docs/chunks/scaffold_project - Shared test fixtures
# Chunk: docs/chunks/prompt_reward_bridge - Prompt formatting and reward adaptation fixtures
"""Shared pytest fixtures for repro-maa tests."""

import pytest

from repro_maa.task_cell import TaskCell


# ---------------------------------------------------------------------------
# prompt_reward_bridge fixtures
# ---------------------------------------------------------------------------

_ABILITIES = ("deduction", "induction", "abduction")


@pytest.fixture(scope="session")
def sample_problems() -> dict[str, dict]:
    """Pre-generated problem dict for each ability (level 1, seed 42).

    Returns a dict mapping ability name to a problem dict with keys
    ``puzzle_text`` and ``ground_truth``.
    """
    result: dict[str, dict] = {}
    for ability in _ABILITIES:
        cell = TaskCell(ability, level=1, seed=42)
        problems = cell.generate(1)
        result[ability] = problems[0]
    return result


@pytest.fixture(scope="session")
def correct_completions(sample_problems: dict[str, dict]) -> dict[str, str]:
    """Completion string wrapping the correct solution for each ability.

    Wraps the ground-truth ``solution_text_format`` in the
    ``<think>...<answer>...</answer>`` format expected by MAA scorers.
    """
    result: dict[str, str] = {}
    for ability, problem in sample_problems.items():
        solution = problem["ground_truth"]["solution_text_format"]
        result[ability] = (
            f"Assistant: <think>Let me reason.</think>"
            f"<answer>{solution}</answer>"
        )
    return result


@pytest.fixture(scope="session")
def wrong_completions() -> dict[str, str]:
    """Completion string with deliberately wrong answers for each ability."""
    return {
        "deduction": (
            "Assistant: <think>Guessing.</think>"
            "<answer>WRONG WRONG WRONG</answer>"
        ),
        "induction": (
            "Assistant: <think>Guessing.</think>"
            "<answer>-999999</answer>"
        ),
        "abduction": (
            "Assistant: <think>Guessing.</think>"
            "<answer>(1) WRONG is WRONG</answer>"
        ),
    }
