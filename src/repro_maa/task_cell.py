# Chunk: docs/chunks/taskcell_abstraction - Unified TaskCell abstraction
"""
Unified TaskCell abstraction over the three MAA abilities.

Each TaskCell represents one slot in the 3×5 ability/difficulty grid and
provides a uniform interface for generating problems and scoring responses.
This isolates the rest of the system from the MAA codebase's internal
structure (different generator classes, different reward function signatures,
different output formats per ability).

Usage::

    from repro_maa.task_cell import TaskCell

    cell = TaskCell("deduction", level=3, seed=42)
    problems = cell.generate(10)
    score = cell.score(model_response, problems[0]["ground_truth"])
"""
from __future__ import annotations

import random
from typing import Any

from repro_maa.maa_compat import (
    DeductionFormatter,
    DeductionSampler,
    InductionGenerator,
    abduction_score,
    deduction_score,
    generate_abduction_problem,
    induction_score,
)

_VALID_ABILITIES = ("deduction", "induction", "abduction")

# ---------------------------------------------------------------------------
# Abduction difficulty parameter mapping (levels 1–5)
# ---------------------------------------------------------------------------

_ABDUCTION_LEVEL_PARAMS: dict[int, dict[str, Any]] = {
    1: dict(num_goals=1, reachable_k=1, chain_depth=2, distractors=3, cycle_prob=0.1),
    2: dict(num_goals=2, reachable_k=1, chain_depth=3, distractors=3, cycle_prob=0.1),
    3: dict(num_goals=3, reachable_k=1, chain_depth=4, distractors=3, cycle_prob=0.1),
    4: dict(num_goals=4, reachable_k=1, chain_depth=5, distractors=3, cycle_prob=0.1),
    5: dict(num_goals=5, reachable_k=1, chain_depth=6, distractors=3, cycle_prob=0.1),
}


class TaskCell:
    """One slot in the 3×5 ability/difficulty grid.

    Parameters
    ----------
    ability : str
        One of ``"deduction"``, ``"induction"``, ``"abduction"``.
    level : int
        Difficulty level from 1 (easiest) to 5 (hardest).
    seed : int
        Random seed for deterministic generation.
    """

    def __init__(self, ability: str, level: int, seed: int = 42) -> None:
        if ability not in _VALID_ABILITIES:
            raise ValueError(
                f"ability must be one of {_VALID_ABILITIES!r}, got {ability!r}"
            )
        if not (1 <= level <= 5):
            raise ValueError(f"level must be in 1..5, got {level!r}")

        self.ability = ability
        self.level = level
        self._seed = seed

    def __repr__(self) -> str:
        return f"TaskCell(ability={self.ability!r}, level={self.level})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n: int) -> list[dict]:
        """Generate *n* problems at this cell's difficulty level.

        Returns a list of dicts, each with:
        - ``"puzzle_text"`` (str): Human-readable problem statement.
        - ``"ground_truth"`` (dict): Contains at least ``"solution_text_format"``
          in the format expected by the corresponding MAA reward scorer.

        Each call advances the internal seed so successive batches contain
        different problems.
        """
        dispatch = {
            "deduction": self._generate_deduction,
            "induction": self._generate_induction,
            "abduction": self._generate_abduction,
        }
        result = dispatch[self.ability](n)
        self._seed += n  # advance seed for next call
        return result

    def score(self, response: str, ground_truth: dict) -> float:
        """Score a model *response* against *ground_truth*.

        The *response* should be in MAA format::

            "Assistant: <think>...</think><answer>...</answer>"

        Returns a float reward value (positive for correct, negative for
        incorrect).
        """
        dispatch = {
            "deduction": deduction_score,
            "induction": induction_score,
            "abduction": abduction_score,
        }
        return dispatch[self.ability](response, ground_truth)

    # ------------------------------------------------------------------
    # Private generators
    # ------------------------------------------------------------------

    def _generate_deduction(self, n: int) -> list[dict]:
        sampler = DeductionSampler(difficulty=self.level, seed=self._seed)
        puzzles = sampler.sample_unique(n)
        results = []
        for formulas, assignment in puzzles:
            fmt = DeductionFormatter(formulas, assignment)
            results.append({
                "puzzle_text": fmt.puzzle_text(),
                "ground_truth": {
                    "solution_text_format": fmt.solution_text(),
                },
            })
        return results

    def _generate_induction(self, n: int) -> list[dict]:
        gen = InductionGenerator(seed=self._seed)
        puzzles = gen.generate_puzzles(num=n, level=self.level)
        return [
            {
                "puzzle_text": p["puzzle_text"],
                "ground_truth": {
                    "solution_text_format": p["solution_text"],
                },
            }
            for p in puzzles
        ]

    def _generate_abduction(self, n: int) -> list[dict]:
        params = _ABDUCTION_LEVEL_PARAMS[self.level]
        results = []
        for i in range(n):
            # Seed random state for determinism. generate_abduction_problem
            # uses the random module internally and does not accept a seed
            # parameter directly.
            random.seed(self._seed + i)
            problem = generate_abduction_problem(
                problem_id=self._seed + i,
                **params,
            )
            puzzle_text = _format_abduction_puzzle(problem)
            solution_text = _format_abduction_solution(problem)
            results.append({
                "puzzle_text": puzzle_text,
                "ground_truth": {
                    "solution_text_format": solution_text,
                },
            })
        return results


# ---------------------------------------------------------------------------
# Abduction formatting helpers
# ---------------------------------------------------------------------------

def _format_abduction_puzzle(problem: dict) -> str:
    """Format an abduction problem dict into a human-readable puzzle prompt."""
    lines = ["Given the following premises:"]
    for premise in problem["premises"]:
        lines.append(f"  - {premise}")

    if problem.get("known_atoms"):
        lines.append("")
        lines.append("Known facts:")
        for atom in problem["known_atoms"]:
            lines.append(f"  - {atom} is true")

    lines.append("")
    lines.append("For each of the following goals, determine if it is reachable or unreachable:")
    for idx, goal in enumerate(problem["goals"], start=1):
        lines.append(f"  ({idx}) {goal}")

    return "\n".join(lines)


def _format_abduction_solution(problem: dict) -> str:
    """Format the ground truth solution for an abduction problem.

    Produces numbered lines matching the format expected by
    ``backward_reasoning.compute_score``::

        (1) GoalA is reachable
        (2) GoalB is unreachable
    """
    reachable_set = set(problem["reachable_goals"])
    lines = []
    for idx, goal in enumerate(problem["goals"], start=1):
        status = "reachable" if goal in reachable_set else "unreachable"
        lines.append(f"({idx}) {goal} is {status}")
    return "\n".join(lines)
