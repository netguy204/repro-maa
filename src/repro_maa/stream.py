# Chunk: docs/chunks/stream_generator - Curiosity-driven curriculum stream generator
"""Curiosity-driven curriculum stream generator.

Orchestrates TaskCell and MDLScorer to implement an epsilon-greedy
curriculum policy that selects the task cell with the highest MDL
curiosity score (learning frontier) at each step.

Usage::

    from repro_maa.stream import CuriosityStream, BatchResult
    from repro_maa.task_cell import TaskCell
    from repro_maa.mdl_scorer import MDLScorer

    cells = [TaskCell(a, l) for a in ("deduction", "induction", "abduction") for l in range(1, 6)]
    stream = CuriosityStream(cells, MDLScorer(), epsilon=0.1, seed=42)
    batch = stream.emit_batch()
    stream.update(cells[0], [1.0, 0.0, 1.0])

Cold-start behavior: when reward history is empty for all cells, MDL scores
are 0.0 everywhere and selection degenerates to random tie-breaking via the
seeded RNG. This is expected — the simulation harness seeds initial histories.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from repro_maa.mdl_scorer import MDLScorer
from repro_maa.task_cell import TaskCell


@dataclass
class BatchResult:
    """Metadata and problems for a single curriculum step.

    Attributes
    ----------
    ability : str
        The meta-ability type ("deduction", "induction", "abduction").
    level : int
        Difficulty level (1–5).
    mdl_score : float
        The MDL curiosity score that drove selection.
    selection_reason : str
        Either ``"curiosity"`` (greedy) or ``"exploration"`` (epsilon).
    batch_size : int
        Number of problems in this batch.
    step : int
        Monotonically increasing step counter.
    reward_history_summary : dict
        Per-cell summary with mean and count at time of selection.
    problems : list[dict]
        The actual problem dicts from ``TaskCell.generate()``.
    """

    ability: str
    level: int
    mdl_score: float
    selection_reason: str
    batch_size: int
    step: int
    reward_history_summary: dict[str, Any]
    problems: list[dict]


class CuriosityStream:
    """Curiosity-driven curriculum stream over a grid of TaskCells.

    Selects the task cell with the highest MDL curiosity score at each
    step (epsilon-greedy), generates a problem batch, and emits it with
    full metadata.

    Parameters
    ----------
    cells : list[TaskCell]
        The task cells to select from (typically 15: 3 abilities × 5 levels).
    scorer : MDLScorer
        The MDL scorer instance used to compute curiosity scores.
    batch_size : int
        Number of problems to generate per step.
    epsilon : float
        Exploration probability. 0 = pure greedy, 1 = pure random.
    window_size : int
        Rolling reward history window per cell.
    seed : int
        Random seed for the selection RNG (separate from TaskCell seeds).
    """

    def __init__(
        self,
        cells: list[TaskCell],
        scorer: MDLScorer,
        batch_size: int = 8,
        epsilon: float = 0.0,
        window_size: int = 20,
        seed: int = 42,
    ) -> None:
        if not cells:
            raise ValueError("cells must be a non-empty list of TaskCell instances")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}")

        self._cells = list(cells)
        self._scorer = scorer
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._window_size = window_size
        self._rng = np.random.default_rng(seed)
        self._step = 0

        # Rolling reward history per cell, keyed by (ability, level)
        self._history: dict[tuple[str, int], collections.deque[float]] = {}
        for cell in self._cells:
            key = (cell.ability, cell.level)
            if key not in self._history:
                self._history[key] = collections.deque(maxlen=window_size)

    def select_cell(self) -> tuple[str, TaskCell, float]:
        """Select the next task cell using epsilon-greedy policy.

        Returns
        -------
        tuple[str, TaskCell, float]
            A tuple of (reason, cell, mdl_score) where reason is
            ``"curiosity"`` or ``"exploration"``.
        """
        # Compute MDL scores for all cells
        scores = []
        for cell in self._cells:
            key = (cell.ability, cell.level)
            rewards = list(self._history[key])
            scores.append(self._scorer.score(rewards))

        # Epsilon-greedy selection
        if self._rng.random() < self._epsilon:
            idx = int(self._rng.integers(len(self._cells)))
            return ("exploration", self._cells[idx], scores[idx])

        # Greedy: find max score, break ties randomly
        max_score = max(scores)
        max_indices = [i for i, s in enumerate(scores) if s == max_score]
        if len(max_indices) == 1:
            idx = max_indices[0]
        else:
            idx = max_indices[int(self._rng.integers(len(max_indices)))]

        return ("curiosity", self._cells[idx], scores[idx])

    def emit_batch(self) -> BatchResult:
        """Select a cell and generate a problem batch with full metadata.

        Returns
        -------
        BatchResult
            The batch of problems and associated metadata.
        """
        reason, cell, mdl_score = self.select_cell()

        problems = cell.generate(self._batch_size)

        # Build reward history summary
        reward_history_summary: dict[str, Any] = {}
        for key, history in self._history.items():
            ability, level = key
            summary_key = f"{ability}_L{level}"
            if len(history) > 0:
                reward_history_summary[summary_key] = {
                    "mean": float(np.mean(list(history))),
                    "count": len(history),
                }
            else:
                reward_history_summary[summary_key] = {
                    "mean": 0.0,
                    "count": 0,
                }

        result = BatchResult(
            ability=cell.ability,
            level=cell.level,
            mdl_score=mdl_score,
            selection_reason=reason,
            batch_size=len(problems),
            step=self._step,
            reward_history_summary=reward_history_summary,
            problems=problems,
        )

        self._step += 1
        return result

    def update(self, cell: TaskCell, rewards: list[float]) -> None:
        """Record new reward outcomes for a cell.

        Parameters
        ----------
        cell : TaskCell
            The cell that produced the rewards.
        rewards : list[float]
            New reward values to append to the cell's history.
        """
        key = (cell.ability, cell.level)
        if key not in self._history:
            self._history[key] = collections.deque(maxlen=self._window_size)
        self._history[key].extend(rewards)
