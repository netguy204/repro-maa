# Chunk: docs/chunks/simulation_harness - Simulation harness for curiosity stream
"""Simulation harness for running curiosity-driven curriculum experiments.

Composes CuriosityStream, TaskCell, MDLScorer, and a pluggable Agent
protocol to run complete curriculum simulations with synthetic (and
optionally live) agents. Includes a fixed-curriculum baseline for
comparison against the MAA paper's ascending schedule.

Usage::

    from repro_maa.simulation import SyntheticAgent, run_simulation, compare_runs
    from repro_maa.simulation import FixedCurriculumBaseline
    from repro_maa.stream import CuriosityStream
    from repro_maa.task_cell import TaskCell
    from repro_maa.mdl_scorer import MDLScorer

    cells = [TaskCell(a, l) for a in ("deduction", "induction", "abduction")
             for l in range(1, 6)]
    agent = SyntheticAgent({("deduction", 1): 0.8, ("deduction", 2): 0.3}, seed=42)

    stream = CuriosityStream(cells, MDLScorer(), epsilon=0.1, seed=42)
    log = run_simulation(stream, agent, n_steps=50, log_path=Path("curiosity.jsonl"))

    baseline = FixedCurriculumBaseline(cells, n_steps=50, seed=42)
    baseline_log = run_simulation(baseline, agent, n_steps=50)

    summary = compare_runs(log, baseline_log)
    print(summary["summary_text"])
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from repro_maa.stream import BatchResult, CuriosityStream
from repro_maa.task_cell import TaskCell


# ============================================================================
# Agent protocol
# ============================================================================

@runtime_checkable
class Agent(Protocol):
    """Minimal agent interface for simulation.

    An agent receives a problem dict plus its cell coordinates and returns
    a scalar reward.
    """

    def respond(self, problem: dict, ability: str, level: int) -> float:
        """Return a reward for the given problem.

        Parameters
        ----------
        problem : dict
            Problem dict with ``"puzzle_text"`` and ``"ground_truth"`` keys.
        ability : str
            The meta-ability (``"deduction"``, ``"induction"``, ``"abduction"``).
        level : int
            The difficulty level (1–5).
        """
        ...


# ============================================================================
# SyntheticAgent
# ============================================================================

class SyntheticAgent:
    """Simulated agent with configurable solve probabilities.

    For each problem, draws a Bernoulli trial based on the solve probability
    for the problem's (ability, level) cell. Returns +3.0 for success,
    -3.0 for failure — matching the dominant MAA reward signal.

    Parameters
    ----------
    solve_matrix : dict[tuple[str, int], float]
        Maps ``(ability, level)`` to solve probability (0.0–1.0).
        Cells not in the matrix default to 0.0.
    seed : int
        Random seed for Bernoulli draws.
    learning_rate : float
        After a successful solve, increment the solve probability by this
        amount (capped at 1.0). 0.0 disables learning.
    """

    def __init__(
        self,
        solve_matrix: dict[tuple[str, int], float],
        seed: int = 42,
        learning_rate: float = 0.0,
    ) -> None:
        self._solve_matrix: dict[tuple[str, int], float] = dict(solve_matrix)
        self._learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)

    def respond(self, problem: dict, ability: str, level: int) -> float:
        """Draw a Bernoulli reward for the given cell.

        Returns +3.0 (correct) or -3.0 (wrong) based on the configured
        solve probability for ``(ability, level)``.
        """
        key = (ability, level)
        prob = self._solve_matrix.get(key, 0.0)
        success = self._rng.random() < prob

        if success and self._learning_rate > 0.0:
            self._solve_matrix[key] = min(1.0, prob + self._learning_rate)

        return 3.0 if success else -3.0


# ============================================================================
# StepRecord and JSONL log format
# ============================================================================

@dataclass
class StepRecord:
    """One step of a simulation run.

    Attributes
    ----------
    step : int
        Zero-based step index.
    ability : str
        Selected cell's ability.
    level : int
        Selected cell's difficulty level.
    mdl_score : float
        MDL curiosity score at time of selection.
    selection_reason : str
        ``"curiosity"``, ``"exploration"``, or ``"fixed_schedule"``.
    batch_rewards : list[float]
        Per-problem rewards returned by the agent.
    batch_mean_reward : float
        Mean of ``batch_rewards``.
    cumulative_reward : float
        Running sum of all rewards up to and including this step.
    reward_history_summary : dict[str, Any]
        Per-cell reward history snapshot.
    """

    step: int
    ability: str
    level: int
    mdl_score: float
    selection_reason: str
    batch_rewards: list[float]
    batch_mean_reward: float
    cumulative_reward: float
    reward_history_summary: dict[str, Any] = field(default_factory=dict)


def to_jsonl_line(record: StepRecord) -> str:
    """Serialize a single StepRecord to a JSON string (no trailing newline)."""
    return json.dumps(asdict(record), separators=(",", ":"))


def write_log(records: list[StepRecord], path: Path) -> None:
    """Write a list of StepRecords to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(to_jsonl_line(record) + "\n")


def read_log(path: Path) -> list[StepRecord]:
    """Read a JSONL log file back into StepRecords."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(StepRecord(**d))
    return records


# ============================================================================
# FixedCurriculumBaseline
# ============================================================================

class FixedCurriculumBaseline:
    """Fixed ascending curriculum that ignores reward feedback.

    Implements enough of the CuriosityStream interface (``emit_batch()``
    and ``update()``) to be used interchangeably with ``run_simulation``.

    Parameters
    ----------
    cells : list[TaskCell]
        The task cells to draw from.
    schedule : list[tuple[str, int, int]]
        Sequence of ``(ability, level, n_steps)`` triples.
    batch_size : int
        Number of problems per batch.
    seed : int
        Random seed (unused, kept for interface parity).
    """

    def __init__(
        self,
        cells: list[TaskCell],
        schedule: list[tuple[str, int, int]],
        batch_size: int = 8,
        seed: int = 42,
    ) -> None:
        self._cells = {(c.ability, c.level): c for c in cells}
        self._schedule = list(schedule)
        self._batch_size = batch_size
        self._step = 0

        # Expand the schedule into a per-step list of (ability, level)
        self._plan: list[tuple[str, int]] = []
        for ability, level, n in self._schedule:
            self._plan.extend([(ability, level)] * n)

    @classmethod
    def maa_default(
        cls,
        cells: list[TaskCell],
        n_steps: int,
        batch_size: int = 8,
        seed: int = 42,
    ) -> FixedCurriculumBaseline:
        """Create the MAA paper's default schedule.

        Splits steps evenly across abilities. For each ability, the first
        half of its allocation trains at level 1, the second half at level 2.
        This approximates the 7B model's "level 1 → 2" progression described
        in the paper.

        Note: the exact per-ability step counts are an approximation —
        the MAA paper does not specify exact counts for each ability.
        """
        abilities = ["deduction", "induction", "abduction"]
        steps_per_ability = n_steps // len(abilities)
        remainder = n_steps % len(abilities)

        schedule: list[tuple[str, int, int]] = []
        for i, ability in enumerate(abilities):
            total = steps_per_ability + (1 if i < remainder else 0)
            l1_steps = total // 2
            l2_steps = total - l1_steps
            if l1_steps > 0:
                schedule.append((ability, 1, l1_steps))
            if l2_steps > 0:
                schedule.append((ability, 2, l2_steps))

        return cls(cells, schedule, batch_size=batch_size, seed=seed)

    def emit_batch(self) -> BatchResult:
        """Emit the next batch according to the fixed schedule."""
        if self._step >= len(self._plan):
            raise StopIteration("Fixed schedule exhausted")

        ability, level = self._plan[self._step]
        cell = self._cells[(ability, level)]
        problems = cell.generate(self._batch_size)

        result = BatchResult(
            ability=ability,
            level=level,
            mdl_score=0.0,
            selection_reason="fixed_schedule",
            batch_size=len(problems),
            step=self._step,
            reward_history_summary={},
            problems=problems,
        )
        self._step += 1
        return result

    def update(self, cell: TaskCell, rewards: list[float]) -> None:
        """No-op — the fixed curriculum ignores feedback."""
        pass


# ============================================================================
# run_simulation
# ============================================================================

def _find_cell(cells: list[TaskCell], ability: str, level: int) -> TaskCell:
    """Look up a TaskCell by ability and level."""
    for cell in cells:
        if cell.ability == ability and cell.level == level:
            return cell
    raise ValueError(f"No cell found for ({ability!r}, {level})")


def run_simulation(
    stream: CuriosityStream | FixedCurriculumBaseline,
    agent: Agent,
    n_steps: int,
    log_path: Path | None = None,
) -> list[StepRecord]:
    """Run a complete simulation for *n_steps*.

    Parameters
    ----------
    stream : CuriosityStream | FixedCurriculumBaseline
        The curriculum strategy (must support ``emit_batch()`` and ``update()``).
    agent : Agent
        The agent that scores each problem.
    n_steps : int
        Number of curriculum steps to run.
    log_path : Path | None
        If provided, write JSONL log to this path.

    Returns
    -------
    list[StepRecord]
        The full simulation trace.
    """
    records: list[StepRecord] = []
    cumulative = 0.0

    for _ in range(n_steps):
        batch = stream.emit_batch()

        # Agent scores each problem
        rewards = [
            agent.respond(problem, batch.ability, batch.level)
            for problem in batch.problems
        ]

        # Feed rewards back to stream
        # For CuriosityStream, we need to find the cell object
        if hasattr(stream, "_cells") and isinstance(stream._cells, list):
            cell = _find_cell(stream._cells, batch.ability, batch.level)
        elif hasattr(stream, "_cells") and isinstance(stream._cells, dict):
            cell = stream._cells.get((batch.ability, batch.level))
        else:
            cell = None

        if cell is not None:
            stream.update(cell, rewards)

        batch_mean = sum(rewards) / len(rewards) if rewards else 0.0
        cumulative += sum(rewards)

        record = StepRecord(
            step=batch.step,
            ability=batch.ability,
            level=batch.level,
            mdl_score=batch.mdl_score,
            selection_reason=batch.selection_reason,
            batch_rewards=rewards,
            batch_mean_reward=batch_mean,
            cumulative_reward=cumulative,
            reward_history_summary=batch.reward_history_summary,
        )
        records.append(record)

    if log_path is not None:
        write_log(records, log_path)

    return records


# ============================================================================
# compare_runs
# ============================================================================

def compare_runs(
    curiosity_log: list[StepRecord],
    baseline_log: list[StepRecord],
) -> dict[str, Any]:
    """Compare curiosity-driven and baseline simulation logs.

    Returns
    -------
    dict
        Contains cumulative reward curves, cell frequency counts,
        final advantage, and a human-readable summary.
    """
    cum_curiosity = [r.cumulative_reward for r in curiosity_log]
    cum_baseline = [r.cumulative_reward for r in baseline_log]

    freq_curiosity: dict[str, int] = {}
    for r in curiosity_log:
        key = f"{r.ability}_L{r.level}"
        freq_curiosity[key] = freq_curiosity.get(key, 0) + 1

    freq_baseline: dict[str, int] = {}
    for r in baseline_log:
        key = f"{r.ability}_L{r.level}"
        freq_baseline[key] = freq_baseline.get(key, 0) + 1

    final_curiosity = cum_curiosity[-1] if cum_curiosity else 0.0
    final_baseline = cum_baseline[-1] if cum_baseline else 0.0
    advantage = final_curiosity - final_baseline

    # Build summary text
    lines = [
        "=== Simulation Comparison ===",
        "",
        f"Steps: curiosity={len(curiosity_log)}, baseline={len(baseline_log)}",
        f"Final cumulative reward: curiosity={final_curiosity:.1f}, baseline={final_baseline:.1f}",
        f"Advantage (curiosity - baseline): {advantage:+.1f}",
        "",
        "Cell selection frequency (curiosity):",
    ]
    for key in sorted(freq_curiosity):
        lines.append(f"  {key}: {freq_curiosity[key]}")
    lines.append("")
    lines.append("Cell selection frequency (baseline):")
    for key in sorted(freq_baseline):
        lines.append(f"  {key}: {freq_baseline[key]}")

    return {
        "cumulative_reward_curiosity": cum_curiosity,
        "cumulative_reward_baseline": cum_baseline,
        "cell_frequency_curiosity": freq_curiosity,
        "cell_frequency_baseline": freq_baseline,
        "final_advantage": advantage,
        "summary_text": "\n".join(lines),
    }


# ============================================================================
# LiveAgent (optional / stretch)
# ============================================================================

class LiveAgent:
    """Agent that calls a local LLM endpoint for real responses.

    Sends problem text to an OpenAI-compatible completions API and scores
    the response using the MAA reward functions via ``TaskCell.score()``.

    Parameters
    ----------
    cells : list[TaskCell]
        Task cells used for scoring responses.
    endpoint : str
        Base URL of the OpenAI-compatible API.
    model : str
        Model name to request.
    """

    def __init__(
        self,
        cells: list[TaskCell],
        endpoint: str = "http://100.88.102.33:8000/v1",
        model: str = "default",
    ) -> None:
        self._cells = {(c.ability, c.level): c for c in cells}
        self._endpoint = endpoint
        self._model = model

    def respond(self, problem: dict, ability: str, level: int) -> float:
        """Send the problem to the LLM and score the response.

        Returns the MAA reward score, or -3.0 if the endpoint is
        unreachable.
        """
        import openai

        client = openai.OpenAI(base_url=self._endpoint, api_key="not-needed")

        prompt = problem.get("puzzle_text", "")
        try:
            completion = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = completion.choices[0].message.content or ""
        except Exception:
            return -3.0

        cell = self._cells.get((ability, level))
        if cell is None:
            return -3.0

        ground_truth = problem.get("ground_truth", {})
        return cell.score(response_text, ground_truth)
