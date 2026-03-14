# Chunk: docs/chunks/simulation_harness - Simulation harness tests
"""
Tests for the simulation harness.

Verifies SyntheticAgent reward distribution, StepRecord serialization,
run_simulation correctness, FixedCurriculumBaseline schedule adherence,
compare_runs output, and full determinism. Uses synthetic problem fixtures
to avoid MAA generator overhead (per TESTING_PHILOSOPHY.md).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from repro_maa.simulation import (
    Agent,
    FixedCurriculumBaseline,
    StepRecord,
    SyntheticAgent,
    compare_runs,
    read_log,
    run_simulation,
    to_jsonl_line,
    write_log,
)
from repro_maa.stream import CuriosityStream
from repro_maa.mdl_scorer import MDLScorer
from repro_maa.task_cell import TaskCell


# ============================================================================
# Fixtures / helpers
# ============================================================================

def _make_cells() -> list[TaskCell]:
    """Create the full 3x5 grid of task cells."""
    return [
        TaskCell(ability, level, seed=100 + i)
        for i, (ability, level) in enumerate(
            (a, l)
            for a in ("deduction", "induction", "abduction")
            for l in range(1, 6)
        )
    ]


def _fake_generate(n: int) -> list[dict]:
    """Stub replacement for TaskCell.generate()."""
    return [
        {"puzzle_text": f"Problem {i}", "ground_truth": {"solution_text_format": f"Answer {i}"}}
        for i in range(n)
    ]


def _make_stream(cells=None, **kwargs):
    """Create a CuriosityStream with defaults."""
    if cells is None:
        cells = _make_cells()
    defaults = dict(scorer=MDLScorer(), batch_size=4, epsilon=0.0, seed=42)
    defaults.update(kwargs)
    return CuriosityStream(cells=cells, **defaults)


# ============================================================================
# SyntheticAgent tests
# ============================================================================

class TestSyntheticAgent:
    def test_protocol_compliance(self):
        """SyntheticAgent satisfies the Agent protocol."""
        agent = SyntheticAgent({("deduction", 1): 0.5})
        assert isinstance(agent, Agent)

    def test_correct_reward_values_always_solve(self):
        """With solve_prob=1.0, agent always returns +3.0."""
        agent = SyntheticAgent({("deduction", 1): 1.0}, seed=0)
        for _ in range(20):
            r = agent.respond({}, "deduction", 1)
            assert r == 3.0

    def test_correct_reward_values_never_solve(self):
        """With solve_prob=0.0, agent always returns -3.0."""
        agent = SyntheticAgent({("deduction", 1): 0.0}, seed=0)
        for _ in range(20):
            r = agent.respond({}, "deduction", 1)
            assert r == -3.0

    def test_unknown_cell_defaults_to_zero(self):
        """A cell not in the solve_matrix defaults to solve_prob=0.0."""
        agent = SyntheticAgent({("deduction", 1): 1.0}, seed=0)
        r = agent.respond({}, "induction", 3)
        assert r == -3.0

    def test_stochastic(self):
        """With solve_prob=0.5, both +3 and -3 appear over many draws."""
        agent = SyntheticAgent({("deduction", 1): 0.5}, seed=123)
        rewards = [agent.respond({}, "deduction", 1) for _ in range(200)]
        assert 3.0 in rewards
        assert -3.0 in rewards
        mean = np.mean(rewards)
        # With 200 draws at p=0.5, mean should be near 0
        assert abs(mean) < 1.5

    def test_deterministic(self):
        """Same seed produces identical reward sequences."""
        a1 = SyntheticAgent({("deduction", 1): 0.5}, seed=77)
        a2 = SyntheticAgent({("deduction", 1): 0.5}, seed=77)
        for _ in range(50):
            assert a1.respond({}, "deduction", 1) == a2.respond({}, "deduction", 1)

    def test_learning(self):
        """With learning_rate>0, success rate increases over time."""
        agent = SyntheticAgent(
            {("deduction", 1): 0.3},
            seed=42,
            learning_rate=0.05,
        )
        # Run 200 trials and compare first half vs second half success rates
        rewards = [agent.respond({}, "deduction", 1) for _ in range(200)]
        first_half = rewards[:100]
        second_half = rewards[100:]
        first_success = sum(1 for r in first_half if r == 3.0)
        second_success = sum(1 for r in second_half if r == 3.0)
        # The second half should have more successes due to learning
        assert second_success >= first_success

    def test_learning_caps_at_one(self):
        """Solve probability never exceeds 1.0 even with high learning rate."""
        agent = SyntheticAgent(
            {("deduction", 1): 0.9},
            seed=42,
            learning_rate=0.5,
        )
        for _ in range(100):
            agent.respond({}, "deduction", 1)
        # Probability should be capped at 1.0
        assert agent._solve_matrix[("deduction", 1)] <= 1.0


# ============================================================================
# StepRecord and log serialization tests
# ============================================================================

class TestLogSerialization:
    def _make_record(self, step=0, cumulative=0.0) -> StepRecord:
        return StepRecord(
            step=step,
            ability="deduction",
            level=2,
            mdl_score=0.42,
            selection_reason="curiosity",
            batch_rewards=[3.0, -3.0, 3.0, 3.0],
            batch_mean_reward=1.5,
            cumulative_reward=cumulative,
            reward_history_summary={"deduction_L2": {"mean": 1.0, "count": 4}},
        )

    def test_step_record_roundtrip(self, tmp_path: Path):
        """Write a StepRecord to JSONL and read it back identically."""
        original = self._make_record(step=5, cumulative=12.0)
        log_path = tmp_path / "test.jsonl"
        write_log([original], log_path)
        loaded = read_log(log_path)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.step == original.step
        assert r.ability == original.ability
        assert r.level == original.level
        assert r.mdl_score == pytest.approx(original.mdl_score)
        assert r.selection_reason == original.selection_reason
        assert r.batch_rewards == original.batch_rewards
        assert r.batch_mean_reward == pytest.approx(original.batch_mean_reward)
        assert r.cumulative_reward == pytest.approx(original.cumulative_reward)
        assert r.reward_history_summary == original.reward_history_summary

    def test_log_schema_completeness(self):
        """A serialized StepRecord JSON contains all required keys."""
        import json
        record = self._make_record()
        line = to_jsonl_line(record)
        d = json.loads(line)
        required = {
            "step", "ability", "level", "mdl_score", "selection_reason",
            "batch_rewards", "batch_mean_reward", "cumulative_reward",
        }
        assert required.issubset(d.keys())

    def test_cumulative_reward_correctness(self, tmp_path: Path):
        """Cumulative reward is the running sum of all batch rewards."""
        records = []
        cumulative = 0.0
        for i in range(5):
            batch_rewards = [3.0, -3.0]
            cumulative += sum(batch_rewards)
            records.append(StepRecord(
                step=i,
                ability="deduction",
                level=1,
                mdl_score=0.0,
                selection_reason="curiosity",
                batch_rewards=batch_rewards,
                batch_mean_reward=0.0,
                cumulative_reward=cumulative,
            ))

        # Each step adds 0 (3 + -3), so cumulative stays 0
        for r in records:
            assert r.cumulative_reward == pytest.approx(0.0)

    def test_multi_record_roundtrip(self, tmp_path: Path):
        """Multiple records written and read back preserve order."""
        records = [self._make_record(step=i, cumulative=float(i * 3)) for i in range(10)]
        log_path = tmp_path / "multi.jsonl"
        write_log(records, log_path)
        loaded = read_log(log_path)
        assert len(loaded) == 10
        for orig, loaded_r in zip(records, loaded):
            assert orig.step == loaded_r.step
            assert orig.cumulative_reward == pytest.approx(loaded_r.cumulative_reward)


# ============================================================================
# run_simulation tests
# ============================================================================

class TestRunSimulation:
    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_returns_correct_count(self, _mock):
        """With n_steps=10, returns 10 StepRecords."""
        cells = _make_cells()
        stream = _make_stream(cells)
        agent = SyntheticAgent({("deduction", 1): 0.5}, seed=42)
        records = run_simulation(stream, agent, n_steps=10)
        assert len(records) == 10

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_steps_sequential(self, _mock):
        """Step numbers are 0, 1, ..., n-1."""
        cells = _make_cells()
        stream = _make_stream(cells)
        agent = SyntheticAgent({("deduction", 1): 0.5}, seed=42)
        records = run_simulation(stream, agent, n_steps=5)
        assert [r.step for r in records] == [0, 1, 2, 3, 4]

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_cumulative_correct(self, _mock):
        """Cumulative reward at step k = sum of all batch rewards from 0..k."""
        cells = _make_cells()
        stream = _make_stream(cells)
        agent = SyntheticAgent({("deduction", 1): 1.0}, seed=42)
        # Set all probs to 1.0 so all rewards are +3.0
        for a in ("deduction", "induction", "abduction"):
            for l in range(1, 6):
                agent._solve_matrix[(a, l)] = 1.0

        records = run_simulation(stream, agent, n_steps=5)

        running_sum = 0.0
        for r in records:
            running_sum += sum(r.batch_rewards)
            assert r.cumulative_reward == pytest.approx(running_sum)

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_deterministic(self, _mock):
        """Same seeds produce identical logs."""
        def _run(seed):
            cells = _make_cells()
            stream = _make_stream(cells, seed=seed)
            agent = SyntheticAgent(
                {(a, l): 0.5 for a in ("deduction", "induction", "abduction") for l in range(1, 6)},
                seed=seed,
            )
            return run_simulation(stream, agent, n_steps=10)

        log1 = _run(99)
        log2 = _run(99)
        for r1, r2 in zip(log1, log2):
            assert r1.step == r2.step
            assert r1.ability == r2.ability
            assert r1.level == r2.level
            assert r1.batch_rewards == r2.batch_rewards
            assert r1.cumulative_reward == pytest.approx(r2.cumulative_reward)

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_writes_log_file(self, _mock, tmp_path: Path):
        """When log_path is provided, JSONL file exists with n_steps lines."""
        cells = _make_cells()
        stream = _make_stream(cells)
        agent = SyntheticAgent({("deduction", 1): 0.5}, seed=42)
        log_path = tmp_path / "sim.jsonl"
        run_simulation(stream, agent, n_steps=7, log_path=log_path)
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 7


# ============================================================================
# FixedCurriculumBaseline tests
# ============================================================================

class TestFixedCurriculumBaseline:
    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_follows_schedule(self, _mock):
        """Batches follow the specified schedule exactly."""
        cells = _make_cells()
        schedule = [("deduction", 1, 3), ("deduction", 2, 2)]
        baseline = FixedCurriculumBaseline(cells, schedule, batch_size=4)

        results = [baseline.emit_batch() for _ in range(5)]

        # First 3: deduction L1
        for r in results[:3]:
            assert r.ability == "deduction"
            assert r.level == 1
            assert r.selection_reason == "fixed_schedule"
            assert r.mdl_score == 0.0

        # Next 2: deduction L2
        for r in results[3:]:
            assert r.ability == "deduction"
            assert r.level == 2

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_maa_default(self, _mock):
        """maa_default produces a schedule covering all 3 abilities, levels 1 and 2."""
        cells = _make_cells()
        baseline = FixedCurriculumBaseline.maa_default(cells, n_steps=24)

        results = [baseline.emit_batch() for _ in range(24)]
        abilities_seen = set()
        levels_seen = set()
        for r in results:
            abilities_seen.add(r.ability)
            levels_seen.add(r.level)

        assert abilities_seen == {"deduction", "induction", "abduction"}
        assert levels_seen == {1, 2}

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_with_run_simulation(self, _mock):
        """run_simulation works with FixedCurriculumBaseline."""
        cells = _make_cells()
        schedule = [("deduction", 1, 3), ("induction", 1, 2)]
        baseline = FixedCurriculumBaseline(cells, schedule, batch_size=4)
        agent = SyntheticAgent({("deduction", 1): 0.8, ("induction", 1): 0.5}, seed=42)

        records = run_simulation(baseline, agent, n_steps=5)
        assert len(records) == 5
        assert all(isinstance(r, StepRecord) for r in records)
        # Cumulative reward should be a running total
        assert records[-1].cumulative_reward == pytest.approx(
            sum(sum(r.batch_rewards) for r in records)
        )

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_step_counter_increments(self, _mock):
        """Step counter increments across emit_batch calls."""
        cells = _make_cells()
        schedule = [("deduction", 1, 5)]
        baseline = FixedCurriculumBaseline(cells, schedule, batch_size=2)

        steps = [baseline.emit_batch().step for _ in range(5)]
        assert steps == [0, 1, 2, 3, 4]

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_update_is_noop(self, _mock):
        """update() does not raise or change behavior."""
        cells = _make_cells()
        schedule = [("deduction", 1, 3)]
        baseline = FixedCurriculumBaseline(cells, schedule, batch_size=2)

        batch1 = baseline.emit_batch()
        baseline.update(cells[0], [3.0, -3.0])
        batch2 = baseline.emit_batch()

        # Second batch still follows schedule (not affected by update)
        assert batch2.ability == "deduction"
        assert batch2.level == 1


# ============================================================================
# compare_runs tests
# ============================================================================

class TestCompareRuns:
    def _make_log(self, n: int, ability: str, level: int, reward: float) -> list[StepRecord]:
        """Create a simple log with uniform rewards."""
        records = []
        cumulative = 0.0
        for i in range(n):
            batch_rewards = [reward] * 4
            cumulative += sum(batch_rewards)
            records.append(StepRecord(
                step=i,
                ability=ability,
                level=level,
                mdl_score=0.0,
                selection_reason="curiosity",
                batch_rewards=batch_rewards,
                batch_mean_reward=reward,
                cumulative_reward=cumulative,
            ))
        return records

    def test_structure(self):
        """Output dict has all required keys."""
        curiosity_log = self._make_log(5, "deduction", 1, 3.0)
        baseline_log = self._make_log(5, "deduction", 1, -3.0)
        result = compare_runs(curiosity_log, baseline_log)

        required_keys = {
            "cumulative_reward_curiosity",
            "cumulative_reward_baseline",
            "cell_frequency_curiosity",
            "cell_frequency_baseline",
            "final_advantage",
            "summary_text",
        }
        assert required_keys.issubset(result.keys())

    def test_correct_tallies(self):
        """Cell frequencies and cumulative rewards match expectations."""
        curiosity_log = self._make_log(3, "deduction", 1, 3.0)
        baseline_log = self._make_log(3, "induction", 2, -3.0)

        result = compare_runs(curiosity_log, baseline_log)

        assert result["cell_frequency_curiosity"] == {"deduction_L1": 3}
        assert result["cell_frequency_baseline"] == {"induction_L2": 3}

        # Curiosity: 3 steps * 4 problems * 3.0 = 36.0
        assert result["cumulative_reward_curiosity"][-1] == pytest.approx(36.0)
        # Baseline: 3 steps * 4 problems * -3.0 = -36.0
        assert result["cumulative_reward_baseline"][-1] == pytest.approx(-36.0)
        assert result["final_advantage"] == pytest.approx(72.0)

    def test_summary_text_is_string(self):
        """summary_text is a non-empty string."""
        log = self._make_log(2, "deduction", 1, 3.0)
        result = compare_runs(log, log)
        assert isinstance(result["summary_text"], str)
        assert len(result["summary_text"]) > 0

    def test_empty_logs(self):
        """compare_runs handles empty logs without error."""
        result = compare_runs([], [])
        assert result["final_advantage"] == 0.0
        assert result["cumulative_reward_curiosity"] == []
        assert result["cumulative_reward_baseline"] == []
