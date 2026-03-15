# Chunk: docs/chunks/training_comparison - Training comparison tests
"""Tests for the training comparison module.

Verifies the measurement apparatus: analysis functions, report generation,
CLI parsing, and log-reading workflow.  Tests use synthetic StepRecord
fixtures — no assertions about whether curiosity beats fixed curriculum
(per TESTING_PHILOSOPHY.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from repro_maa.compare_training import (
    ComparisonConfig,
    compute_cell_allocation,
    compute_per_ability_cumulative,
    compute_selection_variance,
    compute_steps_to_threshold,
    detect_plateaus,
    generate_report,
    parse_args,
    run_comparison,
)
from repro_maa.simulation import StepRecord, write_log


# ============================================================================
# Fixtures / helpers
# ============================================================================

def _make_curiosity_records(n: int = 20) -> list[StepRecord]:
    """Build synthetic curiosity-like records with varied cell selections."""
    rng = np.random.RandomState(42)
    abilities = ["deduction", "induction", "abduction"]
    records = []
    cumulative = 0.0
    for i in range(n):
        ability = abilities[i % 3]
        level = rng.randint(1, 6)
        batch_rewards = [float(rng.uniform(-1, 3)) for _ in range(4)]
        batch_mean = sum(batch_rewards) / len(batch_rewards)
        cumulative += sum(batch_rewards)
        records.append(StepRecord(
            step=i,
            ability=ability,
            level=level,
            mdl_score=float(rng.uniform(0, 5)),
            selection_reason="curiosity" if rng.random() > 0.1 else "exploration",
            batch_rewards=batch_rewards,
            batch_mean_reward=batch_mean,
            cumulative_reward=cumulative,
            reward_history_summary={},
        ))
    return records


def _make_baseline_records(n: int = 20) -> list[StepRecord]:
    """Build synthetic fixed-curriculum records with sequential assignments."""
    abilities = ["deduction", "induction", "abduction"]
    records = []
    cumulative = 0.0
    level_cycle = [1, 1, 2, 2, 3]
    for i in range(n):
        ability = abilities[i % 3]
        level = level_cycle[i % 5]
        batch_rewards = [float(0.5 + (i * 0.1))] * 4
        batch_mean = sum(batch_rewards) / len(batch_rewards)
        cumulative += sum(batch_rewards)
        records.append(StepRecord(
            step=i,
            ability=ability,
            level=level,
            mdl_score=0.0,
            selection_reason="fixed_schedule",
            batch_rewards=batch_rewards,
            batch_mean_reward=batch_mean,
            cumulative_reward=cumulative,
            reward_history_summary={},
        ))
    return records


def _make_plateau_records() -> list[StepRecord]:
    """Build records with a clear plateau for deduction.

    Deduction has 30 rounds: first 15 with reward ~2.0, last 15 with reward ~2.0
    (plateau throughout). Then jumps at round 31.
    """
    records = []
    cumulative = 0.0
    for i in range(30):
        # All deduction, flat reward = plateau
        batch_rewards = [2.0] * 4
        batch_mean = 2.0
        cumulative += sum(batch_rewards)
        records.append(StepRecord(
            step=i,
            ability="deduction",
            level=1,
            mdl_score=1.0,
            selection_reason="curiosity",
            batch_rewards=batch_rewards,
            batch_mean_reward=batch_mean,
            cumulative_reward=cumulative,
            reward_history_summary={},
        ))
    # Add a jump
    for i in range(30, 35):
        batch_rewards = [10.0] * 4
        batch_mean = 10.0
        cumulative += sum(batch_rewards)
        records.append(StepRecord(
            step=i,
            ability="deduction",
            level=1,
            mdl_score=3.0,
            selection_reason="curiosity",
            batch_rewards=batch_rewards,
            batch_mean_reward=batch_mean,
            cumulative_reward=cumulative,
            reward_history_summary={},
        ))
    return records


def _make_ability_dominant_records() -> list[StepRecord]:
    """Records where ability fully determines MDL score."""
    score_map = {"deduction": 5.0, "induction": 3.0, "abduction": 1.0}
    records = []
    cumulative = 0.0
    for i in range(30):
        ability = ["deduction", "induction", "abduction"][i % 3]
        level = (i % 5) + 1
        batch_rewards = [1.0] * 4
        cumulative += sum(batch_rewards)
        records.append(StepRecord(
            step=i,
            ability=ability,
            level=level,
            mdl_score=score_map[ability],
            selection_reason="curiosity",
            batch_rewards=batch_rewards,
            batch_mean_reward=1.0,
            cumulative_reward=cumulative,
            reward_history_summary={},
        ))
    return records


def _make_level_dominant_records() -> list[StepRecord]:
    """Records where level fully determines MDL score."""
    records = []
    cumulative = 0.0
    for i in range(30):
        ability = ["deduction", "induction", "abduction"][i % 3]
        level = (i % 5) + 1
        batch_rewards = [1.0] * 4
        cumulative += sum(batch_rewards)
        records.append(StepRecord(
            step=i,
            ability=ability,
            level=level,
            mdl_score=float(level) * 2.0,  # score depends only on level
            selection_reason="curiosity",
            batch_rewards=batch_rewards,
            batch_mean_reward=1.0,
            cumulative_reward=cumulative,
            reward_history_summary={},
        ))
    return records


# ============================================================================
# Tests: compute_steps_to_threshold
# ============================================================================

class TestComputeStepsToThreshold:
    def test_basic_threshold_reached(self):
        """Deduction reaches cumulative +10.0 — verify correct step returned."""
        records = []
        cumulative = 0.0
        # 10 deduction rounds, each contributing +2.0 cumulative per ability
        for i in range(10):
            batch_rewards = [0.5] * 4  # sum = 2.0
            cumulative += sum(batch_rewards)
            records.append(StepRecord(
                step=i,
                ability="deduction",
                level=1,
                mdl_score=1.0,
                selection_reason="curiosity",
                batch_rewards=batch_rewards,
                batch_mean_reward=0.5,
                cumulative_reward=cumulative,
                reward_history_summary={},
            ))

        result = compute_steps_to_threshold(records, [0.0, 10.0, 50.0])
        # Cumulative per ability: step 0 → 2.0, step 1 → 4.0, ..., step 4 → 10.0
        assert result["deduction"]["0.0"] == 0  # >= 0.0 at first step
        assert result["deduction"]["10.0"] == 4  # 5th step = 10.0
        assert result["deduction"]["50.0"] is None  # never reached

    def test_unreached_thresholds(self):
        """Thresholds not reached return None."""
        records = _make_curiosity_records(5)
        result = compute_steps_to_threshold(records, [9999.0])
        for ability in ("deduction", "induction", "abduction"):
            assert result[ability]["9999.0"] is None


# ============================================================================
# Tests: detect_plateaus
# ============================================================================

class TestDetectPlateaus:
    def test_flat_reward_detected(self):
        """A sequence of flat rewards triggers plateau detection."""
        records = _make_plateau_records()
        plateaus = detect_plateaus(records, window=10, min_improvement=0.5)
        # Deduction had 30 rounds of flat 2.0, then 5 rounds of 10.0
        # Plateaus should be detected in the flat region
        assert len(plateaus["deduction"]) > 0
        # All plateau rounds should be in the flat region (steps < 30)
        # or at early parts of the jump transition
        for step in plateaus["deduction"]:
            assert step < 35  # within the record range

    def test_no_plateau_with_improving_rewards(self):
        """Steadily improving rewards should have fewer/no plateaus."""
        records = []
        cumulative = 0.0
        for i in range(40):
            reward = float(i) * 0.5  # steadily increasing
            batch_rewards = [reward] * 4
            cumulative += sum(batch_rewards)
            records.append(StepRecord(
                step=i,
                ability="deduction",
                level=1,
                mdl_score=1.0,
                selection_reason="curiosity",
                batch_rewards=batch_rewards,
                batch_mean_reward=reward,
                cumulative_reward=cumulative,
                reward_history_summary={},
            ))
        plateaus = detect_plateaus(records, window=10, min_improvement=0.5)
        # With steadily increasing rewards, should detect no plateaus
        assert len(plateaus["deduction"]) == 0


# ============================================================================
# Tests: compute_per_ability_cumulative
# ============================================================================

class TestComputePerAbilityCumulative:
    def test_correct_cumulative_sums(self):
        """Per-ability cumulative sums match expected values."""
        records = _make_curiosity_records(9)
        result = compute_per_ability_cumulative(records)

        # Verify each ability has entries only for its rounds
        for ability in ("deduction", "induction", "abduction"):
            ability_records = [r for r in records if r.ability == ability]
            assert len(result[ability]) == len(ability_records)

            # Verify cumulative sum correctness
            expected_cum = 0.0
            for j, rec in enumerate(ability_records):
                expected_cum += sum(rec.batch_rewards)
                assert abs(result[ability][j] - expected_cum) < 1e-10


# ============================================================================
# Tests: compute_selection_variance
# ============================================================================

class TestComputeSelectionVariance:
    def test_ability_dominant(self):
        """When ability fully determines MDL score, ability_fraction ≈ 1.0."""
        records = _make_ability_dominant_records()
        result = compute_selection_variance(records)
        assert result["ability_fraction"] > 0.9
        assert result["difficulty_fraction"] < 0.1

    def test_difficulty_dominant(self):
        """When level fully determines MDL score, difficulty_fraction dominates."""
        records = _make_level_dominant_records()
        result = compute_selection_variance(records)
        assert result["difficulty_fraction"] > 0.9
        assert result["ability_fraction"] < 0.1

    def test_empty_records(self):
        """Empty records return zero fractions."""
        result = compute_selection_variance([])
        assert result["ability_fraction"] == 0.0
        assert result["difficulty_fraction"] == 0.0
        assert result["residual_fraction"] == 0.0


# ============================================================================
# Tests: generate_report
# ============================================================================

class TestGenerateReport:
    def test_report_structure(self, tmp_path: Path):
        """Report contains expected section headings and is valid Markdown."""
        curiosity = _make_curiosity_records()
        baseline = _make_baseline_records()
        report_path = tmp_path / "report.md"

        report = generate_report(curiosity, baseline, report_path)

        # Check expected section headings
        assert "## Cumulative Reward Comparison" in report
        assert "## Steps to Reward Thresholds" in report
        assert "## Plateau Detection" in report
        assert "## Cell Allocation" in report
        assert "## Per-Ability Reward Curves" in report
        assert "## Selection Granularity Analysis" in report

        # Verify file was written
        assert report_path.exists()
        assert report_path.read_text() == report

    def test_report_no_unclosed_formatting(self, tmp_path: Path):
        """Report has balanced Markdown formatting."""
        curiosity = _make_curiosity_records()
        baseline = _make_baseline_records()
        report_path = tmp_path / "report.md"

        report = generate_report(curiosity, baseline, report_path)

        # Check for balanced bold markers
        bold_count = report.count("**")
        assert bold_count % 2 == 0, "Unbalanced bold markers"


# ============================================================================
# Tests: parse_args
# ============================================================================

class TestParseArgs:
    def test_defaults(self):
        """Default arguments match ComparisonConfig defaults."""
        config = parse_args([])
        assert config.num_rounds == 100
        assert config.batch_size == 4
        assert config.seed == 42
        assert config.output_dir == "comparison_output"
        assert config.skip_training is False
        assert config.curiosity_log is None
        assert config.baseline_log is None
        assert config.plateau_window == 10
        assert config.plateau_min_improvement == 0.5
        assert config.gradient_checkpointing is True
        assert config.bf16 is True

    def test_skip_training(self):
        """--skip-training flag and log paths are parsed correctly."""
        config = parse_args([
            "--skip-training",
            "--curiosity-log", "a.jsonl",
            "--baseline-log", "b.jsonl",
        ])
        assert config.skip_training is True
        assert config.curiosity_log == "a.jsonl"
        assert config.baseline_log == "b.jsonl"

    def test_forwarded_train_params(self):
        """TrainConfig forwarding flags are parsed correctly."""
        config = parse_args([
            "--num-rounds", "50",
            "--batch-size", "8",
            "--learning-rate", "5e-5",
            "--no-gradient-checkpointing",
            "--no-bf16",
            "--epsilon", "0.2",
            "--window-size", "30",
        ])
        assert config.num_rounds == 50
        assert config.batch_size == 8
        assert config.learning_rate == 5e-5
        assert config.gradient_checkpointing is False
        assert config.bf16 is False
        assert config.epsilon == 0.2
        assert config.window_size == 30


# ============================================================================
# Tests: run_comparison (skip_training mode)
# ============================================================================

class TestRunComparisonSkipTraining:
    def test_reads_existing_logs(self, tmp_path: Path):
        """run_comparison with skip_training reads from JSONL log files."""
        curiosity_records = _make_curiosity_records(5)
        baseline_records = _make_baseline_records(5)

        curiosity_log = tmp_path / "curiosity.jsonl"
        baseline_log = tmp_path / "baseline.jsonl"
        write_log(curiosity_records, curiosity_log)
        write_log(baseline_records, baseline_log)

        config = ComparisonConfig(
            skip_training=True,
            curiosity_log=str(curiosity_log),
            baseline_log=str(baseline_log),
        )

        loaded_curiosity, loaded_baseline = run_comparison(config)

        assert len(loaded_curiosity) == 5
        assert len(loaded_baseline) == 5

        # Verify record contents match
        for orig, loaded in zip(curiosity_records, loaded_curiosity):
            assert orig.step == loaded.step
            assert orig.ability == loaded.ability
            assert orig.level == loaded.level
            assert abs(orig.cumulative_reward - loaded.cumulative_reward) < 1e-6

    def test_default_log_paths(self, tmp_path: Path):
        """Without explicit log paths, uses default locations under output_dir."""
        curiosity_records = _make_curiosity_records(3)
        baseline_records = _make_baseline_records(3)

        curiosity_dir = tmp_path / "curiosity"
        curiosity_dir.mkdir()
        baseline_dir = tmp_path / "fixed"
        baseline_dir.mkdir()

        write_log(curiosity_records, curiosity_dir / "stream_log.jsonl")
        write_log(baseline_records, baseline_dir / "stream_log.jsonl")

        config = ComparisonConfig(
            skip_training=True,
            output_dir=str(tmp_path),
        )

        loaded_curiosity, loaded_baseline = run_comparison(config)
        assert len(loaded_curiosity) == 3
        assert len(loaded_baseline) == 3


# ============================================================================
# Tests: compute_cell_allocation
# ============================================================================

class TestComputeCellAllocation:
    def test_counts_correct(self):
        """Cell allocation counts match actual selections."""
        records = _make_baseline_records(9)
        allocation = compute_cell_allocation(records)

        # Verify total count matches record count
        assert sum(allocation.values()) == 9

        # Verify format of keys
        for key in allocation:
            parts = key.split("_L")
            assert len(parts) == 2
            assert parts[0] in ("deduction", "induction", "abduction")
            assert int(parts[1]) in range(1, 6)
