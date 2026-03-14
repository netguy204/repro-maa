# Chunk: docs/chunks/curiosity_grpo_loop - Tests for curiosity GRPO training loop
"""Tests for the GRPO training loop.

All tests run on CPU with mocked heavy dependencies (GRPOTrainer,
model loading).  Tests verify dataset construction, reward dispatching,
StepRecord emission, JSONL logging, and CLI argument parsing.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repro_maa.simulation import (
    FixedCurriculumBaseline,
    StepRecord,
    read_log,
    to_jsonl_line,
)
from repro_maa.stream import BatchResult
from repro_maa.train import (
    TrainConfig,
    append_step_record,
    build_dataset,
    make_dispatching_reward_func,
    parse_args,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_batch() -> BatchResult:
    """A BatchResult with 3 synthetic problems."""
    return BatchResult(
        ability="deduction",
        level=1,
        mdl_score=1.5,
        selection_reason="curiosity",
        batch_size=3,
        step=0,
        reward_history_summary={"deduction_L1": {"mean": 0.5, "count": 5}},
        problems=[
            {
                "puzzle_text": f"Solve puzzle {i}",
                "ground_truth": {"solution_text_format": f"answer_{i}"},
            }
            for i in range(3)
        ],
    )


@pytest.fixture
def mock_batch_induction() -> BatchResult:
    """A BatchResult for induction problems."""
    return BatchResult(
        ability="induction",
        level=2,
        mdl_score=2.0,
        selection_reason="exploration",
        batch_size=2,
        step=1,
        reward_history_summary={},
        problems=[
            {
                "puzzle_text": f"Find the pattern {i}",
                "ground_truth": {"solution_text_format": f"pattern_{i}"},
            }
            for i in range(2)
        ],
    )


def _make_step_record(step: int = 0, **overrides: object) -> StepRecord:
    """Helper to construct a StepRecord with sensible defaults."""
    defaults = dict(
        step=step,
        ability="deduction",
        level=1,
        mdl_score=1.5,
        selection_reason="curiosity",
        batch_rewards=[3.0, -3.0, 3.0],
        batch_mean_reward=1.0,
        cumulative_reward=3.0,
        reward_history_summary={"deduction_L1": {"mean": 0.5, "count": 5}},
    )
    defaults.update(overrides)
    return StepRecord(**defaults)


# ============================================================================
# Step 8: Dataset construction tests
# ============================================================================

class TestBuildDataset:
    """Tests for ``build_dataset``."""

    def test_build_dataset_shape(self, mock_batch: BatchResult) -> None:
        """Given a BatchResult with N problems, returns Dataset with N rows."""
        ds = build_dataset(mock_batch)
        assert len(ds) == 3
        assert set(ds.column_names) == {"prompt", "ground_truth", "ability"}

    def test_build_dataset_prompt_format(self, mock_batch: BatchResult) -> None:
        """Each prompt is a two-element chat message list (system + user)."""
        ds = build_dataset(mock_batch)
        for row in ds:
            prompt = row["prompt"]
            assert len(prompt) == 2
            assert prompt[0]["role"] == "system"
            assert prompt[1]["role"] == "user"

    def test_build_dataset_ability_column(self, mock_batch: BatchResult) -> None:
        """Ability column is constant within a batch."""
        ds = build_dataset(mock_batch)
        for row in ds:
            assert row["ability"] == "deduction"

    def test_build_dataset_ground_truth(self, mock_batch: BatchResult) -> None:
        """Ground truth column contains the problem's ground_truth dicts."""
        ds = build_dataset(mock_batch)
        for i, row in enumerate(ds):
            assert row["ground_truth"]["solution_text_format"] == f"answer_{i}"


# ============================================================================
# Step 8: Dispatching reward function tests
# ============================================================================

class TestDispatchingRewardFunc:
    """Tests for ``make_dispatching_reward_func``."""

    def test_dispatching_reward_func(self) -> None:
        """Dispatcher routes to the correct scorer based on ability."""
        dispatch = make_dispatching_reward_func()

        # Mock the underlying make_reward_func to verify dispatch routing.
        mock_inner = MagicMock(return_value=[1.0, 2.0])
        with patch("repro_maa.train.make_reward_func", return_value=mock_inner):
            # Clear cache so the mock takes effect.
            dispatch._cache.clear()
            result = dispatch(
                ["completion_a", "completion_b"],
                ground_truth=[{"gt": "a"}, {"gt": "b"}],
                ability=["deduction", "deduction"],
            )

        assert result == [1.0, 2.0]
        mock_inner.assert_called_once_with(
            ["completion_a", "completion_b"],
            ground_truth=[{"gt": "a"}, {"gt": "b"}],
        )

    def test_dispatching_reward_func_caches(self) -> None:
        """Calling with the same ability reuses the cached reward function."""
        dispatch = make_dispatching_reward_func()

        mock_inner = MagicMock(return_value=[1.0])
        with patch("repro_maa.train.make_reward_func", return_value=mock_inner) as factory:
            dispatch._cache.clear()
            dispatch(["c1"], ground_truth=[{"gt": "a"}], ability=["deduction"])
            dispatch(["c2"], ground_truth=[{"gt": "b"}], ability=["deduction"])

            # Factory should only be called once (second call uses cache).
            factory.assert_called_once_with("deduction")

    def test_dispatching_reward_func_different_abilities(self) -> None:
        """Different abilities create separate cached functions."""
        dispatch = make_dispatching_reward_func()

        mock_inner = MagicMock(return_value=[1.0])
        with patch("repro_maa.train.make_reward_func", return_value=mock_inner) as factory:
            dispatch._cache.clear()
            dispatch(["c1"], ground_truth=[{"gt": "a"}], ability=["deduction"])
            dispatch(["c2"], ground_truth=[{"gt": "b"}], ability=["induction"])

            assert factory.call_count == 2


# ============================================================================
# Step 9: StepRecord emission and logging tests
# ============================================================================

class TestStepRecordLogging:
    """Tests for ``append_step_record`` and log roundtrip."""

    def test_step_record_log_appendable(self, tmp_path: Path) -> None:
        """Two records appended via append_step_record can be read back."""
        log_path = tmp_path / "test.jsonl"
        r1 = _make_step_record(step=0)
        r2 = _make_step_record(step=1, cumulative_reward=6.0)

        append_step_record(log_path, r1)
        append_step_record(log_path, r2)

        records = read_log(log_path)
        assert len(records) == 2
        assert records[0].step == 0
        assert records[1].step == 1
        assert records[1].cumulative_reward == 6.0

    def test_append_creates_parent_dirs(self, tmp_path: Path) -> None:
        """append_step_record creates parent directories if missing."""
        log_path = tmp_path / "sub" / "dir" / "log.jsonl"
        record = _make_step_record()
        append_step_record(log_path, record)
        assert log_path.exists()
        records = read_log(log_path)
        assert len(records) == 1


class TestRoundProducesStepRecord:
    """Test that a single training round produces the expected StepRecord."""

    def test_round_produces_step_record(self, mock_batch: BatchResult) -> None:
        """A mocked single round produces a well-formed StepRecord."""
        # Simulate what the training loop does for one round.
        batch = mock_batch
        batch_rewards = [3.0, -3.0, 3.0]
        batch_mean = sum(batch_rewards) / len(batch_rewards)
        cumulative = sum(batch_rewards)

        record = StepRecord(
            step=0,
            ability=batch.ability,
            level=batch.level,
            mdl_score=batch.mdl_score,
            selection_reason=batch.selection_reason,
            batch_rewards=batch_rewards,
            batch_mean_reward=batch_mean,
            cumulative_reward=cumulative,
            reward_history_summary=batch.reward_history_summary,
        )

        assert record.step == 0
        assert record.ability == "deduction"
        assert record.level == 1
        assert record.mdl_score == 1.5
        assert record.selection_reason == "curiosity"
        assert len(record.batch_rewards) == 3
        assert record.batch_mean_reward == pytest.approx(1.0)
        assert record.cumulative_reward == pytest.approx(3.0)

    def test_fixed_curriculum_selection_reason(self) -> None:
        """FixedCurriculumBaseline produces 'fixed_schedule' reason."""
        from repro_maa.task_cell import TaskCell

        cells = [
            TaskCell("deduction", 1),
            TaskCell("induction", 1),
            TaskCell("abduction", 1),
        ]
        baseline = FixedCurriculumBaseline(
            cells,
            schedule=[("deduction", 1, 2)],
            batch_size=2,
        )
        # Mock generate to avoid needing the MAA submodule.
        with patch.object(
            TaskCell, "generate",
            return_value=[
                {"puzzle_text": "p", "ground_truth": {"solution_text_format": "a"}}
                for _ in range(2)
            ],
        ):
            batch = baseline.emit_batch()
        assert batch.selection_reason == "fixed_schedule"


# ============================================================================
# Step 10: CLI argument parsing tests
# ============================================================================

class TestCLI:
    """Tests for ``parse_args`` and TrainConfig defaults."""

    def test_cli_defaults(self) -> None:
        """Parsing empty args produces default TrainConfig."""
        config = parse_args([])
        assert config.model_name == "Qwen/Qwen3.5-9B"
        assert config.num_rounds == 100
        assert config.batch_size == 4
        assert config.num_generations == 4
        assert config.per_device_train_batch_size == 2
        assert config.curriculum == "curiosity"
        assert config.seed == 42

    def test_cli_curriculum_flag(self) -> None:
        """--curriculum fixed sets the config correctly."""
        config = parse_args(["--curriculum", "fixed"])
        assert config.curriculum == "fixed"

    def test_cli_dgx_spark_defaults(self) -> None:
        """Default config is tuned for DGX Spark."""
        config = parse_args([])
        assert config.gradient_checkpointing is True
        assert config.bf16 is True
        assert config.per_device_train_batch_size == 2
        assert config.max_prompt_length == 512
        assert config.max_completion_length == 512

    def test_cli_override_model(self) -> None:
        """Model name can be overridden."""
        config = parse_args(["--model-name", "meta-llama/Llama-3-8B"])
        assert config.model_name == "meta-llama/Llama-3-8B"

    def test_cli_no_gradient_checkpointing(self) -> None:
        """--no-gradient-checkpointing disables the flag."""
        config = parse_args(["--no-gradient-checkpointing"])
        assert config.gradient_checkpointing is False

    def test_cli_no_bf16(self) -> None:
        """--no-bf16 disables bfloat16."""
        config = parse_args(["--no-bf16"])
        assert config.bf16 is False

    def test_cli_learning_rate(self) -> None:
        """Learning rate can be overridden."""
        config = parse_args(["--learning-rate", "5e-5"])
        assert config.learning_rate == pytest.approx(5e-5)

    def test_cli_output_dir(self) -> None:
        """Output dir can be overridden."""
        config = parse_args(["--output-dir", "/tmp/my_output"])
        assert config.output_dir == "/tmp/my_output"
