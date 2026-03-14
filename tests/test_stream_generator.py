# Chunk: docs/chunks/stream_generator - CuriosityStream unit tests
"""
Tests for the curiosity-driven curriculum stream generator.

Verifies epsilon-greedy selection, metadata completeness, reward history
management, and determinism. Uses synthetic reward histories injected
directly — no MAA generator calls needed for selection logic tests.
"""
from unittest.mock import patch

import pytest

from repro_maa.mdl_scorer import MDLScorer
from repro_maa.stream import BatchResult, CuriosityStream
from repro_maa.task_cell import TaskCell


# ============================================================================
# Fixtures
# ============================================================================

def _make_cells() -> list[TaskCell]:
    """Create the full 3×5 grid of task cells."""
    return [
        TaskCell(ability, level, seed=100 + i)
        for i, (ability, level) in enumerate(
            (a, l)
            for a in ("deduction", "induction", "abduction")
            for l in range(1, 6)
        )
    ]


def _make_stream(
    cells: list[TaskCell] | None = None,
    epsilon: float = 0.0,
    seed: int = 42,
    window_size: int = 20,
    batch_size: int = 8,
) -> CuriosityStream:
    """Create a CuriosityStream with default settings."""
    if cells is None:
        cells = _make_cells()
    return CuriosityStream(
        cells=cells,
        scorer=MDLScorer(),
        batch_size=batch_size,
        epsilon=epsilon,
        window_size=window_size,
        seed=seed,
    )


def _inject_history(
    stream: CuriosityStream,
    cell: TaskCell,
    rewards: list[float],
) -> None:
    """Inject a reward history directly into the stream's internal state."""
    key = (cell.ability, cell.level)
    stream._history[key].extend(rewards)


def _fake_generate(n: int) -> list[dict]:
    """Stub replacement for TaskCell.generate() when MAA deps are unavailable."""
    return [
        {"puzzle_text": f"Problem {i}", "ground_truth": {"solution_text_format": f"Answer {i}"}}
        for i in range(n)
    ]


# ============================================================================
# select_cell tests
# ============================================================================

class TestSelectCell:
    def test_greedy_selects_highest_mdl(self):
        """With epsilon=0, select_cell picks the cell with the highest MDL score."""
        cells = _make_cells()
        stream = _make_stream(cells=cells, epsilon=0.0)

        # Give one cell mixed rewards (high MDL = learning frontier)
        target = cells[0]  # deduction L1
        _inject_history(stream, target, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        # Give all other cells uniform rewards (low MDL = mastered)
        for cell in cells[1:]:
            _inject_history(stream, cell, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        reason, selected, mdl_score = stream.select_cell()
        assert reason == "curiosity"
        assert selected.ability == target.ability
        assert selected.level == target.level
        assert mdl_score > 0.0

    def test_epsilon_zero_is_purely_greedy(self):
        """With epsilon=0, select_cell never picks a non-maximal cell."""
        cells = _make_cells()
        stream = _make_stream(cells=cells, epsilon=0.0)

        target = cells[7]  # induction L3
        _inject_history(stream, target, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        for cell in cells:
            if cell is not target:
                _inject_history(stream, cell, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        for _ in range(20):
            reason, selected, _ = stream.select_cell()
            assert reason == "curiosity"
            assert selected.ability == target.ability
            assert selected.level == target.level

    def test_epsilon_one_is_purely_random(self):
        """With epsilon=1, select_cell covers multiple cells over many draws."""
        cells = _make_cells()
        stream = _make_stream(cells=cells, epsilon=1.0, seed=123)

        # Give one cell high MDL — with epsilon=1, we should still see others
        _inject_history(stream, cells[0], [1.0, 0.0, 1.0, 0.0])
        for cell in cells[1:]:
            _inject_history(stream, cell, [1.0, 1.0, 1.0, 1.0])

        selected_keys = set()
        for _ in range(100):
            reason, selected, _ = stream.select_cell()
            assert reason == "exploration"
            selected_keys.add((selected.ability, selected.level))

        # Should visit multiple distinct cells
        assert len(selected_keys) > 5

    def test_select_cell_deterministic(self):
        """Same seed and history produce identical selection sequences."""
        cells1 = _make_cells()
        cells2 = _make_cells()
        stream1 = _make_stream(cells=cells1, epsilon=0.3, seed=99)
        stream2 = _make_stream(cells=cells2, epsilon=0.3, seed=99)

        # Inject identical histories
        for c1, c2 in zip(cells1, cells2):
            rewards = [float(i % 3) for i in range(10)]
            _inject_history(stream1, c1, rewards)
            _inject_history(stream2, c2, rewards)

        for _ in range(30):
            r1, s1, m1 = stream1.select_cell()
            r2, s2, m2 = stream2.select_cell()
            assert r1 == r2
            assert s1.ability == s2.ability
            assert s1.level == s2.level
            assert m1 == m2


# ============================================================================
# emit_batch tests
# ============================================================================

class TestEmitBatch:
    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_emit_batch_returns_batch_result(self, _mock_gen):
        """emit_batch returns a BatchResult with all required fields."""
        stream = _make_stream(batch_size=4)
        result = stream.emit_batch()
        assert isinstance(result, BatchResult)
        assert result.ability in ("deduction", "induction", "abduction")
        assert 1 <= result.level <= 5
        assert isinstance(result.problems, list)

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_emit_batch_metadata_complete(self, _mock_gen):
        """Every field in BatchResult has the correct type and range."""
        cells = _make_cells()
        stream = _make_stream(cells=cells, batch_size=4)

        # Inject some history so we get a real MDL score
        _inject_history(stream, cells[0], [1.0, 0.0, 1.0, 0.0])
        for cell in cells[1:]:
            _inject_history(stream, cell, [1.0, 1.0, 1.0, 1.0])

        result = stream.emit_batch()

        assert isinstance(result.ability, str)
        assert result.ability in ("deduction", "induction", "abduction")
        assert isinstance(result.level, int)
        assert 1 <= result.level <= 5
        assert isinstance(result.mdl_score, float)
        assert result.mdl_score >= 0.0
        assert result.selection_reason in ("curiosity", "exploration")
        assert isinstance(result.batch_size, int)
        assert result.batch_size == 4
        assert isinstance(result.step, int)
        assert result.step >= 0
        assert isinstance(result.reward_history_summary, dict)
        assert isinstance(result.problems, list)
        assert len(result.problems) == 4

    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_emit_batch_step_increments(self, _mock_gen):
        """Step counter increments monotonically across emit_batch calls."""
        stream = _make_stream(batch_size=2)

        steps = []
        for _ in range(3):
            result = stream.emit_batch()
            steps.append(result.step)

        assert steps == [0, 1, 2]


# ============================================================================
# update tests
# ============================================================================

class TestUpdate:
    def test_update_records_rewards(self):
        """update() stores rewards in the cell's history."""
        cells = _make_cells()
        stream = _make_stream(cells=cells)

        cell = cells[0]
        stream.update(cell, [1.0, 2.0])

        key = (cell.ability, cell.level)
        assert list(stream._history[key]) == [1.0, 2.0]

    def test_update_rolling_window(self):
        """Rewards beyond window_size are evicted."""
        cells = _make_cells()
        stream = _make_stream(cells=cells, window_size=5)

        cell = cells[0]
        stream.update(cell, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        key = (cell.ability, cell.level)
        assert list(stream._history[key]) == [3.0, 4.0, 5.0, 6.0, 7.0]

    def test_update_affects_selection(self):
        """Updating a cell with mixed rewards makes it the greedy selection."""
        cells = _make_cells()
        stream = _make_stream(cells=cells, epsilon=0.0)

        # Start all cells with uniform histories (low MDL)
        for cell in cells:
            _inject_history(stream, cell, [1.0, 1.0, 1.0, 1.0])

        # Update one cell with mixed rewards (high MDL)
        target = cells[5]  # induction L1
        stream.update(target, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        reason, selected, _ = stream.select_cell()
        assert selected.ability == target.ability
        assert selected.level == target.level


# ============================================================================
# Determinism test
# ============================================================================

class TestDeterminism:
    @patch.object(TaskCell, "generate", side_effect=_fake_generate)
    def test_full_sequence_deterministic(self, _mock_gen):
        """Two streams with the same seed produce identical batch sequences."""
        cells1 = _make_cells()
        cells2 = _make_cells()
        stream1 = _make_stream(cells=cells1, epsilon=0.2, seed=77, batch_size=2)
        stream2 = _make_stream(cells=cells2, epsilon=0.2, seed=77, batch_size=2)

        # Inject identical initial histories
        for c1, c2 in zip(cells1, cells2):
            rewards = [float(i % 2) for i in range(8)]
            _inject_history(stream1, c1, rewards)
            _inject_history(stream2, c2, rewards)

        for i in range(10):
            b1 = stream1.emit_batch()
            b2 = stream2.emit_batch()

            assert b1.ability == b2.ability, f"Step {i}: ability mismatch"
            assert b1.level == b2.level, f"Step {i}: level mismatch"
            assert b1.mdl_score == b2.mdl_score, f"Step {i}: mdl_score mismatch"
            assert b1.selection_reason == b2.selection_reason, f"Step {i}: reason mismatch"
            assert b1.step == b2.step, f"Step {i}: step mismatch"

            # Update both streams with the same synthetic rewards
            target1 = next(c for c in cells1 if c.ability == b1.ability and c.level == b1.level)
            target2 = next(c for c in cells2 if c.ability == b2.ability and c.level == b2.level)
            rewards = [float(i % 3), float((i + 1) % 3)]
            stream1.update(target1, rewards)
            stream2.update(target2, rewards)
