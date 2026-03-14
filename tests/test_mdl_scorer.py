# Chunk: docs/chunks/mdl_curiosity_scorer - MDL curiosity scorer unit tests
"""Tests for the MDL-based curiosity scorer."""

import pytest

from repro_maa.mdl_scorer import MDLScorer


@pytest.fixture
def scorer():
    return MDLScorer()


class TestMDLScorerInterface:
    """Success criterion 1: MDLScorer accepts list[float] and returns float."""

    def test_scorer_returns_float(self, scorer):
        result = scorer.score([1.0, 2.0, 3.0])
        assert isinstance(result, float)


class TestMDLScorerMastered:
    """Success criterion 2: Mastered cells score low."""

    def test_mastered_cell_scores_low(self, scorer):
        mastered = scorer.score([3.0] * 20)
        mixed = scorer.score([3.0, -3.0] * 10)
        assert mastered < mixed


class TestMDLScorerUnreachable:
    """Success criterion 3: Unreachable cells score low."""

    def test_unreachable_cell_scores_low(self, scorer):
        unreachable = scorer.score([-3.0] * 20)
        mixed = scorer.score([3.0, -3.0] * 10)
        assert unreachable < mixed


class TestMDLScorerFrontier:
    """Success criterion 4: Frontier cells score highest."""

    def test_frontier_cell_scores_highest(self, scorer):
        frontier = scorer.score([3.0, -3.0] * 10)
        mastered = scorer.score([3.0] * 20)
        unreachable = scorer.score([-3.0] * 20)
        assert frontier > mastered
        assert frontier > unreachable


class TestMDLScorerMonotonic:
    """Success criterion 5: Monotonic decrease toward mastery."""

    def test_monotonic_decrease_toward_mastery(self, scorer):
        scores = []
        for success_pct in [50, 60, 70, 80, 90, 100]:
            n_success = success_pct * 20 // 100
            n_fail = 20 - n_success
            window = [3.0] * n_success + [-3.0] * n_fail
            scores.append(scorer.score(window))

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at {50 + i * 10}% success ({scores[i]}) should be >= "
                f"score at {60 + i * 10}% success ({scores[i + 1]})"
            )


class TestMDLScorerEdgeCases:
    """Success criterion 6: Edge cases handled correctly."""

    def test_empty_window_returns_zero(self, scorer):
        assert scorer.score([]) == 0.0

    def test_single_element_returns_zero(self, scorer):
        assert scorer.score([3.0]) == 0.0

    def test_two_element_identical_returns_zero(self, scorer):
        assert scorer.score([3.0, 3.0]) == 0.0

    def test_two_element_different_returns_positive(self, scorer):
        result = scorer.score([3.0, -3.0])
        assert result > 0.0


class TestMDLScorerDeterminism:
    """Determinism: same inputs produce identical outputs."""

    def test_deterministic(self, scorer):
        window = [3.0, -3.0, 1.0, -1.0, 2.0, -2.0] * 3
        result1 = scorer.score(window)
        result2 = scorer.score(window)
        assert result1 == result2
