# Chunk: docs/chunks/taskcell_abstraction - TaskCell unit and integration tests
"""
Tests for the TaskCell abstraction layer.

Verifies the uniform interface over the three MAA abilities (deduction,
induction, abduction) across all five difficulty levels.
"""
import pytest

from repro_maa.task_cell import TaskCell


# ============================================================================
# Helper
# ============================================================================

def _wrap_response(answer_body: str) -> str:
    """Wrap an answer body in the expected model response format."""
    return f"Assistant: <think>Reasoning here.</think><answer>{answer_body}</answer>"


# ============================================================================
# Construction tests
# ============================================================================

class TestConstruction:
    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_valid_construction(self, ability):
        cell = TaskCell(ability, 1)
        assert cell.ability == ability
        assert cell.level == 1

    def test_invalid_ability_raises(self):
        with pytest.raises(ValueError, match="ability"):
            TaskCell("invalid_ability", 1)

    def test_invalid_level_too_low(self):
        with pytest.raises(ValueError, match="level"):
            TaskCell("deduction", 0)

    def test_invalid_level_too_high(self):
        with pytest.raises(ValueError, match="level"):
            TaskCell("deduction", 6)

    def test_repr(self):
        cell = TaskCell("deduction", 3)
        assert repr(cell) == "TaskCell(ability='deduction', level=3)"


# ============================================================================
# Generate contract tests
# ============================================================================

class TestGenerateContract:
    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_returns_correct_count(self, ability):
        cell = TaskCell(ability, 1, seed=42)
        problems = cell.generate(2)
        assert len(problems) == 2

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_problem_has_required_keys(self, ability):
        cell = TaskCell(ability, 1, seed=42)
        problems = cell.generate(1)
        p = problems[0]
        assert "puzzle_text" in p
        assert isinstance(p["puzzle_text"], str)
        assert len(p["puzzle_text"]) > 0
        assert "ground_truth" in p
        assert isinstance(p["ground_truth"], dict)
        assert "solution_text_format" in p["ground_truth"]


# ============================================================================
# Score contract tests
# ============================================================================

class TestScoreContract:
    def test_deduction_correct_scores_positive(self):
        cell = TaskCell("deduction", 1, seed=42)
        problems = cell.generate(1)
        gt = problems[0]["ground_truth"]
        # Use the solution as the answer (should score positive)
        answer = str(gt["solution_text_format"])
        score = cell.score(_wrap_response(answer), gt)
        assert isinstance(score, float) or isinstance(score, (int, float))
        assert score > 0

    def test_deduction_wrong_scores_negative(self):
        cell = TaskCell("deduction", 1, seed=42)
        problems = cell.generate(1)
        gt = problems[0]["ground_truth"]
        # Mangle the answer
        answer = str(gt["solution_text_format"]).replace("True", "WRONG").replace("False", "WRONG")
        score = cell.score(_wrap_response(answer), gt)
        assert score < 0

    def test_induction_correct_scores_positive(self):
        cell = TaskCell("induction", 1, seed=42)
        problems = cell.generate(1)
        gt = problems[0]["ground_truth"]
        answer = str(gt["solution_text_format"])
        score = cell.score(_wrap_response(answer), gt)
        assert score > 0

    def test_induction_wrong_scores_negative(self):
        cell = TaskCell("induction", 1, seed=42)
        problems = cell.generate(1)
        gt = problems[0]["ground_truth"]
        score = cell.score(_wrap_response("999999"), gt)
        assert score < 0

    def test_abduction_correct_scores_positive(self):
        cell = TaskCell("abduction", 1, seed=42)
        problems = cell.generate(1)
        gt = problems[0]["ground_truth"]
        answer = str(gt["solution_text_format"])
        score = cell.score(_wrap_response(answer), gt)
        assert score > 0

    def test_abduction_wrong_scores_negative(self):
        cell = TaskCell("abduction", 1, seed=42)
        problems = cell.generate(1)
        gt = problems[0]["ground_truth"]
        answer = str(gt["solution_text_format"]).replace("reachable", "WRONG")
        score = cell.score(_wrap_response(answer), gt)
        assert score < 0


# ============================================================================
# Determinism tests
# ============================================================================

class TestDeterminism:
    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_same_seed_produces_identical_output(self, ability):
        cell_a = TaskCell(ability, 1, seed=123)
        cell_b = TaskCell(ability, 1, seed=123)
        problems_a = cell_a.generate(3)
        problems_b = cell_b.generate(3)
        assert problems_a == problems_b


# ============================================================================
# All 15 cells parametrized (slow)
# ============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
class TestAll15Cells:
    def test_generate_one(self, ability, level):
        cell = TaskCell(ability, level, seed=42)
        problems = cell.generate(1)
        assert len(problems) == 1
        p = problems[0]
        assert isinstance(p["puzzle_text"], str)
        assert len(p["puzzle_text"]) > 0
        assert "solution_text_format" in p["ground_truth"]
