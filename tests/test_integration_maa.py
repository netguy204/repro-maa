# Chunk: docs/chunks/scaffold_project - Integration tests for MAA generators and scorers
"""
Integration tests verifying that we can import and invoke the MAA data
generators and reward scoring functions through the compatibility shim.

Per TESTING_PHILOSOPHY.md: "we test that we can import and invoke it, not that
it's correct. Our boundary is the wrapper, not the internals."
"""
import pytest

from repro_maa.maa_compat import (
    DeductionSampler,
    DeductionFormatter,
    InductionGenerator,
    generate_abduction_problem,
    deduction_score,
    induction_score,
    abduction_score,
    mixed_score,
)

# ============================================================================
# Data generator tests (Success criteria #2)
# ============================================================================


class TestDeductionGenerator:
    def test_produces_valid_problem(self):
        sampler = DeductionSampler(difficulty=1, seed=42)
        puzzles = sampler.sample_unique(1)

        assert len(puzzles) == 1
        formulas, assignment = puzzles[0]
        assert isinstance(formulas, list)
        assert len(formulas) > 0
        assert all(isinstance(f, str) for f in formulas)
        assert isinstance(assignment, dict)
        assert all(isinstance(v, bool) for v in assignment.values())

    def test_formatter_produces_text(self):
        sampler = DeductionSampler(difficulty=1, seed=42)
        puzzles = sampler.sample_unique(1)
        formulas, assignment = puzzles[0]
        fmt = DeductionFormatter(formulas, assignment)
        assert "Below are" in fmt.puzzle_text()
        assert "is True" in fmt.solution_text() or "is False" in fmt.solution_text()


class TestInductionGenerator:
    def test_produces_valid_problem(self):
        gen = InductionGenerator(seed=42)
        puzzles = gen.generate_puzzles(num=1, level=1)

        assert len(puzzles) == 1
        p = puzzles[0]
        assert "puzzle_text" in p
        assert "solution_text" in p
        assert "complete_sequence" in p

    def test_solution_is_numeric(self):
        gen = InductionGenerator(seed=42)
        puzzles = gen.generate_puzzles(num=1, level=1)
        # solution_text should be an int (the final sequence element)
        assert isinstance(puzzles[0]["solution_text"], int)


class TestAbductionGenerator:
    def test_produces_valid_problem(self):
        problem = generate_abduction_problem(
            problem_id=1,
            num_goals=1,
            reachable_k=1,
            chain_depth=2,
            distractors=3,
            cycle_prob=0.1,
        )

        assert isinstance(problem, dict)
        for key in ("premises", "known_atoms", "goals", "reachable_goals", "unreachable_goals"):
            assert key in problem, f"Missing key: {key}"
        assert isinstance(problem["premises"], list)
        assert isinstance(problem["goals"], list)
        assert len(problem["reachable_goals"]) == 1


# Parametrized multi-level test (slow)
@pytest.mark.slow
@pytest.mark.parametrize("level", [1, 2, 3])
class TestGeneratorsAtMultipleLevels:
    def test_deduction(self, level):
        sampler = DeductionSampler(difficulty=level, seed=42)
        puzzles = sampler.sample_unique(1)
        assert len(puzzles) == 1

    def test_induction(self, level):
        gen = InductionGenerator(seed=42)
        puzzles = gen.generate_puzzles(num=1, level=level)
        assert len(puzzles) == 1

    def test_abduction(self, level):
        from repro_maa.maa_compat import generate_abduction_problem as gen_abd

        # Use conservative params for speed
        problem = gen_abd(
            problem_id=1,
            num_goals=1,
            reachable_k=1,
            chain_depth=level + 1,
            distractors=3,
            cycle_prob=0.1,
        )
        assert "premises" in problem


# ============================================================================
# Reward scorer tests (Success criteria #3)
# ============================================================================

# All scorers expect a solution_str with an "Assistant:" header and
# <think>...</think><answer>...</answer> structure.

def _wrap_response(answer_body: str) -> str:
    """Wrap an answer body in the expected model response format."""
    return f"Assistant: <think>Reasoning here.</think><answer>{answer_body}</answer>"


class TestDeductionScore:
    def test_correct_answer(self):
        gt = {
            "solution_text_format": "(1) A is True\n(2) B is False\n(3) C is True\n(4) D is False"
        }
        answer = "(1) A is True\n(2) B is False\n(3) C is True\n(4) D is False"
        score = deduction_score(_wrap_response(answer), gt)
        assert score > 0, f"Expected positive score for correct answer, got {score}"

    def test_wrong_answer(self):
        gt = {
            "solution_text_format": "(1) A is True\n(2) B is False\n(3) C is True\n(4) D is False"
        }
        # Flip all values
        answer = "(1) A is False\n(2) B is True\n(3) C is False\n(4) D is True"
        score = deduction_score(_wrap_response(answer), gt)
        assert score < 0, f"Expected negative score for wrong answer, got {score}"


class TestInductionScore:
    def test_correct_answer(self):
        gt = {"solution_text_format": 42}
        score = induction_score(_wrap_response("42"), gt)
        assert score > 0, f"Expected positive score for correct answer, got {score}"

    def test_wrong_answer(self):
        gt = {"solution_text_format": 42}
        score = induction_score(_wrap_response("99"), gt)
        assert score < 0, f"Expected negative score for wrong answer, got {score}"


class TestAbductionScore:
    def test_correct_answer(self):
        gt = {
            "solution_text_format": "(1) A is reachable\n(2) B is unreachable"
        }
        answer = "(1) A is reachable\n(2) B is unreachable"
        score = abduction_score(_wrap_response(answer), gt)
        assert score > 0, f"Expected positive score for correct answer, got {score}"

    def test_wrong_answer(self):
        gt = {
            "solution_text_format": "(1) A is reachable\n(2) B is unreachable"
        }
        answer = "(1) A is unreachable\n(2) B is reachable"
        score = abduction_score(_wrap_response(answer), gt)
        assert score < 0, f"Expected negative score for wrong answer, got {score}"


class TestMixedScore:
    def test_routes_deduction(self):
        gt = {
            "solution_text_format": "(1) A is True\n(2) B is False"
        }
        answer = "(1) A is True\n(2) B is False"
        resp = _wrap_response(answer)
        assert mixed_score(resp, gt) == deduction_score(resp, gt)

    def test_routes_induction(self):
        gt = {"solution_text_format": 42}
        resp = _wrap_response("42")
        assert mixed_score(resp, gt) == induction_score(resp, gt)

    def test_routes_abduction(self):
        gt = {
            "solution_text_format": "(1) A is reachable\n(2) B is unreachable"
        }
        answer = "(1) A is reachable\n(2) B is unreachable"
        resp = _wrap_response(answer)
        assert mixed_score(resp, gt) == abduction_score(resp, gt)
