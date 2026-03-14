# Chunk: docs/chunks/prompt_reward_bridge - Prompt formatting and reward adaptation tests
"""
Tests for the prompt_reward_bridge module.

Verifies prompt formatting (TaskCell problem → chat messages) and the
TRL-compatible reward function adapter.
"""
import pytest

from repro_maa.prompt_reward_bridge import format_chat_prompt, make_reward_func
from repro_maa.task_cell import TaskCell


# ============================================================================
# Helpers
# ============================================================================

def _wrap_response(answer_body: str) -> str:
    """Wrap an answer body in the expected MAA model response format."""
    return f"Assistant: <think>Reasoning here.</think><answer>{answer_body}</answer>"


# ============================================================================
# format_chat_prompt tests
# ============================================================================

class TestFormatChatPrompt:
    """Tests for :func:`format_chat_prompt`."""

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_returns_two_message_list(self, ability, sample_problems):
        messages = format_chat_prompt(sample_problems[ability])
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_system_message_contains_think_answer_instructions(self, ability, sample_problems):
        messages = format_chat_prompt(sample_problems[ability])
        system_content = messages[0]["content"]
        assert "<think>" in system_content
        assert "</think>" in system_content
        assert "<answer>" in system_content
        assert "</answer>" in system_content

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_user_message_contains_puzzle_text(self, ability, sample_problems):
        problem = sample_problems[ability]
        messages = format_chat_prompt(problem)
        assert messages[1]["content"] == problem["puzzle_text"]

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_all_abilities_produce_valid_prompts(self, ability, sample_problems):
        messages = format_chat_prompt(sample_problems[ability])
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert isinstance(msg["role"], str)
            assert isinstance(msg["content"], str)
            assert len(msg["content"]) > 0

    @pytest.mark.parametrize("level", [1, 2, 3])
    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_different_difficulty_levels(self, ability, level):
        cell = TaskCell(ability, level=level, seed=42)
        problems = cell.generate(1)
        messages = format_chat_prompt(problems[0])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert len(messages[1]["content"]) > 0


# ============================================================================
# make_reward_func tests
# ============================================================================

class TestMakeRewardFunc:
    """Tests for :func:`make_reward_func`."""

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_reward_func_returns_list_of_floats(
        self, ability, sample_problems, correct_completions
    ):
        reward_fn = make_reward_func(ability)
        problem = sample_problems[ability]
        scores = reward_fn(
            completions=[correct_completions[ability]],
            ground_truth=[problem["ground_truth"]],
        )
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_reward_parity_with_task_cell_score(
        self, ability, sample_problems, correct_completions
    ):
        """Adapted reward function produces identical scores to TaskCell.score()."""
        cell = TaskCell(ability, level=1, seed=42)
        problem = sample_problems[ability]
        completion = correct_completions[ability]

        expected = cell.score(completion, problem["ground_truth"])
        reward_fn = make_reward_func(ability)
        actual = reward_fn(
            completions=[completion],
            ground_truth=[problem["ground_truth"]],
        )
        assert actual[0] == pytest.approx(expected)

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_reward_parity_wrong_answers(
        self, ability, sample_problems, wrong_completions
    ):
        """Wrong answers produce matching negative scores through both paths."""
        cell = TaskCell(ability, level=1, seed=42)
        problem = sample_problems[ability]
        completion = wrong_completions[ability]

        expected = cell.score(completion, problem["ground_truth"])
        reward_fn = make_reward_func(ability)
        actual = reward_fn(
            completions=[completion],
            ground_truth=[problem["ground_truth"]],
        )
        assert actual[0] == pytest.approx(expected)
        assert actual[0] < 0

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_multiple_completions_scored_independently(
        self, ability, sample_problems, correct_completions, wrong_completions
    ):
        """A batch of completions is scored element-wise."""
        problem = sample_problems[ability]
        gt = problem["ground_truth"]
        correct = correct_completions[ability]
        wrong = wrong_completions[ability]

        reward_fn = make_reward_func(ability)
        scores = reward_fn(
            completions=[correct, wrong, correct],
            ground_truth=[gt, gt, gt],
        )
        assert len(scores) == 3
        assert scores[0] > 0  # correct
        assert scores[1] < 0  # wrong
        assert scores[2] > 0  # correct

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_trl_calling_convention(
        self, ability, sample_problems, correct_completions
    ):
        """Reward function works with TRL's exact calling convention.

        TRL's GRPOTrainer calls: reward_func(completions=..., ground_truth=..., **extra)
        """
        problem = sample_problems[ability]
        reward_fn = make_reward_func(ability)
        # Simulate TRL passing extra kwargs (e.g., other dataset columns)
        scores = reward_fn(
            completions=[correct_completions[ability]],
            ground_truth=[problem["ground_truth"]],
            some_extra_column=["ignored_value"],
        )
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_invalid_ability_raises(self):
        with pytest.raises(ValueError, match="ability"):
            make_reward_func("nonexistent")


# ============================================================================
# Completion format handling
# ============================================================================

class TestCompletionFormatHandling:
    """Verify the adapter handles both raw and prefixed completions."""

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_raw_completion_without_assistant_prefix(
        self, ability, sample_problems
    ):
        """Completions without 'Assistant:' prefix are handled correctly."""
        problem = sample_problems[ability]
        solution = problem["ground_truth"]["solution_text_format"]
        # Raw completion without "Assistant: " prefix — adapter should add it
        raw_completion = f"<think>Let me reason.</think><answer>{solution}</answer>"

        reward_fn = make_reward_func(ability)
        scores = reward_fn(
            completions=[raw_completion],
            ground_truth=[problem["ground_truth"]],
        )
        assert scores[0] > 0

    @pytest.mark.parametrize("ability", ["deduction", "induction", "abduction"])
    def test_prefixed_completion_passed_through(
        self, ability, sample_problems, correct_completions
    ):
        """Completions with 'Assistant:' prefix are passed through unchanged."""
        problem = sample_problems[ability]
        reward_fn = make_reward_func(ability)
        scores = reward_fn(
            completions=[correct_completions[ability]],
            ground_truth=[problem["ground_truth"]],
        )
        assert scores[0] > 0
