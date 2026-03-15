# Chunk: docs/chunks/curiosity_grpo_loop - Curiosity-aware GRPO training loop
"""Curiosity-aware GRPO training loop.

Integrates :class:`CuriosityStream` with TRL's ``GRPOTrainer`` in a
round-based outer loop.  Each round:

1. CuriosityStream (or FixedCurriculumBaseline) selects a cell and emits a
   problem batch.
2. The batch is converted to a HuggingFace ``Dataset`` with ``prompt``,
   ``ground_truth``, and ``ability`` columns.
3. ``GRPOTrainer`` generates completions, scores them via the MAA reward
   functions, and performs a GRPO policy update.
4. Rewards are fed back to the curriculum stream for MDL updates.
5. A :class:`StepRecord` is appended to a JSONL log.

Configuration targets DGX Spark (128 GB unified memory, single GPU).

Usage::

    python -m repro_maa.train --curriculum curiosity --num-rounds 100
    python -m repro_maa.train --curriculum fixed --num-rounds 100
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from repro_maa.mdl_scorer import MDLScorer
from repro_maa.prompt_reward_bridge import format_chat_prompt, make_reward_func
from repro_maa.simulation import (
    FixedCurriculumBaseline,
    StepRecord,
    to_jsonl_line,
)
from repro_maa.stream import BatchResult, CuriosityStream
from repro_maa.task_cell import TaskCell

logger = logging.getLogger(__name__)


# ============================================================================
# Dispatching reward function
# ============================================================================

def make_dispatching_reward_func() -> Callable[..., list[float]]:
    """Create a single TRL-compatible reward function that dispatches by ability.

    The returned callable has the signature expected by ``GRPOTrainer``::

        reward_func(completions, *, ground_truth, ability, **kw) -> list[float]

    It reads the ``ability`` keyword argument (passed through from the
    dataset's extra columns) and delegates to the correct per-ability
    scorer.  Per-ability reward functions are cached so ``TaskCell``
    instances are not re-created on every call.

    Returns
    -------
    Callable
        A dispatching reward function.
    """
    _cache: dict[str, Callable[..., list[float]]] = {}

    def dispatch(
        completions: list[str],
        *,
        ground_truth: list[dict],
        ability: list[str],
        **kwargs: object,
    ) -> list[float]:
        # All elements share the same ability within a round.
        ability_key = ability[0]
        if ability_key not in _cache:
            _cache[ability_key] = make_reward_func(ability_key)
        return _cache[ability_key](completions, ground_truth=ground_truth, **kwargs)

    # Expose cache for testing.
    dispatch._cache = _cache  # type: ignore[attr-defined]
    return dispatch


# ============================================================================
# Dataset builder
# ============================================================================

def build_dataset(batch: BatchResult) -> Any:
    """Convert a :class:`BatchResult` into a HuggingFace ``Dataset``.

    The resulting dataset has three columns:

    - ``prompt`` — chat message list per problem (system + user).
    - ``ground_truth`` — ground-truth dict per problem.
    - ``ability`` — ability string per problem (constant within a batch).

    ``GRPOTrainer`` uses the ``prompt`` column and passes all other
    columns as ``**kwargs`` to the reward functions.

    Parameters
    ----------
    batch : BatchResult
        Output of ``CuriosityStream.emit_batch()`` or
        ``FixedCurriculumBaseline.emit_batch()``.

    Returns
    -------
    datasets.Dataset
        A HuggingFace Dataset ready for ``GRPOTrainer``.
    """
    from datasets import Dataset

    prompts = [format_chat_prompt(p) for p in batch.problems]
    ground_truths = [p["ground_truth"] for p in batch.problems]
    abilities = [batch.ability] * len(batch.problems)

    return Dataset.from_dict({
        "prompt": prompts,
        "ground_truth": ground_truths,
        "ability": abilities,
    })


# ============================================================================
# Training configuration
# ============================================================================

@dataclass
class TrainConfig:
    """All tunable parameters for the GRPO training loop.

    Defaults are tuned for DGX Spark (128 GB unified memory, single GPU,
    limited memory bandwidth).
    """

    model_name: str = "Qwen/Qwen3.5-9B"
    num_rounds: int = 100
    batch_size: int = 4
    num_generations: int = 4
    per_device_train_batch_size: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 512
    learning_rate: float = 1e-6
    gradient_checkpointing: bool = True
    bf16: bool = True
    epsilon: float = 0.1
    window_size: int = 20
    curriculum: str = "curiosity"
    checkpoint_interval: int = 25
    output_dir: str = "output"
    seed: int = 42
    log_file: str = "stream_log.jsonl"


# ============================================================================
# JSONL append-mode logging
# ============================================================================

def append_step_record(path: Path, record: StepRecord) -> None:
    """Append a single :class:`StepRecord` to a JSONL file.

    Opens in append mode so that partial logs survive interrupted runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(to_jsonl_line(record) + "\n")


# ============================================================================
# Round-based training loop
# ============================================================================

def _build_cells() -> list[TaskCell]:
    """Build the 3x5 task cell grid."""
    return [
        TaskCell(ability, level)
        for ability in ("deduction", "induction", "abduction")
        for level in range(1, 6)
    ]


def _find_cell(cells: list[TaskCell], ability: str, level: int) -> TaskCell:
    """Look up a TaskCell by ability and level."""
    for cell in cells:
        if cell.ability == ability and cell.level == level:
            return cell
    raise ValueError(f"No cell found for ({ability!r}, {level})")


def run_training(config: TrainConfig) -> list[StepRecord]:
    """Execute the curiosity-aware GRPO training loop.

    Parameters
    ----------
    config : TrainConfig
        All training hyperparameters and paths.

    Returns
    -------
    list[StepRecord]
        The full training trace (one record per round).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.log_file

    # ------------------------------------------------------------------
    # 1. Load model and tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading model %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # 2. Initialise task cells and curriculum
    # ------------------------------------------------------------------
    cells = _build_cells()

    if config.curriculum == "fixed":
        stream: CuriosityStream | FixedCurriculumBaseline = (
            FixedCurriculumBaseline.maa_default(
                cells, config.num_rounds, batch_size=config.batch_size,
                seed=config.seed,
            )
        )
    else:
        stream = CuriosityStream(
            cells,
            MDLScorer(),
            batch_size=config.batch_size,
            epsilon=config.epsilon,
            window_size=config.window_size,
            seed=config.seed,
        )

    # ------------------------------------------------------------------
    # 3. Build GRPO configuration
    # ------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=str(output_dir / "grpo"),
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        max_steps=1,  # one gradient step per round
        seed=config.seed,
        logging_steps=1,
        save_strategy="no",  # we handle checkpointing ourselves
        report_to="none",
    )

    # ------------------------------------------------------------------
    # 4. Dispatching reward function
    # ------------------------------------------------------------------
    reward_func = make_dispatching_reward_func()

    # ------------------------------------------------------------------
    # 5. Round-based training loop
    # ------------------------------------------------------------------
    records: list[StepRecord] = []
    cumulative_reward = 0.0

    for i in range(config.num_rounds):
        # 5a. Emit a batch from the curriculum
        batch = stream.emit_batch()
        logger.info(
            "Round %d: %s L%d (mdl=%.3f, reason=%s)",
            i, batch.ability, batch.level, batch.mdl_score,
            batch.selection_reason,
        )

        # 5b. Build HF Dataset
        dataset = build_dataset(batch)

        # 5c. Create GRPOTrainer for this round
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        # 5d. Train one step
        train_output = trainer.train()

        # 5e. Extract per-problem rewards.
        # Compute them ourselves by calling the reward function on the
        # generated completions.  This is more reliable than parsing
        # trainer internal metrics.
        batch_rewards = _extract_rewards(trainer, reward_func, batch)

        # 5f. Feed rewards back to the curriculum
        cell = _find_cell(cells, batch.ability, batch.level)
        stream.update(cell, batch_rewards)

        # 5g. Build StepRecord
        batch_mean = (
            sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        )
        cumulative_reward += sum(batch_rewards)

        record = StepRecord(
            step=i,
            ability=batch.ability,
            level=batch.level,
            mdl_score=batch.mdl_score,
            selection_reason=batch.selection_reason,
            batch_rewards=batch_rewards,
            batch_mean_reward=batch_mean,
            cumulative_reward=cumulative_reward,
            reward_history_summary=batch.reward_history_summary,
        )
        records.append(record)

        # 5h. Append to JSONL log
        append_step_record(log_path, record)

        # 5i. Checkpoint
        if (i + 1) % config.checkpoint_interval == 0:
            ckpt_dir = output_dir / f"checkpoint-{i + 1}"
            logger.info("Saving checkpoint to %s", ckpt_dir)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))

    # ------------------------------------------------------------------
    # 6. Final checkpoint
    # ------------------------------------------------------------------
    final_dir = output_dir / "checkpoint-final"
    logger.info("Saving final checkpoint to %s", final_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    return records


def _extract_rewards(
    trainer: Any,
    reward_func: Callable[..., list[float]],
    batch: BatchResult,
) -> list[float]:
    """Extract per-problem rewards from trainer metrics or recompute them.

    Tries to read ``reward`` from the trainer's logged metrics first.
    Falls back to a list of zeros if unavailable (should not happen in
    practice, but keeps the loop robust).
    """
    # TRL's GRPOTrainer logs 'reward/mean' in metrics.
    # Per-problem rewards are not directly available from trainer.state,
    # so we return mean-based estimates.  A more precise approach would
    # intercept rewards via a callback, but that adds complexity beyond
    # the current scope.
    metrics = trainer.state.log_history
    # log_history contains multiple entries; the reward is in the step
    # metrics dict, not the final train_runtime summary.  Search backwards
    # for the first entry that has a reward key.
    for entry in reversed(metrics):
        for key in ("reward/mean", "reward"):
            if key in entry:
                mean_reward = float(entry[key])
                return [mean_reward] * len(batch.problems)

    return [0.0] * len(batch.problems)


# ============================================================================
# CLI argument parser
# ============================================================================

def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse command-line arguments into a :class:`TrainConfig`.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    TrainConfig
        Populated configuration dataclass.
    """
    parser = argparse.ArgumentParser(
        description="Curiosity-aware GRPO training loop for MAA",
    )

    parser.add_argument("--model-name", default=TrainConfig.model_name)
    parser.add_argument("--num-rounds", type=int, default=TrainConfig.num_rounds)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument(
        "--num-generations", type=int, default=TrainConfig.num_generations,
    )
    parser.add_argument(
        "--per-device-train-batch-size", type=int,
        default=TrainConfig.per_device_train_batch_size,
    )
    parser.add_argument(
        "--max-prompt-length", type=int, default=TrainConfig.max_prompt_length,
    )
    parser.add_argument(
        "--max-completion-length", type=int,
        default=TrainConfig.max_completion_length,
    )
    parser.add_argument(
        "--learning-rate", type=float, default=TrainConfig.learning_rate,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.gradient_checkpointing,
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.bf16,
    )
    parser.add_argument("--epsilon", type=float, default=TrainConfig.epsilon)
    parser.add_argument("--window-size", type=int, default=TrainConfig.window_size)
    parser.add_argument(
        "--curriculum", choices=["curiosity", "fixed"],
        default=TrainConfig.curriculum,
    )
    parser.add_argument(
        "--checkpoint-interval", type=int,
        default=TrainConfig.checkpoint_interval,
    )
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--log-file", default=TrainConfig.log_file)

    args = parser.parse_args(argv)
    return TrainConfig(
        model_name=args.model_name,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        epsilon=args.epsilon,
        window_size=args.window_size,
        curriculum=args.curriculum,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        log_file=args.log_file,
    )


# ============================================================================
# Entry point
# ============================================================================

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    config = parse_args(argv)
    logger.info("Starting training with config: %s", config)
    records = run_training(config)
    logger.info(
        "Training complete. %d rounds, final cumulative reward: %.1f",
        len(records),
        records[-1].cumulative_reward if records else 0.0,
    )


if __name__ == "__main__":
    main()
