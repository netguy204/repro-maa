#!/usr/bin/env python3
"""Diagnostic benchmark for repro-maa training pipeline.

Runs a training loop and checks for failure modes that prevent learning.

Two modes:
  --smoke  (default) 3 rounds, ~20 min. Validates environment, reward
           variance, gradient signal, completion quality, memory stability.
  --full   15 rounds, ~1-2 hours. All smoke checks plus reward_trend
           (proves the policy is actually learning over time).

Usage:
    uv run auto/bench.py --smoke    # quick validation
    uv run auto/bench.py --full     # prove learning works

Exit code 0 = all checks pass, 1 = at least one failure.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Defaults for smoke mode; full mode overrides NUM_ROUNDS
NUM_ROUNDS_SMOKE = 3
NUM_ROUNDS_FULL = 15
NUM_GENERATIONS = 16
BATCH_SIZE = 1
MAX_COMPLETION_LENGTH = 2048
PER_DEVICE_BATCH_SIZE = 8
MEMORY_GROWTH_THRESHOLD_GB = 2.0  # max acceptable growth between rounds


def check_environment() -> tuple[bool, str]:
    """Check CUDA, model loading, and key imports."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, f"CUDA not available (torch {torch.__version__})"
        device = torch.cuda.get_device_name(0)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from repro_maa.train import run_training, TrainConfig
        from repro_maa.prompt_reward_bridge import format_chat_prompt, make_reward_func
        from repro_maa.task_cell import TaskCell

        return True, f"torch {torch.__version__}, CUDA on {device}"
    except Exception as e:
        return False, str(e)


def run_bench(num_rounds: int = NUM_ROUNDS_SMOKE) -> dict:
    """Run the benchmark and return structured results."""
    results = {}
    is_full = num_rounds >= NUM_ROUNDS_FULL

    # --- Check 1: Environment ---
    ok, detail = check_environment()
    results["environment"] = {"pass": ok, "detail": detail}
    if not ok:
        # Can't continue without environment
        for key in ("problem_diversity", "reward_variance", "gradient_signal",
                     "completion_quality", "memory_stability"):
            results[key] = {"pass": False, "detail": "skipped (environment failed)"}
        return results

    import torch
    from repro_maa.train import run_training, TrainConfig, _extract_rewards, build_dataset
    from repro_maa.train import _build_cells, _find_cell, make_dispatching_reward_func
    from repro_maa.stream import CuriosityStream
    from repro_maa.mdl_scorer import MDLScorer
    from repro_maa.task_cell import TaskCell
    from repro_maa.prompt_reward_bridge import _extract_answer, _EXPECTED_TAGS
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Use a temp directory for output
    import tempfile
    output_dir = Path(tempfile.mkdtemp(prefix="bench_"))

    config = TrainConfig(
        num_rounds=num_rounds,
        batch_size=BATCH_SIZE,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        max_completion_length=MAX_COMPLETION_LENGTH,
        output_dir=str(output_dir),
        curriculum="curiosity",
    )

    # Load model once
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    cells = _build_cells()
    stream = CuriosityStream(
        cells, MDLScorer(),
        batch_size=config.batch_size,
        epsilon=config.epsilon,
        window_size=config.window_size,
        seed=config.seed,
    )

    grpo_config = GRPOConfig(
        output_dir=str(output_dir / "grpo"),
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        generation_batch_size=config.num_generations,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        max_steps=1,
        seed=config.seed,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    reward_func = make_dispatching_reward_func()

    # Tracking
    ground_truths = []
    reward_stds = []
    losses = []
    grads = []
    clipped_ratios = []
    mean_rewards = []  # per-round mean reward for trend analysis
    memory_per_round = []

    for i in range(num_rounds):
        print(f"\n--- Round {i} ---")

        batch = stream.emit_batch()
        print(f"  {batch.ability} L{batch.level}")

        # Track ground truths for diversity check
        for p in batch.problems:
            gt = p.get("ground_truth", {})
            ground_truths.append(str(gt))

        dataset = build_dataset(batch)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        train_output = trainer.train()

        # Extract metrics from log_history
        metrics = trainer.state.log_history
        for entry in reversed(metrics):
            if "reward_std" in entry:
                reward_stds.append(float(entry["reward_std"]))
                losses.append(float(entry.get("loss", 0)))
                grads.append(float(entry.get("grad_norm", 0)))
                clipped_ratios.append(float(entry.get("completions/clipped_ratio", 1.0)))
                mean_rewards.append(float(entry.get("reward", 0)))
                break

        # Extract rewards for stream update
        batch_rewards = _extract_rewards(trainer, reward_func, batch)
        cell = _find_cell(cells, batch.ability, batch.level)
        stream.update(cell, batch_rewards)

        # Cleanup
        del trainer, train_output, dataset
        gc.collect()
        torch.cuda.empty_cache()

        # Memory tracking
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        memory_per_round.append(alloc_gb)
        print(f"  reward_std={reward_stds[-1] if reward_stds else '?'}, "
              f"loss={losses[-1] if losses else '?'}, "
              f"memory={alloc_gb:.1f}GB")

    # --- Check 2: Problem diversity ---
    unique_gts = len(set(ground_truths))
    total_gts = len(ground_truths)
    div_ok = unique_gts > 1
    results["problem_diversity"] = {
        "pass": div_ok,
        "detail": f"{unique_gts} unique ground truths out of {total_gts}",
    }

    # --- Check 3: Reward variance ---
    mean_std = sum(reward_stds) / len(reward_stds) if reward_stds else 0
    var_ok = mean_std > 0.01
    results["reward_variance"] = {
        "pass": var_ok,
        "detail": f"mean reward_std={mean_std:.4f} across {len(reward_stds)} rounds",
    }

    # --- Check 4: Gradient signal ---
    mean_loss = sum(abs(x) for x in losses) / len(losses) if losses else 0
    mean_grad = sum(grads) / len(grads) if grads else 0
    any_nonzero = any(abs(l) > 1e-10 for l in losses)
    grad_ok = any_nonzero and mean_grad > 0
    results["gradient_signal"] = {
        "pass": grad_ok,
        "detail": f"mean |loss|={mean_loss:.6f}, mean grad_norm={mean_grad:.4f}, "
                  f"any_nonzero_loss={any_nonzero}",
    }

    # --- Check 5: Completion quality ---
    mean_clipped = sum(clipped_ratios) / len(clipped_ratios) if clipped_ratios else 1.0
    clip_ok = mean_clipped < 0.95
    results["completion_quality"] = {
        "pass": clip_ok,
        "detail": f"mean clipped_ratio={mean_clipped:.3f}",
    }

    # --- Check 6: Memory stability ---
    if len(memory_per_round) >= 2:
        delta = memory_per_round[-1] - memory_per_round[0]
        mem_ok = delta < MEMORY_GROWTH_THRESHOLD_GB
        results["memory_stability"] = {
            "pass": mem_ok,
            "detail": f"round0={memory_per_round[0]:.1f}GB, "
                      f"round{len(memory_per_round)-1}={memory_per_round[-1]:.1f}GB, "
                      f"delta={delta:.1f}GB",
        }
    else:
        results["memory_stability"] = {"pass": True, "detail": "insufficient rounds"}

    # --- Check 7 (full mode only): Reward trend ---
    if is_full and len(mean_rewards) >= 10:
        window = max(len(mean_rewards) // 3, 3)
        early = sum(mean_rewards[:window]) / window
        late = sum(mean_rewards[-window:]) / window
        improving = late > early
        results["reward_trend"] = {
            "pass": improving,
            "detail": f"early_{window}={early:.3f}, late_{window}={late:.3f}, "
                      f"delta={late - early:+.3f}",
        }
    elif is_full:
        results["reward_trend"] = {
            "pass": False,
            "detail": f"insufficient rounds ({len(mean_rewards)}) for trend analysis",
        }

    # Compute val_metric (mean reward over last 5 rounds — where the
    # policy ended up, not diluted by early untrained rounds)
    if mean_rewards:
        tail = mean_rewards[-5:]
        results["val_metric"] = sum(tail) / len(tail)

    # Per-round reward progression for the log
    if mean_rewards:
        results["reward_progression"] = mean_rewards

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_results(results: dict) -> bool:
    """Print structured results and return overall pass/fail."""
    print("\n--- bench results ---")
    all_pass = True
    checks = ["environment", "problem_diversity", "reward_variance",
              "gradient_signal", "completion_quality", "memory_stability"]
    if "reward_trend" in results:
        checks.append("reward_trend")
    for key in checks:
        r = results.get(key, {"pass": False, "detail": "not run"})
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        print(f"{key:25s} {status:4s}  ({r['detail']})")
    print("---")
    print(f"overall: {'PASS' if all_pass else 'FAIL'}")
    if "val_metric" in results:
        print(f"val_metric: {results['val_metric']}")
    if "reward_progression" in results:
        rewards = results["reward_progression"]
        print(f"reward_progression: {[f'{r:.2f}' for r in rewards]}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="repro-maa training benchmark")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", default=True,
                       help="Quick 3-round validation (default)")
    group.add_argument("--full", action="store_true",
                       help="15-round learning verification")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override number of rounds")
    args = parser.parse_args()

    if args.rounds is not None:
        num_rounds = args.rounds
    elif args.full:
        num_rounds = NUM_ROUNDS_FULL
    else:
        num_rounds = NUM_ROUNDS_SMOKE

    mode = "full" if num_rounds >= NUM_ROUNDS_FULL else "smoke"
    print(f"=== bench mode: {mode} ({num_rounds} rounds) ===")

    results = run_bench(num_rounds)
    passed = print_results(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
