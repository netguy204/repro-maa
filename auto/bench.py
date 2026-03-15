#!/usr/bin/env python3
"""Diagnostic benchmark for repro-maa training pipeline.

Runs a short training loop (3 rounds) and checks for the failure modes
that prevent productive learning:

1. Environment health (CUDA, model loading, imports)
2. Problem diversity (different puzzles each round)
3. Reward variance (GRPO needs std > 0 for gradients)
4. Gradient signal (non-zero loss and grad_norm)
5. Completion quality (not all truncated, answers parseable)
6. Memory stability (no leak between rounds)

Usage:
    uv run auto/bench.py

Exit code 0 = all checks pass, 1 = at least one failure.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

NUM_ROUNDS = 3
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


def run_bench() -> dict:
    """Run the benchmark and return structured results."""
    results = {}

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
        num_rounds=NUM_ROUNDS,
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
    answers_found = 0
    total_completions = 0
    memory_per_round = []

    for i in range(NUM_ROUNDS):
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

    # Compute a val_metric (mean reward across all rounds)
    if reward_stds:
        # Use the mean reward from the last round's metrics
        for entry in reversed(metrics):
            if "reward" in entry:
                results["val_metric"] = float(entry["reward"])
                break

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_results(results: dict) -> bool:
    """Print structured results and return overall pass/fail."""
    print("\n--- bench results ---")
    all_pass = True
    for key in ("environment", "problem_diversity", "reward_variance",
                "gradient_signal", "completion_quality", "memory_stability"):
        r = results.get(key, {"pass": False, "detail": "not run"})
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        print(f"{key:25s} {status:4s}  ({r['detail']})")
    print("---")
    print(f"overall: {'PASS' if all_pass else 'FAIL'}")
    if "val_metric" in results:
        print(f"val_metric: {results['val_metric']}")
    return all_pass


if __name__ == "__main__":
    results = run_bench()
    passed = print_results(results)
    sys.exit(0 if passed else 1)
