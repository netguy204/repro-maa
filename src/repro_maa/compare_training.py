# Chunk: docs/chunks/training_comparison - Training comparison and analysis
"""Compare curiosity vs. fixed curriculum training runs.

Orchestrates two training experiments (curiosity-driven and MAA fixed
ascending curriculum), generates side-by-side visualizations, and produces
a quantitative comparison report.  Builds entirely on existing infrastructure:

- :mod:`repro_maa.train` for the GRPO training loop
- :mod:`repro_maa.simulation` for :class:`StepRecord`, :func:`read_log`
- :mod:`repro_maa.visualize` for all four plot types

Usage::

    python -m repro_maa.compare_training --num-rounds 100
    python -m repro_maa.compare_training --skip-training --output-dir prev_run
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from repro_maa.simulation import StepRecord, read_log
from repro_maa.train import TrainConfig, run_training
from repro_maa.visualize import generate_all as viz_generate_all

logger = logging.getLogger(__name__)

ABILITIES = ("deduction", "induction", "abduction")
LEVELS = range(1, 6)
DEFAULT_THRESHOLDS = [0.0, 10.0, 50.0, 100.0]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ComparisonConfig:
    """Parameters for a comparison run."""

    # Training parameters (forwarded to TrainConfig)
    num_rounds: int = 100
    batch_size: int = 4
    seed: int = 42
    model_name: str = "Qwen/Qwen3.5-9B"
    num_generations: int = 4
    per_device_train_batch_size: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 512
    learning_rate: float = 1e-6
    gradient_checkpointing: bool = True
    bf16: bool = True
    epsilon: float = 0.1
    window_size: int = 20
    checkpoint_interval: int = 25

    # Comparison-specific parameters
    output_dir: str = "comparison_output"
    skip_training: bool = False
    curiosity_log: str | None = None
    baseline_log: str | None = None

    # Plateau detection parameters
    plateau_window: int = 10
    plateau_min_improvement: float = 0.5

    # Reward thresholds for steps-to-threshold analysis
    reward_thresholds: list[float] = field(
        default_factory=lambda: list(DEFAULT_THRESHOLDS),
    )


# ============================================================================
# Report data structure
# ============================================================================

@dataclass
class ReportData:
    """Computed analysis results for the comparison report."""

    curiosity_records: list[StepRecord]
    baseline_records: list[StepRecord]
    reward_thresholds: dict[str, dict[str, int | None]]
    plateau_rounds: dict[str, list[int]]
    cell_allocation: dict[str, dict[str, int]]
    per_ability_cumulative: dict[str, dict[str, list[float]]]
    selection_variance: dict[str, float]


# ============================================================================
# Training orchestrator
# ============================================================================

def run_comparison(
    config: ComparisonConfig,
) -> tuple[list[StepRecord], list[StepRecord]]:
    """Run or load both training experiments.

    Parameters
    ----------
    config : ComparisonConfig
        Comparison configuration.

    Returns
    -------
    tuple[list[StepRecord], list[StepRecord]]
        (curiosity_records, baseline_records).
    """
    output_path = Path(config.output_dir)

    if config.skip_training:
        curiosity_path = Path(
            config.curiosity_log
            or str(output_path / "curiosity" / "stream_log.jsonl")
        )
        baseline_path = Path(
            config.baseline_log
            or str(output_path / "fixed" / "stream_log.jsonl")
        )
        logger.info("Reading existing logs: %s, %s", curiosity_path, baseline_path)
        return read_log(curiosity_path), read_log(baseline_path)

    # Build common TrainConfig kwargs
    shared = dict(
        model_name=config.model_name,
        num_rounds=config.num_rounds,
        batch_size=config.batch_size,
        num_generations=config.num_generations,
        per_device_train_batch_size=config.per_device_train_batch_size,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        epsilon=config.epsilon,
        window_size=config.window_size,
        checkpoint_interval=config.checkpoint_interval,
        seed=config.seed,
    )

    logger.info("Starting curiosity training run")
    curiosity_config = TrainConfig(
        curriculum="curiosity",
        output_dir=str(output_path / "curiosity"),
        **shared,
    )
    curiosity_records = run_training(curiosity_config)

    logger.info("Starting fixed curriculum training run")
    fixed_config = TrainConfig(
        curriculum="fixed",
        output_dir=str(output_path / "fixed"),
        **shared,
    )
    baseline_records = run_training(fixed_config)

    return curiosity_records, baseline_records


# ============================================================================
# Analysis functions
# ============================================================================

def compute_steps_to_threshold(
    records: list[StepRecord],
    thresholds: list[float] | None = None,
) -> dict[str, dict[str, int | None]]:
    """Find the first round where per-ability cumulative reward exceeds each threshold.

    Parameters
    ----------
    records : list[StepRecord]
        Training log.
    thresholds : list[float] | None
        Cumulative reward thresholds.  Defaults to ``[0.0, 10.0, 50.0, 100.0]``.

    Returns
    -------
    dict[str, dict[str, int | None]]
        ``{ability: {threshold_str: step_or_None}}``.
    """
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)

    per_ability_cum: dict[str, float] = defaultdict(float)
    result: dict[str, dict[str, int | None]] = {
        a: {str(t): None for t in thresholds} for a in ABILITIES
    }

    for record in records:
        per_ability_cum[record.ability] += sum(record.batch_rewards)
        cum = per_ability_cum[record.ability]
        for t in thresholds:
            key = str(t)
            if result[record.ability][key] is None and cum >= t:
                result[record.ability][key] = record.step

    return result


def detect_plateaus(
    records: list[StepRecord],
    window: int = 10,
    min_improvement: float = 0.5,
) -> dict[str, list[int]]:
    """Identify rounds where per-ability learning plateaued.

    A plateau starts at round *r* when the rolling mean reward over the
    previous *window* rounds selecting that ability has not improved by
    more than *min_improvement* compared to the *window* rounds before
    that.

    Parameters
    ----------
    records : list[StepRecord]
        Training log.
    window : int
        Rolling window size.
    min_improvement : float
        Minimum improvement to avoid being classified as a plateau.

    Returns
    -------
    dict[str, list[int]]
        ``{ability: [plateau_start_round, ...]}``.
    """
    per_ability_rewards: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        per_ability_rewards[record.ability].append(
            (record.step, record.batch_mean_reward)
        )

    plateaus: dict[str, list[int]] = {a: [] for a in ABILITIES}
    for ability, entries in per_ability_rewards.items():
        if len(entries) < 2 * window:
            continue
        rewards = [r for _, r in entries]
        steps = [s for s, _ in entries]
        for i in range(2 * window, len(rewards) + 1):
            prev_window = rewards[i - 2 * window : i - window]
            curr_window = rewards[i - window : i]
            prev_mean = np.mean(prev_window)
            curr_mean = np.mean(curr_window)
            improvement = curr_mean - prev_mean
            if improvement < min_improvement:
                plateaus[ability].append(steps[i - 1])

    return plateaus


def compute_per_ability_cumulative(
    records: list[StepRecord],
) -> dict[str, list[float]]:
    """Compute per-ability cumulative reward curves.

    Only rounds selecting a given ability contribute to its curve.

    Parameters
    ----------
    records : list[StepRecord]
        Training log.

    Returns
    -------
    dict[str, list[float]]
        ``{ability: [cumulative_sum_at_each_selection, ...]}``.
    """
    result: dict[str, list[float]] = {a: [] for a in ABILITIES}
    running: dict[str, float] = {a: 0.0 for a in ABILITIES}
    for record in records:
        running[record.ability] += sum(record.batch_rewards)
        result[record.ability].append(running[record.ability])
    return result


def compute_selection_variance(
    records: list[StepRecord],
) -> dict[str, float]:
    """ANOVA-style variance decomposition of MDL scores.

    Decomposes MDL score variance into ability-explained,
    difficulty-explained, and residual fractions.

    Parameters
    ----------
    records : list[StepRecord]
        Training log (typically the curiosity run).

    Returns
    -------
    dict[str, float]
        Keys: ``ability_fraction``, ``difficulty_fraction``,
        ``residual_fraction``.
    """
    if not records:
        return {
            "ability_fraction": 0.0,
            "difficulty_fraction": 0.0,
            "residual_fraction": 0.0,
        }

    scores = np.array([r.mdl_score for r in records])
    grand_mean = scores.mean()
    ss_total = np.sum((scores - grand_mean) ** 2)

    if ss_total < 1e-12:
        return {
            "ability_fraction": 0.0,
            "difficulty_fraction": 0.0,
            "residual_fraction": 0.0,
        }

    # SS for ability factor
    ability_groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        ability_groups[r.ability].append(r.mdl_score)
    ss_ability = sum(
        len(vals) * (np.mean(vals) - grand_mean) ** 2
        for vals in ability_groups.values()
        if len(vals) >= 2
    )

    # SS for difficulty factor
    level_groups: dict[int, list[float]] = defaultdict(list)
    for r in records:
        level_groups[r.level].append(r.mdl_score)
    ss_difficulty = sum(
        len(vals) * (np.mean(vals) - grand_mean) ** 2
        for vals in level_groups.values()
        if len(vals) >= 2
    )

    ability_frac = float(ss_ability / ss_total)
    difficulty_frac = float(ss_difficulty / ss_total)
    residual_frac = max(0.0, 1.0 - ability_frac - difficulty_frac)

    return {
        "ability_fraction": ability_frac,
        "difficulty_fraction": difficulty_frac,
        "residual_fraction": residual_frac,
    }


def compute_cell_allocation(
    records: list[StepRecord],
) -> dict[str, int]:
    """Count cell selection frequencies.

    Parameters
    ----------
    records : list[StepRecord]
        Training log.

    Returns
    -------
    dict[str, int]
        ``{"ability_L{level}": count, ...}``.
    """
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        key = f"{record.ability}_L{record.level}"
        counts[key] += 1
    return dict(counts)


# ============================================================================
# Report generator
# ============================================================================

def generate_report(
    curiosity_records: list[StepRecord],
    baseline_records: list[StepRecord],
    output_path: Path,
    *,
    thresholds: list[float] | None = None,
    plateau_window: int = 10,
    plateau_min_improvement: float = 0.5,
) -> str:
    """Generate a Markdown comparison report.

    Parameters
    ----------
    curiosity_records : list[StepRecord]
        Curiosity run training log.
    baseline_records : list[StepRecord]
        Fixed curriculum run training log.
    output_path : Path
        Where to write the report.
    thresholds : list[float] | None
        Reward thresholds for steps-to-threshold analysis.
    plateau_window : int
        Window size for plateau detection.
    plateau_min_improvement : float
        Minimum improvement threshold for plateau detection.

    Returns
    -------
    str
        The report contents.
    """
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)

    # Compute all metrics
    curiosity_thresholds = compute_steps_to_threshold(curiosity_records, thresholds)
    baseline_thresholds = compute_steps_to_threshold(baseline_records, thresholds)

    curiosity_plateaus = detect_plateaus(
        curiosity_records, plateau_window, plateau_min_improvement,
    )
    baseline_plateaus = detect_plateaus(
        baseline_records, plateau_window, plateau_min_improvement,
    )

    curiosity_per_ability = compute_per_ability_cumulative(curiosity_records)
    baseline_per_ability = compute_per_ability_cumulative(baseline_records)

    curiosity_allocation = compute_cell_allocation(curiosity_records)
    baseline_allocation = compute_cell_allocation(baseline_records)

    selection_var = compute_selection_variance(curiosity_records)

    # Final cumulative rewards
    curiosity_final = (
        curiosity_records[-1].cumulative_reward if curiosity_records else 0.0
    )
    baseline_final = (
        baseline_records[-1].cumulative_reward if baseline_records else 0.0
    )
    advantage = curiosity_final - baseline_final

    # Determine run parameters from records
    num_rounds_curiosity = len(curiosity_records)
    num_rounds_baseline = len(baseline_records)

    # --- Build report ---
    lines: list[str] = []

    lines.append("# Training Comparison Report")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Curiosity run: {num_rounds_curiosity} rounds")
    lines.append(f"- Fixed curriculum run: {num_rounds_baseline} rounds")
    lines.append("")

    # Cumulative Reward Comparison
    lines.append("## Cumulative Reward Comparison")
    lines.append("")
    lines.append(f"| Metric | Curiosity | Fixed | Advantage |")
    lines.append(f"|--------|-----------|-------|-----------|")
    lines.append(
        f"| Final cumulative reward | {curiosity_final:.1f} | "
        f"{baseline_final:.1f} | {advantage:+.1f} |"
    )
    lines.append("")

    # Steps to Reward Thresholds
    lines.append("## Steps to Reward Thresholds")
    lines.append("")
    for ability in ABILITIES:
        lines.append(f"### {ability.title()}")
        lines.append("")
        lines.append("| Threshold | Curiosity (step) | Fixed (step) |")
        lines.append("|-----------|-----------------|--------------|")
        for t in thresholds:
            key = str(t)
            c_step = curiosity_thresholds[ability].get(key)
            b_step = baseline_thresholds[ability].get(key)
            c_str = str(c_step) if c_step is not None else "—"
            b_str = str(b_step) if b_step is not None else "—"
            lines.append(f"| {t} | {c_str} | {b_str} |")
        lines.append("")

    # Plateau Detection
    lines.append("## Plateau Detection")
    lines.append("")
    lines.append(
        f"Plateau window: {plateau_window} rounds, "
        f"min improvement: {plateau_min_improvement}"
    )
    lines.append("")
    for ability in ABILITIES:
        c_plateaus = curiosity_plateaus.get(ability, [])
        b_plateaus = baseline_plateaus.get(ability, [])
        lines.append(f"### {ability.title()}")
        lines.append("")
        lines.append(
            f"- Curiosity plateau rounds: "
            f"{', '.join(str(r) for r in c_plateaus) if c_plateaus else 'none detected'}"
        )
        lines.append(
            f"- Fixed plateau rounds: "
            f"{', '.join(str(r) for r in b_plateaus) if b_plateaus else 'none detected'}"
        )
        # Highlight reallocation behavior
        if c_plateaus and b_plateaus:
            if len(c_plateaus) < len(b_plateaus):
                lines.append(
                    f"- Curiosity showed fewer plateau rounds "
                    f"({len(c_plateaus)} vs {len(b_plateaus)}), "
                    f"suggesting faster reallocation away from stalled cells."
                )
            elif len(c_plateaus) > len(b_plateaus):
                lines.append(
                    f"- Fixed curriculum showed fewer plateau rounds "
                    f"({len(b_plateaus)} vs {len(c_plateaus)})."
                )
        lines.append("")

    # Cell Allocation
    lines.append("## Cell Allocation")
    lines.append("")
    all_cells = sorted(
        set(list(curiosity_allocation.keys()) + list(baseline_allocation.keys()))
    )
    lines.append("| Cell | Curiosity | Fixed |")
    lines.append("|------|-----------|-------|")
    for cell in all_cells:
        c_count = curiosity_allocation.get(cell, 0)
        b_count = baseline_allocation.get(cell, 0)
        lines.append(f"| {cell} | {c_count} | {b_count} |")
    lines.append("")

    # Cells favored by one curriculum but not the other
    curiosity_only = [
        c for c in all_cells
        if curiosity_allocation.get(c, 0) > 0
        and baseline_allocation.get(c, 0) == 0
    ]
    baseline_only = [
        c for c in all_cells
        if baseline_allocation.get(c, 0) > 0
        and curiosity_allocation.get(c, 0) == 0
    ]
    if curiosity_only:
        lines.append(
            f"Cells selected only by curiosity: {', '.join(curiosity_only)}"
        )
    if baseline_only:
        lines.append(
            f"Cells selected only by fixed: {', '.join(baseline_only)}"
        )
    if curiosity_only or baseline_only:
        lines.append("")

    # Per-Ability Reward Curves
    lines.append("## Per-Ability Reward Curves")
    lines.append("")
    lines.append("| Ability | Curiosity (final) | Fixed (final) | Selections (C) | Selections (F) |")
    lines.append("|---------|-------------------|---------------|----------------|----------------|")
    for ability in ABILITIES:
        c_final = curiosity_per_ability[ability][-1] if curiosity_per_ability[ability] else 0.0
        b_final = baseline_per_ability[ability][-1] if baseline_per_ability[ability] else 0.0
        c_sel = len(curiosity_per_ability[ability])
        b_sel = len(baseline_per_ability[ability])
        lines.append(
            f"| {ability.title()} | {c_final:.1f} | {b_final:.1f} | {c_sel} | {b_sel} |"
        )
    lines.append("")

    # Selection Granularity Analysis
    lines.append("## Selection Granularity Analysis")
    lines.append("")
    lines.append(
        "ANOVA-style variance decomposition of MDL scores from the curiosity run. "
        "References hypotheses H1–H4 from `docs/investigations/selection_granularity/`."
    )
    lines.append("")
    lines.append(f"- Ability-explained variance: {selection_var['ability_fraction']:.1%}")
    lines.append(
        f"- Difficulty-explained variance: {selection_var['difficulty_fraction']:.1%}"
    )
    lines.append(f"- Residual variance: {selection_var['residual_fraction']:.1%}")
    lines.append("")

    if selection_var["ability_fraction"] > 0.6:
        lines.append(
            "**H1 supported**: Ability axis explains >60% of MDL score variance. "
            "The ability dimension carries the dominant selection signal."
        )
    elif selection_var["difficulty_fraction"] > 0.6:
        lines.append(
            "**H1 not supported**: Difficulty axis explains >60% of variance. "
            "The difficulty dimension carries the dominant selection signal."
        )
    else:
        lines.append(
            "**H1 inconclusive**: Neither axis explains >60% of variance. "
            "Both ability and difficulty contribute meaningful signal."
        )
    lines.append("")

    report = "\n".join(lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info("Report written to %s", output_path)

    return report


# ============================================================================
# Plot generation wrapper
# ============================================================================

def generate_plots(
    curiosity_log_path: Path,
    baseline_log_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate comparison plots via :func:`visualize.generate_all`.

    Parameters
    ----------
    curiosity_log_path : Path
        Path to the curiosity run's JSONL log.
    baseline_log_path : Path
        Path to the fixed curriculum run's JSONL log.
    output_dir : Path
        Output directory for plot PNGs.

    Returns
    -------
    list[Path]
        Paths to the generated plot files.
    """
    return viz_generate_all(
        curiosity_log_path=curiosity_log_path,
        baseline_log_path=baseline_log_path,
        output_dir=output_dir,
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: list[str] | None = None) -> ComparisonConfig:
    """Parse command-line arguments into a :class:`ComparisonConfig`.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    ComparisonConfig
        Populated configuration.
    """
    parser = argparse.ArgumentParser(
        description="Compare curiosity vs. fixed curriculum training runs",
    )

    # Comparison-specific flags
    parser.add_argument(
        "--output-dir", default=ComparisonConfig.output_dir,
    )
    parser.add_argument(
        "--skip-training", action="store_true", default=False,
    )
    parser.add_argument("--curiosity-log", default=None)
    parser.add_argument("--baseline-log", default=None)
    parser.add_argument(
        "--plateau-window", type=int, default=ComparisonConfig.plateau_window,
    )
    parser.add_argument(
        "--plateau-min-improvement", type=float,
        default=ComparisonConfig.plateau_min_improvement,
    )

    # TrainConfig forwarding flags
    parser.add_argument(
        "--num-rounds", type=int, default=ComparisonConfig.num_rounds,
    )
    parser.add_argument(
        "--batch-size", type=int, default=ComparisonConfig.batch_size,
    )
    parser.add_argument("--seed", type=int, default=ComparisonConfig.seed)
    parser.add_argument(
        "--model-name", default=ComparisonConfig.model_name,
    )
    parser.add_argument(
        "--num-generations", type=int,
        default=ComparisonConfig.num_generations,
    )
    parser.add_argument(
        "--per-device-train-batch-size", type=int,
        default=ComparisonConfig.per_device_train_batch_size,
    )
    parser.add_argument(
        "--max-prompt-length", type=int,
        default=ComparisonConfig.max_prompt_length,
    )
    parser.add_argument(
        "--max-completion-length", type=int,
        default=ComparisonConfig.max_completion_length,
    )
    parser.add_argument(
        "--learning-rate", type=float,
        default=ComparisonConfig.learning_rate,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=ComparisonConfig.gradient_checkpointing,
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=ComparisonConfig.bf16,
    )
    parser.add_argument(
        "--epsilon", type=float, default=ComparisonConfig.epsilon,
    )
    parser.add_argument(
        "--window-size", type=int, default=ComparisonConfig.window_size,
    )
    parser.add_argument(
        "--checkpoint-interval", type=int,
        default=ComparisonConfig.checkpoint_interval,
    )

    args = parser.parse_args(argv)
    return ComparisonConfig(
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        seed=args.seed,
        model_name=args.model_name,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        epsilon=args.epsilon,
        window_size=args.window_size,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.output_dir,
        skip_training=args.skip_training,
        curiosity_log=args.curiosity_log,
        baseline_log=args.baseline_log,
        plateau_window=args.plateau_window,
        plateau_min_improvement=args.plateau_min_improvement,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the comparison script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    config = parse_args(argv)
    logger.info("Starting comparison with config: %s", config)

    # 1. Run or load both training experiments
    curiosity_records, baseline_records = run_comparison(config)
    logger.info(
        "Loaded %d curiosity records, %d baseline records",
        len(curiosity_records),
        len(baseline_records),
    )

    # 2. Generate comparison plots
    output_path = Path(config.output_dir)
    curiosity_log_path = Path(
        config.curiosity_log
        or str(output_path / "curiosity" / "stream_log.jsonl")
    )
    baseline_log_path = Path(
        config.baseline_log
        or str(output_path / "fixed" / "stream_log.jsonl")
    )
    plot_paths = generate_plots(
        curiosity_log_path, baseline_log_path, output_path / "plots",
    )
    logger.info("Generated %d plots", len(plot_paths))

    # 3. Generate comparison report
    report_path = output_path / "comparison_report.md"
    generate_report(
        curiosity_records,
        baseline_records,
        report_path,
        thresholds=config.reward_thresholds,
        plateau_window=config.plateau_window,
        plateau_min_improvement=config.plateau_min_improvement,
    )

    # 4. Print artifact paths
    print("\n=== Comparison Complete ===")
    print(f"Report: {report_path}")
    for p in plot_paths:
        print(f"Plot:   {p}")


if __name__ == "__main__":
    main()
