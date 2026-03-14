# Chunk: docs/chunks/stream_visualization - Stream visualization module
"""Post-hoc visualization of curiosity-stream simulation logs.

Reads JSONL simulation logs (via ``read_log`` / ``StepRecord`` from
``simulation.py``) and produces four matplotlib figures as PNG files:

1. **Cell selection timeline** — which cell was selected at each step
2. **MDL score evolution** — MDL curiosity score per cell over time
3. **Cumulative reward comparison** — curiosity vs fixed curriculum
4. **Selection heatmap** — 3×5 grid of selection counts

Usage::

    from repro_maa.visualize import generate_all
    generate_all(Path("curiosity.jsonl"), Path("baseline.jsonl"), Path("plots"))

CLI::

    python -m repro_maa.visualize curiosity.jsonl --baseline baseline.jsonl --output-dir plots
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from repro_maa.simulation import StepRecord, read_log  # noqa: E402

# ============================================================================
# Constants
# ============================================================================

ABILITIES = ["deduction", "induction", "abduction"]
LEVELS = [1, 2, 3, 4, 5]

ABILITY_COLORS = {
    "deduction": "#1f77b4",
    "induction": "#ff7f0e",
    "abduction": "#2ca02c",
}

LEVEL_LINESTYLES = {
    1: "solid",
    2: "dashed",
    3: "dashdot",
    4: "dotted",
    5: (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
}

# ============================================================================
# Data extraction helpers
# ============================================================================


def _cell_key(ability: str, level: int) -> str:
    """Return a display label for a cell, e.g. ``'deduction L1'``."""
    return f"{ability} L{level}"


def _all_cell_labels() -> list[str]:
    """Return all 15 cell labels sorted by ability then level."""
    return [_cell_key(a, l) for a in ABILITIES for l in LEVELS]


def _extract_selections(
    records: list[StepRecord],
) -> tuple[list[int], list[str]]:
    """Extract step numbers and selected cell labels."""
    steps = [r.step for r in records]
    labels = [_cell_key(r.ability, r.level) for r in records]
    return steps, labels


def _extract_mdl_timeseries(
    records: list[StepRecord],
) -> dict[str, list[tuple[int, float]]]:
    """Extract per-cell MDL score timeseries.

    Since only the *selected* cell's MDL score is recorded per step,
    this returns ``{cell_label: [(step, mdl_score), ...]}`` only for
    steps where each cell was selected.
    """
    result: dict[str, list[tuple[int, float]]] = {}
    for r in records:
        key = _cell_key(r.ability, r.level)
        result.setdefault(key, []).append((r.step, r.mdl_score))
    return result


def _extract_cumulative_rewards(
    records: list[StepRecord],
) -> tuple[list[int], list[float]]:
    """Extract step numbers and cumulative reward values."""
    steps = [r.step for r in records]
    rewards = [r.cumulative_reward for r in records]
    return steps, rewards


def _extract_selection_counts(records: list[StepRecord]) -> np.ndarray:
    """Build a 3×5 selection count matrix (abilities × levels).

    Rows correspond to ABILITIES, columns to LEVELS (1–5).
    """
    counts = np.zeros((len(ABILITIES), len(LEVELS)), dtype=int)
    ability_idx = {a: i for i, a in enumerate(ABILITIES)}
    for r in records:
        row = ability_idx.get(r.ability)
        col = r.level - 1
        if row is not None and 0 <= col < len(LEVELS):
            counts[row, col] += 1
    return counts


# ============================================================================
# Plot functions
# ============================================================================


def plot_cell_selection_timeline(
    curiosity_records: list[StepRecord],
    baseline_records: list[StepRecord] | None = None,
) -> Figure:
    """Cell selection timeline — which cell was selected at each step.

    Creates 1 or 2 vertically-stacked subplots. Points are color-coded
    by ability.
    """
    all_labels = _all_cell_labels()
    label_to_y = {label: i for i, label in enumerate(all_labels)}

    n_plots = 2 if baseline_records else 1
    fig, axes = plt.subplots(
        n_plots, 1, figsize=(12, 4 * n_plots), squeeze=False
    )

    datasets = [("Curiosity-Driven", curiosity_records)]
    if baseline_records:
        datasets.append(("Fixed Curriculum", baseline_records))

    for idx, (title, records) in enumerate(datasets):
        ax = axes[idx, 0]
        steps, labels = _extract_selections(records)
        y_vals = [label_to_y[lbl] for lbl in labels]
        colors = [ABILITY_COLORS[r.ability] for r in records]

        ax.scatter(steps, y_vals, c=colors, s=12, alpha=0.7, edgecolors="none")
        ax.set_yticks(range(len(all_labels)))
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.set_xlabel("Step")
        ax.set_title(f"Cell Selection Timeline — {title}")
        ax.set_xlim(left=0)
        ax.set_ylim(-0.5, len(all_labels) - 0.5)

    fig.tight_layout()
    return fig


def plot_mdl_evolution(records: list[StepRecord]) -> Figure:
    """MDL score evolution — line plot per cell over time.

    Each line shows the MDL score at steps where that cell was selected.
    Lines are color-coded by ability with different styles per level.
    """
    timeseries = _extract_mdl_timeseries(records)

    fig, ax = plt.subplots(figsize=(12, 6))

    for cell_label, data in sorted(timeseries.items()):
        # Parse ability and level from label
        parts = cell_label.split(" L")
        ability = parts[0]
        level = int(parts[1])

        steps_arr = [d[0] for d in data]
        scores = [d[1] for d in data]

        ax.plot(
            steps_arr,
            scores,
            label=cell_label,
            color=ABILITY_COLORS.get(ability, "gray"),
            linestyle=LEVEL_LINESTYLES.get(level, "solid"),
            marker=".",
            markersize=3,
            alpha=0.8,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("MDL Score")
    ax.set_title("MDL Score Evolution")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cumulative_reward(
    curiosity_records: list[StepRecord],
    baseline_records: list[StepRecord] | None = None,
) -> Figure:
    """Cumulative reward comparison — one or two lines over steps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    c_steps, c_rewards = _extract_cumulative_rewards(curiosity_records)
    ax.plot(c_steps, c_rewards, label="Curiosity-Driven", linewidth=2)

    if baseline_records:
        b_steps, b_rewards = _extract_cumulative_rewards(baseline_records)
        ax.plot(b_steps, b_rewards, label="Fixed Curriculum", linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_selection_heatmap(
    curiosity_records: list[StepRecord],
    baseline_records: list[StepRecord] | None = None,
) -> Figure:
    """Selection heatmap — 3×5 grid of selection counts.

    Creates 1 or 2 side-by-side subplots with annotated heatmaps.
    """
    n_plots = 2 if baseline_records else 1
    fig, axes = plt.subplots(
        1, n_plots, figsize=(6 * n_plots, 4), squeeze=False
    )

    datasets = [("Curiosity-Driven", curiosity_records)]
    if baseline_records:
        datasets.append(("Fixed Curriculum", baseline_records))

    for idx, (title, records) in enumerate(datasets):
        ax = axes[0, idx]
        counts = _extract_selection_counts(records)

        im = ax.imshow(counts, cmap="YlOrRd", aspect="auto")

        # Annotate cells with counts
        for i in range(len(ABILITIES)):
            for j in range(len(LEVELS)):
                ax.text(
                    j, i, str(counts[i, j]),
                    ha="center", va="center", fontsize=10,
                    color="white" if counts[i, j] > counts.max() / 2 else "black",
                )

        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([str(l) for l in LEVELS])
        ax.set_yticks(range(len(ABILITIES)))
        ax.set_yticklabels(ABILITIES)
        ax.set_xlabel("Level")
        ax.set_ylabel("Ability")
        ax.set_title(f"Selection Heatmap — {title}")

    fig.tight_layout()
    return fig


# ============================================================================
# Orchestrator
# ============================================================================

PLOT_FILENAMES = {
    "cell_selection_timeline": "cell_selection_timeline.png",
    "mdl_evolution": "mdl_evolution.png",
    "cumulative_reward": "cumulative_reward.png",
    "selection_heatmap": "selection_heatmap.png",
}


def generate_all(
    curiosity_log_path: Path,
    baseline_log_path: Path | None = None,
    output_dir: Path = Path("plots"),
) -> list[Path]:
    """Generate all four visualizations and save as PNGs.

    Parameters
    ----------
    curiosity_log_path : Path
        Path to the curiosity-driven simulation JSONL log.
    baseline_log_path : Path | None
        Optional path to the fixed-curriculum baseline JSONL log.
    output_dir : Path
        Directory for output PNG files (created if needed).

    Returns
    -------
    list[Path]
        Paths to the created PNG files.
    """
    curiosity_records = read_log(curiosity_log_path)
    baseline_records = read_log(baseline_log_path) if baseline_log_path else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    # 1. Cell selection timeline
    fig = plot_cell_selection_timeline(curiosity_records, baseline_records)
    path = output_dir / PLOT_FILENAMES["cell_selection_timeline"]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    output_paths.append(path)

    # 2. MDL evolution (curiosity stream only)
    fig = plot_mdl_evolution(curiosity_records)
    path = output_dir / PLOT_FILENAMES["mdl_evolution"]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    output_paths.append(path)

    # 3. Cumulative reward comparison
    fig = plot_cumulative_reward(curiosity_records, baseline_records)
    path = output_dir / PLOT_FILENAMES["cumulative_reward"]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    output_paths.append(path)

    # 4. Selection heatmap
    fig = plot_selection_heatmap(curiosity_records, baseline_records)
    path = output_dir / PLOT_FILENAMES["selection_heatmap"]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    output_paths.append(path)

    plt.close("all")
    return output_paths


# ============================================================================
# CLI entry point
# ============================================================================


def cli_main() -> None:
    """Command-line entry point for visualization."""
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from simulation logs.",
    )
    parser.add_argument(
        "curiosity_log",
        type=Path,
        help="Path to the curiosity-driven simulation JSONL log.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to the fixed-curriculum baseline JSONL log.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for output PNG files (default: plots).",
    )

    args = parser.parse_args()
    paths = generate_all(args.curiosity_log, args.baseline, args.output_dir)
    for p in paths:
        print(f"Created: {p}")


if __name__ == "__main__":
    cli_main()
