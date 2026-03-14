# Chunk: docs/chunks/stream_visualization - Visualization tests
"""
Tests for the stream visualization module.

Structural tests verifying that output files are created, figures have
expected dimensions/axes, and data arrays have expected shapes. No
aesthetic assertions (per TESTING_PHILOSOPHY.md).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from repro_maa.simulation import StepRecord, write_log
from repro_maa.visualize import (
    generate_all,
    plot_cell_selection_timeline,
    plot_cumulative_reward,
    plot_mdl_evolution,
    plot_selection_heatmap,
)


# ============================================================================
# Fixtures / helpers
# ============================================================================

ABILITIES = ["deduction", "induction", "abduction"]
LEVELS = [1, 2, 3, 4, 5]


def _make_records(
    n: int,
    strategy: str = "curiosity",
    seed: int = 42,
) -> list[StepRecord]:
    """Create synthetic StepRecords with varied ability/level selections.

    Cycles through the 3×5 cell grid so that plots have non-trivial data.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    records: list[StepRecord] = []
    cumulative = 0.0
    cells = [(a, l) for a in ABILITIES for l in LEVELS]

    for i in range(n):
        ability, level = cells[i % len(cells)]
        batch_rewards = [rng.choice([3.0, -3.0]) for _ in range(4)]
        batch_mean = sum(batch_rewards) / len(batch_rewards)
        cumulative += sum(batch_rewards)
        mdl_score = rng.random() * 2.0  # random MDL score in [0, 2)

        selection_reason = strategy if strategy == "fixed_schedule" else (
            "curiosity" if rng.random() > 0.1 else "exploration"
        )

        records.append(StepRecord(
            step=i,
            ability=ability,
            level=level,
            mdl_score=mdl_score,
            selection_reason=selection_reason,
            batch_rewards=batch_rewards,
            batch_mean_reward=batch_mean,
            cumulative_reward=cumulative,
            reward_history_summary={
                f"{ability}_L{level}": {"mean": batch_mean, "count": i + 1}
            },
        ))

    return records


@pytest.fixture
def curiosity_records() -> list[StepRecord]:
    """30 steps of curiosity-driven records."""
    return _make_records(30, strategy="curiosity", seed=42)


@pytest.fixture
def baseline_records() -> list[StepRecord]:
    """30 steps of fixed-schedule baseline records."""
    return _make_records(30, strategy="fixed_schedule", seed=99)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ============================================================================
# Cell selection timeline tests
# ============================================================================


class TestCellSelectionTimeline:
    def test_single_stream_axes(self, curiosity_records):
        """Single stream produces a figure with 1 axes."""
        fig = plot_cell_selection_timeline(curiosity_records)
        assert isinstance(fig, matplotlib.figure.Figure)
        axes = fig.get_axes()
        assert len(axes) == 1

        ax = axes[0]
        assert "selection" in ax.get_title().lower()
        assert ax.get_xlabel().lower() == "step"

    def test_two_streams(self, curiosity_records, baseline_records):
        """Two streams produce a figure with 2 subplots."""
        fig = plot_cell_selection_timeline(curiosity_records, baseline_records)
        assert isinstance(fig, matplotlib.figure.Figure)
        axes = fig.get_axes()
        assert len(axes) == 2


# ============================================================================
# MDL score evolution tests
# ============================================================================


class TestMdlEvolution:
    def test_axes(self, curiosity_records):
        """Returns a Figure with MDL-related y-axis label and plotted lines."""
        fig = plot_mdl_evolution(curiosity_records)
        assert isinstance(fig, matplotlib.figure.Figure)
        axes = fig.get_axes()
        assert len(axes) >= 1

        ax = axes[0]
        assert "mdl" in ax.get_ylabel().lower()
        assert ax.get_xlabel().lower() == "step"

        # Should have at least one line (one per cell that was selected)
        lines = ax.get_lines()
        assert len(lines) >= 1


# ============================================================================
# Cumulative reward comparison tests
# ============================================================================


class TestCumulativeReward:
    def test_two_lines(self, curiosity_records, baseline_records):
        """Two-stream comparison produces 2 lines with reward label."""
        fig = plot_cumulative_reward(curiosity_records, baseline_records)
        assert isinstance(fig, matplotlib.figure.Figure)
        axes = fig.get_axes()
        assert len(axes) == 1

        ax = axes[0]
        assert "reward" in ax.get_ylabel().lower()

        lines = ax.get_lines()
        assert len(lines) == 2

        # Legend should have 2 entries
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2

    def test_single_line(self, curiosity_records):
        """Single stream produces 1 line."""
        fig = plot_cumulative_reward(curiosity_records)
        ax = fig.get_axes()[0]
        lines = ax.get_lines()
        assert len(lines) == 1


# ============================================================================
# Selection heatmap tests
# ============================================================================


class TestSelectionHeatmap:
    def test_two_panels_shape(self, curiosity_records, baseline_records):
        """Two-stream heatmap produces 2 subplots with 3×5 grids."""
        fig = plot_selection_heatmap(curiosity_records, baseline_records)
        assert isinstance(fig, matplotlib.figure.Figure)
        axes = fig.get_axes()
        assert len(axes) == 2

        for ax in axes:
            # Check ytick labels match abilities
            ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
            assert ytick_labels == ABILITIES

            # Check xtick labels match levels
            xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
            assert xtick_labels == [str(l) for l in LEVELS]

    def test_single_panel(self, curiosity_records):
        """Single stream produces 1 subplot."""
        fig = plot_selection_heatmap(curiosity_records)
        axes = fig.get_axes()
        assert len(axes) == 1


# ============================================================================
# generate_all tests
# ============================================================================


class TestGenerateAll:
    def test_creates_png_files(self, tmp_path, curiosity_records, baseline_records):
        """generate_all creates 4 PNG files in the output directory."""
        curiosity_log = tmp_path / "curiosity.jsonl"
        baseline_log = tmp_path / "baseline.jsonl"
        output_dir = tmp_path / "output"

        write_log(curiosity_records, curiosity_log)
        write_log(baseline_records, baseline_log)

        paths = generate_all(curiosity_log, baseline_log, output_dir)

        assert len(paths) == 4
        expected_names = {
            "cell_selection_timeline.png",
            "mdl_evolution.png",
            "cumulative_reward.png",
            "selection_heatmap.png",
        }
        actual_names = {p.name for p in paths}
        assert actual_names == expected_names

        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_single_log(self, tmp_path, curiosity_records):
        """generate_all with only curiosity log still creates all 4 files."""
        curiosity_log = tmp_path / "curiosity.jsonl"
        output_dir = tmp_path / "output"

        write_log(curiosity_records, curiosity_log)

        paths = generate_all(curiosity_log, output_dir=output_dir)

        assert len(paths) == 4
        for p in paths:
            assert p.exists()


# ============================================================================
# CLI test
# ============================================================================


class TestCLI:
    def test_cli_runs_successfully(self, tmp_path, curiosity_records):
        """python -m repro_maa.visualize runs and produces PNGs."""
        curiosity_log = tmp_path / "curiosity.jsonl"
        output_dir = tmp_path / "cli_output"

        write_log(curiosity_records, curiosity_log)

        result = subprocess.run(
            [
                sys.executable, "-m", "repro_maa.visualize",
                str(curiosity_log),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_dir.exists()

        expected_files = [
            "cell_selection_timeline.png",
            "mdl_evolution.png",
            "cumulative_reward.png",
            "selection_heatmap.png",
        ]
        for fname in expected_files:
            assert (output_dir / fname).exists(), f"Missing: {fname}"
