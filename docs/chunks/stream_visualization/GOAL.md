---
status: ACTIVE
ticket: null
parent_chunk: null
code_paths:
- src/repro_maa/visualize.py
- src/repro_maa/__main__.py
- tests/test_visualize.py
- pyproject.toml
code_references:
- ref: src/repro_maa/visualize.py
  implements: "Stream visualization module — data extraction, four plot functions, generate_all orchestrator, and CLI entry point"
- ref: src/repro_maa/visualize.py#plot_cell_selection_timeline
  implements: "Cell selection timeline plot showing which cell was selected at each step, color-coded by ability"
- ref: src/repro_maa/visualize.py#plot_mdl_evolution
  implements: "MDL score evolution line plot per cell over time"
- ref: src/repro_maa/visualize.py#plot_cumulative_reward
  implements: "Cumulative reward comparison between curiosity-driven and fixed curriculum"
- ref: src/repro_maa/visualize.py#plot_selection_heatmap
  implements: "3×5 selection count heatmap (abilities × levels)"
- ref: src/repro_maa/visualize.py#generate_all
  implements: "Orchestrator that generates all four plots and saves as PNGs"
- ref: src/repro_maa/visualize.py#cli_main
  implements: "CLI entry point for python -m repro_maa.visualize"
- ref: src/repro_maa/__main__.py
  implements: "Thin CLI wrapper delegating to visualize.cli_main"
- ref: tests/test_visualize.py
  implements: "Structural tests for all four plots, generate_all, and CLI"
narrative: curiosity_stream_mvp
investigation: null
subsystems: []
friction_entries: []
bug_type: null
depends_on:
- simulation_harness
created_after:
- scaffold_project
---

# Chunk Goal

## Minor Goal

Build stream analysis and visualization tooling that reads the stream log from the simulation harness and produces static plots for post-hoc analysis of the curiosity policy's behavior. This is the final piece of the MVP — it makes the curiosity signal's behavior legible to a researcher.

The visualizations must answer four questions: (1) which cells did the stream select over time? (2) how did each cell's MDL score evolve? (3) did the curiosity stream accumulate reward faster than the fixed curriculum? (4) how was training time distributed across the 3×5 cell grid?

## Success Criteria

1. **Cell selection timeline**: A plot showing which (ability, level) cell was selected at each step, for both curiosity-driven and fixed curriculum streams. Color-coded by ability.
2. **MDL score evolution**: Line plot showing MDL curiosity score per cell over time. Should visually demonstrate that mastered cells drop and frontier cells rise.
3. **Cumulative reward comparison**: A two-line plot comparing cumulative reward over steps for curiosity-driven vs. fixed curriculum.
4. **Selection heatmap**: A 3×5 grid (abilities × levels) showing how many times each cell was selected, as a heatmap. One for curiosity, one for fixed baseline, side by side.
5. **All plots saved as PNG files** to a configurable output directory.
6. **`python -m curiosity_stream.visualize <log_path>`** runs the full visualization from the command line.
7. **Tests** verify: output files are created, plots have expected dimensions/axes (structural tests per TESTING_PHILOSOPHY.md — we test the instrument, not the aesthetics).