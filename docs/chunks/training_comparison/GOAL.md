---
status: ACTIVE
ticket: null
parent_chunk: null
code_paths:
- src/repro_maa/compare_training.py
- tests/test_compare_training.py
- src/repro_maa/__init__.py
code_references:
  - ref: src/repro_maa/compare_training.py#ComparisonConfig
    implements: "Configuration dataclass wrapping shared training + comparison-specific parameters"
  - ref: src/repro_maa/compare_training.py#ReportData
    implements: "Data structure holding computed analysis results for the comparison report"
  - ref: src/repro_maa/compare_training.py#run_comparison
    implements: "Training orchestrator that runs or loads both curiosity and fixed curriculum experiments"
  - ref: src/repro_maa/compare_training.py#compute_steps_to_threshold
    implements: "Per-ability steps-to-reward-threshold analysis"
  - ref: src/repro_maa/compare_training.py#detect_plateaus
    implements: "Rolling-window plateau detection for per-ability reward curves"
  - ref: src/repro_maa/compare_training.py#compute_per_ability_cumulative
    implements: "Per-ability cumulative reward curve computation"
  - ref: src/repro_maa/compare_training.py#compute_selection_variance
    implements: "ANOVA-style variance decomposition of MDL scores for selection_granularity investigation"
  - ref: src/repro_maa/compare_training.py#compute_cell_allocation
    implements: "Cell selection frequency counting"
  - ref: src/repro_maa/compare_training.py#generate_report
    implements: "Markdown comparison report generator with all quantitative sections"
  - ref: src/repro_maa/compare_training.py#generate_plots
    implements: "Thin wrapper delegating to visualize.generate_all for comparison plots"
  - ref: src/repro_maa/compare_training.py#parse_args
    implements: "CLI argument parsing into ComparisonConfig"
  - ref: src/repro_maa/compare_training.py#main
    implements: "CLI entry point orchestrating comparison pipeline"
  - ref: tests/test_compare_training.py
    implements: "Test suite verifying measurement apparatus (analysis functions, report, CLI, log reading)"
  - ref: src/repro_maa/__init__.py
    implements: "Public API exports for ComparisonConfig, run_comparison, generate_report"
narrative: curiosity_training_run
investigation: null
subsystems: []
friction_entries: []
bug_type: null
depends_on:
- curiosity_grpo_loop
created_after:
- mdl_curiosity_scorer
- simulation_harness
- stream_generator
- stream_visualization
- taskcell_abstraction
---

# Chunk Goal

## Minor Goal

Create a comparison launch script that runs two training experiments — one with curiosity curriculum, one with the MAA fixed ascending curriculum — and produces a unified analysis. Use the existing stream visualization module to generate plots (cell selection timeline, MDL evolution, cumulative reward comparison, selection heatmap). Add a summary report comparing training efficiency: steps to reach reward thresholds, cell allocation differences, and whether the curiosity policy reallocates away from plateaued cells faster than the fixed schedule.

## Success Criteria

1. **Launch script**: A single command (e.g., `python -m repro_maa.compare_training`) that runs both curiosity and fixed curriculum training sequentially (or resumes from existing checkpoints/logs).
2. **Comparison plots**: All four visualization types (cell selection timeline, MDL evolution, cumulative reward, selection heatmap) generated for both runs side by side using the existing visualization module.
3. **Summary report**: A text or markdown report with quantitative comparisons: total cumulative reward, steps to first positive reward per ability, cell allocation distribution, and plateau detection (rounds where reward stopped improving).
4. **Reproducibility**: The comparison is reproducible given the same seed — both runs use deterministic curriculum selection and seeded problem generation.
5. **Results inform selection_granularity investigation**: The output provides data to evaluate hypotheses H1-H4 from `docs/investigations/selection_granularity/` — specifically, whether the curiosity policy's cell selections show ability-dominant or difficulty-dominant patterns during real training.