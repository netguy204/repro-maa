---
status: DRAFTING
advances_trunk_goal: "Success Criteria #4: Baseline comparison via real training — train with curiosity-driven vs fixed curriculum and compare reward curves"
proposed_chunks:
  - prompt: "Build the glue layer between the existing MAA task generators and TRL's GRPOTrainer. Adapt TaskCell problems into Qwen3.5-9B chat prompt format (system message + user problem statement, expecting <think>...</think><answer>...</answer> responses). Wrap the MAA reward functions (formula.py, backward_reasoning.py, squence.py) into TRL's reward_funcs interface — each function receives a list of completions and returns a list of float rewards. Include tests that verify prompt formatting round-trips and reward scoring matches the existing MAA functions."
    chunk_directory: prompt_reward_bridge
    depends_on: []
  - prompt: "Implement a curiosity-aware GRPO training loop that integrates CuriosityStream with TRL's GRPOTrainer. The outer loop runs in rounds: CuriosityStream selects a batch of problems from the highest-MDL cell, GRPOTrainer generates completions and trains on them, MAA reward scores are fed back to CuriosityStream for MDL updates. Configure for DGX Spark (Qwen/Qwen3.5-9B, 128GB unified memory, single GPU, conservative batch sizes for limited memory bandwidth). Support both curiosity-driven and fixed-curriculum modes via the existing FixedCurriculumBaseline. Log the full stream trace (cell selections, MDL scores, rewards) in the same JSONL format the visualization module already reads."
    chunk_directory: curiosity_grpo_loop
    depends_on: [0]
  - prompt: "Create a comparison launch script that runs two training experiments — one with curiosity curriculum, one with the MAA fixed ascending curriculum — and produces a unified analysis. Use the existing stream visualization module to generate plots (cell selection timeline, MDL evolution, cumulative reward comparison, selection heatmap). Add a summary report comparing training efficiency: steps to reach reward thresholds, cell allocation differences, and whether the curiosity policy reallocates away from plateaued cells faster than the fixed schedule."
    chunk_directory: training_comparison
    depends_on: [1]
created_after: ["stream_visualization"]
---

<!--
STATUS VALUES:
- DRAFTING: The narrative is being refined; chunks not yet created
- ACTIVE: Chunks are being created and implemented from this narrative
- COMPLETED: All chunks have been created and the narrative's ambition is realized

ADVANCES_TRUNK_GOAL:
- Reference the specific section of docs/trunk/GOAL.md this narrative advances
- Example: "Required Properties: Must support multi-repository workflows"

PROPOSED_CHUNKS:
- Starts empty; entries are added as prompts are turned into chunks via /chunk-create
- Each entry records which prompt was refined and where the resulting chunk lives
- prompt: The prompt text from this document that was used to create the chunk
- chunk_directory: The created chunk directory (e.g., "feature_name"), null until created
- depends_on: Optional array of integer indices expressing implementation dependencies.

  SEMANTICS (null vs empty distinction):
  | Value           | Meaning                                 | Oracle behavior |
  |-----------------|----------------------------------------|-----------------|
  | omitted/null    | "I don't know dependencies for this"  | Consult oracle  |
  | []              | "Explicitly has no dependencies"       | Bypass oracle   |
  | [0, 2]          | "Depends on prompts at indices 0 & 2"  | Bypass oracle   |

  - Indices are zero-based and reference other prompts in this same array
  - At chunk-create time, index references are translated to chunk directory names
  - Use `[]` when you've analyzed the chunks and determined they're independent
  - Omit the field when you don't have enough context to determine dependencies
- DO NOT POPULATE this array during narrative creation. It will be populated as
  chunks are created.
- Use `ve chunk list-proposed` to see all proposed chunks that haven't been created yet
-->

## Advances Trunk Goal

This narrative directly advances **Success Criteria #4** ("Baseline comparison via real training") from `docs/trunk/GOAL.md`. The curiosity_stream_mvp narrative proved the selection policy works in simulation; this narrative validates it with real gradient updates on a real language model.

It also advances the **selection_granularity investigation** (`docs/investigations/selection_granularity/`) — hypotheses H1-H4 about whether ability or difficulty carries more selection signal can only be conclusively tested with real training dynamics, not synthetic agents.

## Driving Ambition

The curiosity_stream_mvp narrative built and validated the full curiosity-driven curriculum pipeline in simulation: MDL scorer, stream generator, simulation harness, and visualization. But simulation uses a synthetic agent with fixed solve probabilities — it can't capture the feedback loop where training actually changes what the model can do, which changes what problems are productive, which changes what the curiosity signal selects.

To validate whether the curiosity policy produces a genuinely better training schedule, we need to close the loop: train a real LLM with GRPO (the RL algorithm used in DeepSeekMath and similar to REINFORCE++ from the MAA paper), where the curriculum is driven by CuriosityStream selecting problems based on live reward feedback.

The target setup is a single NVIDIA DGX Spark (128GB unified memory, limited memory bandwidth) running Qwen/Qwen3.5-9B. We use TRL's GRPOTrainer as the training framework — it's single-GPU native, supports custom reward functions, and handles generation + policy updates in one loop. Training time budget is hours, not days.

The key experiment: train the same base model twice — once with curiosity-driven curriculum selection, once with the MAA paper's fixed ascending curriculum (levels 1→2) — and compare reward curves, cell selection patterns, and training efficiency.

## Chunks

0. **Prompt & reward bridge** (`prompt_reward_bridge`) — Glue layer adapting MAA TaskCell problems into Qwen3.5-9B chat format and wrapping MAA reward functions into TRL's `reward_funcs` interface. Pure formatting and scoring code, no training logic. Depends on: nothing (uses existing TaskCell and reward functions).

1. **Curiosity GRPO training loop** (`curiosity_grpo_loop`) — Core training script integrating CuriosityStream with GRPOTrainer. Runs in rounds: stream selects problems → GRPO generates completions and trains → rewards fed back to stream. Configured for DGX Spark hardware. Supports both curiosity and fixed-curriculum modes. Logs stream trace in existing JSONL format. Depends on: chunk 0.

2. **Training comparison & analysis** (`training_comparison`) — Launch script running both curricula and producing unified analysis via the existing visualization module. Adds a summary report comparing training efficiency metrics. Depends on: chunk 1.

## Completion Criteria

When complete, a researcher can:

- Run a single command that trains Qwen3.5-9B on the MAA task space using a curiosity-driven curriculum, on a single DGX Spark, in a few hours.
- See a side-by-side comparison of reward curves between curiosity-driven and fixed-curriculum training, demonstrating whether adaptive selection improves training efficiency.
- Inspect the same stream metadata and visualizations from the MVP (cell selection timeline, MDL evolution, selection heatmap) — but now driven by real training dynamics instead of synthetic agents.
- Use the results to evaluate the selection_granularity investigation hypotheses (H1-H4) with empirical training data.