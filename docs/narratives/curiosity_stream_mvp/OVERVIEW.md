---
status: DRAFTING
advances_trunk_goal: "All four Success Criteria and Required Properties: MDL-based curiosity signal, High-signal filtering, Program-verifiable rewards, Observable stream metadata"
proposed_chunks:
  - prompt: "Set up this repository as a working Python project that can import and exercise critical parts of the Meta-Ability-Alignment submodule. Create pyproject.toml, pin dependencies, verify generators and reward functions are importable, provide a smoke test."
    chunk_directory: scaffold_project
    depends_on: []
  - prompt: "Wrap the three MAA data generators (Deduction, Induction, Abduction) in a unified TaskCell abstraction that can generate problems at any difficulty level and score agent responses using the existing reward functions. Each cell exposes: generate(n) → problems, score(response, ground_truth) → reward. This is the foundational data layer the rest of the system builds on."
    chunk_directory: taskcell_abstraction
    depends_on: [0]
  - prompt: "Implement the MDL curiosity scorer: given a window of recent reward outcomes for a task cell, compute the Minimum Description Length under a bimodal clustering model. The scorer should return high values for cells with structured-but-not-trivial reward distributions (the learning frontier) and low values for cells that are mastered (all pass) or unreachable (all fail). Include unit tests with synthetic reward distributions."
    chunk_directory: mdl_curiosity_scorer
    depends_on: [1]
  - prompt: "Build the stream generator that ties the task cells and MDL scorer together into a curiosity-driven curriculum policy. Each step: compute MDL score for all 15 cells using recent reward history, select the highest-signal cell, sample a batch of problems from it, and emit the batch with full metadata (ability, level, MDL score, predicted signal). Support both greedy selection and epsilon-greedy exploration."
    chunk_directory: stream_generator
    depends_on: [1, 2]
  - prompt: "Create a simulation harness that runs the stream generator against a synthetic agent model (a configurable function that maps difficulty to solve probability) to produce a full stream trace. Compare the curiosity-driven stream against the MAA paper's fixed ascending curriculum (levels 1→2) by measuring cumulative reward over N steps. Output a stream log with per-batch metadata and summary statistics."
    chunk_directory: simulation_harness
    depends_on: [3]
  - prompt: "Build stream analysis and visualization: read the stream log from the simulation harness and produce plots showing (1) which cells were selected over time, (2) MDL score evolution per cell, (3) cumulative reward comparison between curiosity-driven and fixed curriculum, (4) a heatmap of the 3×5 cell grid showing selection frequency. Output as static matplotlib figures."
    chunk_directory: stream_visualization
    depends_on: [4]
created_after: []
---

## Advances Trunk Goal

This narrative advances all four **Success Criteria** from `docs/trunk/GOAL.md`:

1. **Working stream generator** — chunks 0–2 build the generator from data layer through MDL scorer to selection policy.
2. **Curiosity signal is measurable** — chunk 1 implements the MDL computation; chunk 4 visualizes it.
3. **Stream analysis artifacts** — chunks 3–4 produce the log, comparison, and plots.
4. **Baseline comparison** — chunk 3 runs the curiosity stream against the fixed curriculum.

It also directly delivers the **Required Properties**: MDL-based curiosity signal, high-signal filtering, program-verifiable rewards (via existing reward functions), and observable stream metadata.

## Driving Ambition

The MAA paper uses a hand-designed ascending curriculum (level 1→2→…) to train reasoning specialists. This works but wastes training steps: the 7B model converges by Level 2 and gains nothing from higher levels. We want to replace this with a curiosity-driven stream that automatically finds the productive training frontier.

The core idea, borrowed from Causal Curiosity (Sontakke et al., 2021), is to use **Minimum Description Length** as an intrinsic reward over the agent's reward distribution per task cell. Cells where the agent always succeeds or always fails have low MDL (boring); cells where outcomes are structured but not trivial have high MDL (the learning frontier). The stream generator picks the highest-MDL cell at each step.

The MVP delivers: a unified interface to the three MAA task generators, an MDL curiosity scorer, a stream generator that selects tasks based on curiosity, a simulation comparing it to the fixed curriculum, and visualization of the results. This is the minimum needed to validate whether the curiosity approach outperforms fixed scheduling — without requiring a full GPU training run.

## Chunks

0. **Project scaffold** (`scaffold_project`) — Set up `pyproject.toml`, pin dependencies, verify that MAA generators and reward functions are importable, provide a smoke test. No dependencies.

1. **Task cell abstraction** — Wrap the three MAA generators and reward functions in a unified `TaskCell` interface with `generate(n)` and `score()` methods. Depends on chunk 0 (needs working imports).

2. **MDL curiosity scorer** — Given a window of recent rewards for a cell, compute the MDL under a bimodal model. High MDL = learning frontier. Includes unit tests with synthetic distributions. Depends on chunk 1 (needs `TaskCell` to define the reward type).

3. **Stream generator** — The curriculum policy: compute MDL for all 15 cells, pick the highest-signal cell, emit a batch with metadata. Supports greedy and epsilon-greedy selection. Depends on chunks 1 and 2.

4. **Simulation harness** — Run the stream generator against a synthetic agent (configurable solve-probability function) and compare curiosity-driven vs. fixed curriculum over N steps. Produces the stream log. Depends on chunk 3.

5. **Stream analysis and visualization** — Read the stream log, produce matplotlib plots: cell selection over time, MDL evolution, cumulative reward comparison, and a 3×5 selection heatmap. Depends on chunk 4.

## Completion Criteria

When complete, a researcher can:

- Run a single command that generates a curiosity-driven training stream over the MAA task space, using an MDL-based signal to select which problems to train on next.
- See a concrete comparison showing whether the curiosity stream allocates training more efficiently than the paper's fixed level 1→2 curriculum.
- Inspect per-batch metadata and visualizations that explain *why* each task cell was selected — making the curiosity policy's behavior transparent and analyzable.
- Use the stream generator as a drop-in curriculum policy for the existing VeRL training pipeline (the `TaskCell` interface produces problems in the same format the MAA reward functions expect).