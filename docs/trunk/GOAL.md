# Project Goal

## Problem Statement

The Meta-Ability Alignment (MAA) paper (Hu et al., 2025 — arXiv:2505.10554) demonstrates that independently training three core reasoning meta-abilities — **Deduction**, **Induction**, and **Abduction** — via reinforcement learning with program-verifiable rewards, then merging the resulting specialist models, produces a stronger general reasoner than training on any single ability or on mixed data.

This project takes that research as a starting point and asks: **can we build an RL training stream that teaches an agent to curiously explore high-signal learning opportunities?** Rather than hand-designing a fixed curriculum schedule (levels 1→5 per ability), we want to produce an *example stream* — an ordered sequence of training problems — where the ordering itself is guided by a curiosity-driven policy that identifies which problems, across all three meta-abilities and difficulty levels, would yield the most learning signal at each step.

### Causal Curiosity as the Selection Mechanism

The curiosity signal is inspired by *Causal Curiosity* (Sontakke et al., ICML 2021 — `curiosity.pdf`). That paper introduces an intrinsic reward based on **Minimum Description Length (MDL)**: an agent is rewarded for finding action sequences whose resulting trajectories cluster cleanly into distinct groups, thereby isolating single causal factors. The key insight: **low-complexity outcomes signal that the agent has found a behavior that exposes exactly one source of variation**.

We adapt this principle to curriculum selection over the MAA task space:

- **Environments** → the 15 task cells (3 abilities × 5 difficulty levels), each parameterized by the paper's difficulty hyper-parameters: `⟨nℓ, fℓ, dℓ⟩` for Deduction, `⟨k, Σ, m⟩` for Induction, `⟨d, g, h, γ⟩` for Abduction.
- **Action** → the choice of which cell to sample the next training batch from.
- **Trajectories** → the agent's reward outcomes (format reward + answer reward per the REINFORCE++ scheme) on problems from that cell.
- **Curiosity signal** → **in which task cell does the agent's performance carry the most information?** This is the MDL criterion applied to reward distributions: cells where the success/failure pattern has low description length (e.g., always +3 or always −3) are mastered or unreachable; cells with *structured but not trivial* reward patterns — where the agent sometimes succeeds and sometimes fails in learnable ways — are the frontier.

The MAA paper trains with a fixed ascending curriculum (level 1→2 for 7B, occasionally level 3 for 32B) and finds that "the 7B model converges by Level 2, and its reward does not improve further at higher levels." This is exactly the scenario where curiosity-driven selection adds value: rather than wasting training steps on levels that are too easy or too hard, the stream generator dynamically identifies the productive frontier as the agent's capabilities shift during training.

This connects the causal curiosity framework's hierarchical discovery (recursively splitting environments by one factor at a time) to a hierarchical curriculum: the stream generator can first identify which *meta-ability* the agent should focus on (the paper shows deduction gets weight λd=1.0 while abduction gets only λa=0.1, suggesting unequal learning dynamics), then which *difficulty level* within that ability offers the richest signal — a two-level causal decomposition of learning opportunity.

The target audience is RL researchers who want to reproduce and extend the MAA pipeline with adaptive, curiosity-based curriculum selection.

## Required Properties

- **MDL-based curiosity signal**: The system must produce an ordered stream of training problems (drawn from Deduction, Induction, and Abduction generators) where problem selection is guided by a Minimum Description Length criterion over the agent's reward distribution per task cell — not by a fixed schedule. The MDL signal operationalizes the causal curiosity reward: −L(O|M) where O is the set of reward outcomes and M is a clustering model over those outcomes.
- **High-signal filtering**: Problems in the stream must be selected to maximize the agent's learning rate. Concretely, the stream should favor problems where the agent's current reward is neither trivially high (already mastered) nor trivially low (too hard to learn from) — the "zone of proximal development." In MDL terms, this is the cell where the reward distribution has the highest description length under a bimodal model — neither pure nor random, but structured.
- **Program-verifiable rewards**: All problems in the stream must be verifiable without human annotation, preserving the MAA paper's core design principle. The existing reward functions (`formula.py`, `backward_reasoning.py`, `squence.py`) serve as the verification layer.
- **Reproducibility**: Given the same seed and model checkpoint, the stream must be deterministic.
- **Observable stream metadata**: Each item in the stream must carry metadata — the meta-ability type, difficulty level, predicted learning signal, and actual reward received — so that the curiosity policy's behavior can be analyzed post-hoc.

## Constraints

- **Builds on the MAA codebase**: The `Meta-Ability-Alignment/` submodule provides the data generators, VeRL training infrastructure, and reward functions. The stream generator should integrate with these components where possible.
- **Single-GPU training**: The full pipeline — curriculum selection, model inference, reward computation, and weight updates — must run on a single GPU. The target hardware is an NVIDIA DGX Spark (128GB unified memory, limited memory bandwidth) or comparable single-GPU setup.
- **Python 3.10+**: Consistent with the MAA repo's environment requirements

## Out of Scope

- **New meta-ability task types**: We use only the three existing task families (Deduction, Induction, Abduction). Designing new task generators is out of scope.
- **Model merging (Stage B)**: The mergekit integration for parameter-space merging is not part of this project.
- **Benchmark evaluation**: Evaluating on MATH, GSM8K, or other downstream benchmarks is out of scope. Success is measured by training reward curves and curriculum behavior, not end-task performance.
- **Multi-GPU distributed training**: Training must be runnable on a single GPU. Multi-node or multi-GPU orchestration is out of scope.

## Success Criteria

1. **Working stream generator**: A script that, given an agent's current performance profile (reward statistics per ability/level from the existing `formula.py`, `backward_reasoning.py`, `squence.py` reward functions), computes the MDL curiosity signal per task cell and emits the next batch of training problems from the highest-signal cell.
2. **Curiosity signal is measurable**: The MDL score per cell is computable and logged. The stream demonstrates measurable preference for frontier cells — where the bimodal clustering of reward outcomes has the highest description length (neither trivially separable nor uniform noise).
3. **Stream analysis artifacts**: The system produces a log showing, for each emitted batch: which ability/level was selected, the MDL curiosity score that drove the selection, the predicted vs. actual reward distribution, and a comparison to what a fixed curriculum would have selected — enabling post-hoc analysis of whether the curiosity policy made better choices.
4. **Baseline comparison via real training**: Train a language model twice — once with the curiosity-driven curriculum and once with the MAA paper's fixed ascending curriculum — and compare reward curves. The curiosity-driven run should lead to faster reward improvement on at least one of the three meta-abilities. The MAA paper notes the 7B model "converges by Level 2" with fixed scheduling — the curiosity stream should identify this plateau earlier and reallocate training to more productive cells.