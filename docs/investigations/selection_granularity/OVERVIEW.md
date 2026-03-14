---
status: ONGOING
trigger: "Operator questioned whether the 3×5 ability×difficulty grid is the right selection granularity — hypothesizing that the ability axis carries most of the signal and difficulty may be noise."
proposed_chunks: []
created_after: ["scaffold_project"]
---

<!--
DO NOT DELETE THIS COMMENT until the investigation reaches a terminal status.
This documents the frontmatter schema and guides investigation workflow.

STATUS VALUES:
- ONGOING: Investigation is active; exploration and analysis in progress
- SOLVED: The investigation question has been answered. If proposed_chunks exist,
  implementation work remains—SOLVED indicates the investigation is complete, not
  that all resulting work is done.
- NOTED: Findings documented but no action required; kept for future reference
- DEFERRED: Investigation paused; may be revisited later when conditions change

TRIGGER:
- Brief description of what prompted this investigation
- Examples:
  - "Test failures in CI after dependency upgrade"
  - "User reported slow response times on dashboard"
  - "Exploring whether GraphQL would simplify our API"
- The trigger naturally captures whether this is an issue (problem to solve)
  or a concept (opportunity to explore)

PROPOSED_CHUNKS:
- Starts empty; entries are added if investigation reveals actionable work
- Each entry records a chunk prompt for work that should be done
- Format: list of {prompt, chunk_directory, depends_on} where:
  - prompt: The proposed chunk prompt text
  - chunk_directory: Populated when/if the chunk is actually created via /chunk-create
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
- Unlike narrative chunks (which are planned upfront), these emerge from investigation findings
-->

## Trigger

The curiosity_stream_mvp narrative models the training problem space as a 3×5 grid (3 meta-abilities × 5 difficulty levels = 15 task cells). The operator questioned whether this is the right granularity — specifically, whether the ability axis (deduction vs. induction vs. abduction) carries most of the curriculum selection signal and the difficulty axis is comparatively noise.

This matters because the MDL curiosity scorer's effectiveness depends on having meaningful variance across cells. If difficulty levels within an ability are largely interchangeable (the model either handles the ability or it doesn't), then 12 of our 15 cells are redundant and the selection problem degenerates. Conversely, if difficulty is the primary axis, we might want finer granularity there and coarser grouping of abilities. Getting this wrong means the curiosity signal optimizes over the wrong dimension.

## Success Criteria

1. **Quantify ability vs. difficulty contribution**: Measure what fraction of MDL score variance comes from the ability axis vs. the difficulty axis when running the simulation harness.
2. **Compare selection strategies**: Run the simulation with three configurations — ability-only selection (3 cells), difficulty-only selection (5 cells per ability), and the full 3×5 grid — and compare cumulative reward curves.
3. **Determine whether the grid should be restructured**: Produce a recommendation on whether to keep the 3×5 grid, collapse to ability-only selection, or adopt a hybrid (select ability first, then adaptively sample difficulty).

## Testable Hypotheses

### H1: The ability axis carries more selection signal than difficulty

- **Rationale**: The MAA paper provides strong indirect evidence. Merging weights are wildly unequal (λd=1.0, λi=0.2, λa=0.1), indicating the abilities have very different learning dynamics. The Oracle Ensemble gap (+11.1% vs. merged model's +2.5%) shows the abilities make complementary errors — they learn different things. Meanwhile, difficulty plateaus fast: the 7B model converges by Level 2 across all three abilities, suggesting the difficulty frontier is narrow.
- **Test**: Using the simulation harness, run the stream generator over the full 3×5 grid and decompose the MDL score variance into ability-marginal and difficulty-marginal components (ANOVA-style). If the ability factor explains >60% of variance, this hypothesis is supported.
- **Prediction**: Ability explains the majority of variance. Difficulty adds signal early (distinguishing level 1 from level 2) but becomes flat once the agent passes the initial learning phase.
- **Status**: UNTESTED

### H2: The difficulty axis plateaus because the fixed curriculum doesn't explore it well

- **Rationale**: The paper's finding that "the 7B model converges by Level 2 and doesn't improve at higher levels" might be an artifact of the fixed ascending schedule rather than a fundamental property. A curiosity-driven selector might find productive work at Level 3+ that the rigid schedule misses — perhaps because the agent needs to revisit lower levels intermittently, or because higher levels become tractable after ability-specific breakthroughs.
- **Test**: Run the curiosity stream over the full 3×5 grid with a learning synthetic agent (solve probability improves over time). Track whether the stream ever selects Level 3+ cells *and* whether those selections correlate with reward improvement. If the curiosity stream finds productive Level 3+ work that the fixed curriculum skips, this hypothesis is supported.
- **Prediction**: Partially supported — the curiosity stream will occasionally find Level 3 useful for deduction (which has the strongest learning signal) but not for abduction (which barely learns at all).
- **Status**: UNTESTED

### H3: A two-level hierarchical selector outperforms the flat 3×5 grid

- **Rationale**: The causal curiosity paper (Sontakke et al., 2021) uses hierarchical discovery — recursively splitting environments by one factor at a time. Applied here: first select which ability to focus on (the high-variance decision), then select which difficulty level within that ability (the low-variance refinement). This mirrors how the merging weights suggest unequal attention: deduction deserves 5× the training time of abduction.
- **Test**: Implement a two-level selector: outer level picks ability via MDL over ability-marginal rewards, inner level picks difficulty via MDL within the selected ability. Compare against the flat grid selector on cumulative reward. If the hierarchical selector matches or beats the flat grid with simpler signal, this hypothesis is supported.
- **Prediction**: The hierarchical selector will perform comparably to the flat grid but with more interpretable behavior — the outer selection will stabilize first (locking onto the most productive ability), then the inner selection will fine-tune difficulty.
- **Status**: UNTESTED

### H4: With only 3 abilities, ability-only selection is too coarse for MDL to work well

- **Rationale**: MDL-based curiosity scores a bimodal clustering of reward outcomes. With only 3 cells, each cell's reward window is large (all training steps contribute to one of three buckets), which may over-smooth the signal. The MDL criterion needs enough cells to distinguish "structured" from "trivial" reward distributions. Three cells may not provide enough contrast.
- **Test**: Run ability-only (3-cell) selection and measure whether the MDL scores for the three cells ever diverge meaningfully, or whether they track each other closely (indicating the signal has no discriminative power). Compare the MDL score spread (max - min across cells) for 3-cell vs. 15-cell configurations.
- **Prediction**: The 3-cell MDL spread will be smaller but still meaningful — the abilities are different enough that even with 3 cells, one will clearly dominate early.
- **Status**: UNTESTED

## Exploration Log

### 2026-03-14: Initial evidence gathering from MAA paper

Reviewed the MAA paper (arXiv:2505.10554) for evidence on ability vs. difficulty dynamics. Key findings extracted:

**Ability axis evidence:**
- Merging weights λd=1.0, λi=0.2, λa=0.1 — deduction contributes 5-10× more than abduction
- Oracle Ensemble gets +11.1% (math avg) vs. merged model's +2.5% — abilities are highly complementary with different error patterns
- Behavior frequency analysis: deduction appears in 29.4% of responses, induction 14%, abduction so rare they report raw counts (~17 instances)
- Each ability-aligned model has different downstream strengths: deduction best on MATH500/AIME, induction best on average

**Difficulty axis evidence:**
- 7B model converges by Level 2 for all three abilities — higher levels don't improve reward
- 32B model "occasionally benefits from Level 3 but shows unstable reward curves"
- Both scales use only Levels 1-2 in their final training recipe
- Difficulty hyper-parameters scale super-exponentially (2^n for deduction, |Σ|^k for induction) — the jump from level 2 to 3 may be too large

**Preliminary assessment:** The ability axis appears to dominate, but the difficulty evidence may be confounded by the fixed curriculum design. Investigation needs simulation data to separate these factors.

## Findings

### Verified Findings

None yet — awaiting simulation harness availability.

### Hypotheses/Opinions

- The paper's unequal merging weights and Oracle Ensemble gap are the strongest signals that ability selection matters more than difficulty selection. These are empirical findings from the paper, not our measurements.
- The difficulty plateau at Level 2 may be a property of the fixed curriculum rather than a fundamental ceiling. This is the most important open question — if H2 is verified, the 3×5 grid is justified despite H1 also being true.

## Proposed Chunks

No chunks proposed yet. Potential work depends on findings:

- If H1 is verified and H2 is falsified: Simplify the stream generator to ability-only selection with continuous difficulty sampling, removing the fixed 5-level grid.
- If H3 is verified: Implement the hierarchical two-level selector as an alternative mode in the stream generator.
- If all hypotheses are inconclusive: Add variance decomposition tracking to the simulation harness output and gather more data.

## Resolution Rationale

*Not yet resolved. This investigation will be explored incrementally as the simulation_harness and stream_visualization chunks become available. The key experiments (H1-H4) all require a working simulation with the MDL scorer and stream generator.*