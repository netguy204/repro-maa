---
status: DEFERRED
trigger: "Operator proposed replacing the MAA logic puzzle tasks with programming triples (program, input, output) as an alternative task domain for curiosity-driven curriculum training."
proposed_chunks: []
created_after: ["selection_granularity"]
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

The MAA paper's three meta-abilities (Deduction, Induction, Abduction) are operationalized as logic puzzles with heterogeneous formats. The operator observed that these same three reasoning modes map naturally to a single unified domain — **(program, input, output) triples**:

1. **Deduction**: Given program + input, predict output (execution/tracing)
2. **Induction**: Given input + output, predict program (synthesis)
3. **Abduction**: Given program + output, predict input (inverse execution)

This could replace the MAA logic puzzle generators entirely, with several potential advantages: trivially verifiable rewards (just execute the code), natural difficulty scaling (program complexity), and a shared substrate that makes the ability-vs-difficulty question from the selection_granularity investigation cleaner to study.

Worth investigating because if this domain works, it could become the primary task space for the curiosity training experiments — potentially a better vehicle than logic puzzles for studying curriculum selection dynamics.

## Success Criteria

1. **Feasibility assessment**: Determine whether (program, input, output) triples can serve as a drop-in replacement for the MAA task generators, satisfying the same interface (`TaskCell.generate(n)` and `TaskCell.score()`).
2. **Generator design**: Sketch a program generator that produces triples at controlled difficulty levels with deterministic, seedable generation.
3. **Reward verification**: Confirm that all three task modes can be reward-scored by execution alone, including edge cases (infinite loops, runtime errors, ambiguous outputs).
4. **Difficulty calibration**: Identify what difficulty dimensions exist (program length, nesting depth, number of variables, data structure complexity) and whether they provide enough variance for MDL-based curiosity to have signal.

## Testable Hypotheses

### H1: Programming triples provide cleaner ability separation than logic puzzles

- **Rationale**: In the MAA setup, the three abilities use completely different problem formats (propositional SAT, sequence completion, goal reachability), making it hard to tell whether curriculum selection is responding to the *reasoning mode* or just the *format*. With programming triples, all three tasks share the same substrate (code), so any curriculum preference must reflect reasoning mode differences.
- **Test**: Build a prototype generator, run the simulation harness with synthetic agents, and compare MDL score variance decomposition (ability vs difficulty) against the logic puzzle baseline.
- **Status**: UNTESTED

### H2: Execution-based rewards are more reliable than format-parsing rewards

- **Rationale**: The MAA reward functions parse structured text output (`<think>...</think><answer>...</answer>`) and have multiple failure modes (parse errors → -2, format errors → -1). Execution-based rewards have a single failure mode (wrong answer) and no format ambiguity.
- **Test**: Compare reward score distributions: fraction of rewards that are "noise" (format/parse failures vs actual reasoning failures).
- **Status**: UNTESTED

### H3: A restricted Python DSL is sufficient and avoids sandboxing complexity

- **Rationale**: Full Python execution requires sandboxing, timeouts, and import restrictions. A restricted DSL (arithmetic, list operations, conditionals, simple loops) would be easier to generate, execute safely, and control difficulty — at the cost of reduced naturalness.
- **Test**: Prototype both approaches and compare: generation quality, execution safety, and whether the model's training signal differs.
- **Status**: UNTESTED

### H4: Inverse execution (abduction) is significantly harder than the other two modes

- **Rationale**: Given a program and its output, finding a valid input may require search or constraint solving. For non-injective programs, multiple inputs could produce the same output. This mode may be intrinsically harder, mirroring the MAA finding that abduction gets the lowest merging weight (λa=0.1).
- **Test**: Measure baseline solve rates across the three modes at matched difficulty levels.
- **Status**: UNTESTED

## Exploration Log

### 2026-03-14: Initial concept framing

Identified the (program, input, output) triple mapping during discussion of the curiosity_training_run narrative. Key observations:

**Advantages over MAA logic puzzles:**
- Unified domain: all three abilities share the same substrate (programs)
- Trivial reward verification: execute the code, check the answer
- Natural difficulty scaling: program complexity, variable count, nesting depth
- More practical: programming is closer to real-world LLM use cases than Boolean SAT
- Cleaner signal for selection_granularity investigation: ability differences must reflect reasoning mode, not format differences

**Open questions:**
- What program language/DSL to use
- How to handle non-determinism and ambiguity in abduction mode
- Whether existing program synthesis datasets (MBPP, HumanEval, ARC) could seed the generator
- Sandboxing requirements for execution-based rewards

**Decision**: Defer investigation until after the curiosity_training_run narrative completes with MAA logic puzzles. The logic puzzles are already implemented and provide a working baseline. If training results are promising, revisit this as an alternative (or superior) task domain.

## Findings

### Verified Findings

None yet — investigation is deferred pending results from curiosity_training_run.

### Hypotheses/Opinions

- The mapping from (deduction, induction, abduction) to (execution, synthesis, inverse execution) is conceptually sound and well-established in the cognitive science literature.
- The shared substrate advantage is the strongest argument: it eliminates format-as-confound from the curriculum selection experiments.
- Sandboxing is solvable but not trivial — a restricted DSL would be the pragmatic choice for a first implementation.

## Proposed Chunks

No chunks proposed yet. Potential work depends on findings from the curiosity_training_run narrative:

- If curiosity training shows promise with logic puzzles: Build a programming triple generator and re-run the comparison to see if results improve.
- If curiosity training is inconclusive: Programming triples might provide cleaner signal — worth trying as an alternative domain.
- If curiosity training fails: Investigate whether the task domain (not the curriculum policy) was the problem.

## Resolution Rationale

*Deferred. This investigation will be revisited after the curiosity_training_run narrative completes. The current MAA logic puzzles provide a working baseline for validating the curiosity-driven curriculum approach. If that validation succeeds, programming triples become a high-priority follow-up for a potentially cleaner and more practical task domain.*