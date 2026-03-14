<!--
This document captures HOW you'll achieve the chunk's GOAL.
It should be specific enough that each step is a reasonable unit of work
to hand to an agent.
-->

# Implementation Plan

## Approach

Build a single training script (`src/repro_maa/train.py`) that wires together the existing `CuriosityStream` / `FixedCurriculumBaseline` curriculum selectors with TRL's `GRPOTrainer` in a round-based outer loop. Each "round" is one curriculum step: select a cell → build a dataset of prompt/ground-truth pairs → let GRPOTrainer run one training step (generate completions + compute rewards + GRPO policy update) → feed the rewards back to the stream for MDL updates → log a `StepRecord` → optionally save a checkpoint.

**Key design choices:**

1. **Round-based outer loop, not a monolithic Trainer.train() call.** TRL's `GRPOTrainer` supports per-step datasets, but to switch the dataset every round (because the curriculum selects a different cell), we manage the outer loop ourselves and call the trainer for individual steps. This gives us control over the curriculum feedback loop.

2. **GRPOTrainer with custom reward functions.** We use `make_reward_func` from `prompt_reward_bridge` for scoring. TRL's `GRPOTrainer` accepts `reward_funcs` — callables that receive completions and keyword columns. We pass `ground_truth` and `ability` as extra dataset columns so the reward function can dispatch correctly.

3. **Single reward function that dispatches by ability.** Rather than three separate reward functions, we create a single dispatcher that reads the `ability` column from kwargs and delegates to the correct MAA scorer. This is cleaner since each batch is homogeneous (one ability per round), but the reward function must still handle the column.

4. **DGX Spark configuration.** Conservative settings for 128GB unified memory with limited bandwidth: `per_device_train_batch_size=2`, `num_generations=4` (GRPOTrainer generates multiple completions per prompt), `max_completion_length=512`, `max_prompt_length=512`, gradient checkpointing enabled. These are configurable via CLI args.

5. **JSONL logging reuses StepRecord.** Each round logs exactly one `StepRecord` entry, using the same schema the visualization module reads. The batch_rewards come from the reward function's output on the generated completions.

**Testing strategy (per TESTING_PHILOSOPHY.md):**

- **Unit tests**: Test the training configuration builder, dataset construction from BatchResult, the dispatching reward function, and the logging integration — all without loading a model or GPU.
- **Structural tests**: Verify the training script's argument parser, that the round loop produces well-formed StepRecords, and that checkpoint directories are created.
- **No GPU tests**: Per testing philosophy, all tests run on CPU. We mock the GRPOTrainer to verify our code calls it correctly without actually training.
- **Integration with existing modules**: Verify that CuriosityStream.emit_batch() output feeds cleanly into dataset construction, and that reward function output feeds back into CuriosityStream.update().

## Sequence

### Step 1: Add training dependencies to pyproject.toml

Add `torch`, `transformers`, `trl`, and `peft` to a new `training` optional dependency group. These are heavy dependencies that most of the codebase doesn't need, so they stay optional.

Location: `pyproject.toml`

Changes:
- Add `[project.optional-dependencies] training = ["torch", "transformers", "trl", "peft"]`
- Keep existing `dev` group unchanged

### Step 2: Create the dispatching reward function

Build a single reward function factory that reads the `ability` keyword argument from TRL's dataset column passthrough and dispatches to the correct MAA scorer. This wraps `make_reward_func` from `prompt_reward_bridge` but handles the per-round ability switching.

Location: `src/repro_maa/train.py` (new file, top section)

Inputs: `completions: list[str]`, `ground_truth: list[dict]`, `ability: list[str]` (from dataset column)
Output: `list[float]` rewards

The function should:
- Extract the ability from the first element (all elements in a batch share the same ability since each round selects one cell)
- Delegate to the appropriate `make_reward_func(ability)` closure
- Cache the per-ability reward functions to avoid re-creating TaskCell instances

### Step 3: Create the dataset builder

Write a function that takes a `BatchResult` from `CuriosityStream.emit_batch()` and produces a HuggingFace `Dataset` suitable for GRPOTrainer.

Location: `src/repro_maa/train.py`

The dataset must contain:
- `prompt`: list of chat message lists (from `format_chat_prompt(problem)` for each problem in the batch)
- `ground_truth`: list of ground-truth dicts (from each problem's `"ground_truth"` key)
- `ability`: list of ability strings (all the same within a round, used by the reward dispatcher)

GRPOTrainer expects the `prompt` column and passes all other columns as `**kwargs` to the reward functions.

### Step 4: Create the training configuration dataclass

Define a `TrainConfig` dataclass that holds all tunable parameters with DGX Spark defaults:

Location: `src/repro_maa/train.py`

Fields:
- `model_name: str = "Qwen/Qwen3.5-9B"` — base model
- `num_rounds: int = 100` — number of curriculum rounds
- `batch_size: int = 4` — problems per curriculum round (fed to CuriosityStream)
- `num_generations: int = 4` — completions per prompt (GRPO group size)
- `per_device_train_batch_size: int = 2` — micro-batch for gradient computation
- `max_prompt_length: int = 512`
- `max_completion_length: int = 512`
- `learning_rate: float = 1e-6`
- `gradient_checkpointing: bool = True`
- `bf16: bool = True` — use bfloat16 (Spark supports it)
- `epsilon: float = 0.1` — CuriosityStream exploration rate
- `window_size: int = 20` — reward history window
- `curriculum: str = "curiosity"` — "curiosity" or "fixed"
- `checkpoint_interval: int = 25` — save every N rounds
- `output_dir: str = "output"` — checkpoints and logs go here
- `seed: int = 42`
- `log_file: str = "stream_log.jsonl"` — JSONL log path (relative to output_dir)

### Step 5: Implement the round-based training loop

The core function: `run_training(config: TrainConfig) -> list[StepRecord]`

Location: `src/repro_maa/train.py`

Algorithm:
1. Load model and tokenizer with `AutoModelForCausalLM.from_pretrained` + quantization/gradient checkpointing settings.
2. Initialize TaskCells for the 3×5 grid.
3. Initialize curriculum: `CuriosityStream(cells, MDLScorer(), ...)` or `FixedCurriculumBaseline.maa_default(cells, config.num_rounds)` based on `config.curriculum`.
4. Build `GRPOConfig` from `TrainConfig` fields.
5. Create the dispatching reward function.
6. For each round `i` in `range(config.num_rounds)`:
   a. `batch = stream.emit_batch()` — get problems from the curriculum
   b. `dataset = build_dataset(batch)` — convert to HF Dataset
   c. Create or reinitialize `GRPOTrainer` with the round's dataset and reward function.
   d. `trainer.train()` — one training step (GRPOTrainer generates completions, scores them, updates policy).
   e. Extract the rewards from the trainer's logged metrics.
   f. Feed rewards back: `stream.update(cell, rewards)`
   g. Build and append a `StepRecord` with the round's metadata.
   h. Write the record to the JSONL log (append mode).
   i. If `i % config.checkpoint_interval == 0`, save model checkpoint.
7. Save final checkpoint.
8. Return the full list of `StepRecord`s.

**GRPOTrainer integration details:**
- TRL's `GRPOTrainer.__init__` accepts `model`, `reward_funcs`, `args` (GRPOConfig), and optionally `train_dataset`.
- We reconstruct the trainer each round with a new dataset (the selected cell's problems). This is the simplest approach — GRPOTrainer is lightweight to construct since it holds a reference to the model, not a copy.
- Alternative: use `trainer.train_dataset = new_dataset` if GRPOTrainer supports dataset swapping. Investigate during implementation.
- `GRPOConfig` extends `transformers.TrainingArguments`, so we set `max_steps=1` per trainer invocation to run exactly one gradient step per round.

### Step 6: Implement the CLI argument parser and main entry point

Add argument parsing so the script is runnable as `python -m repro_maa.train`.

Location: `src/repro_maa/train.py` (bottom section) + `src/repro_maa/__main__.py` (add training subcommand)

CLI arguments map 1:1 to `TrainConfig` fields:
- `--model-name`, `--num-rounds`, `--batch-size`, `--num-generations`
- `--per-device-train-batch-size`, `--max-prompt-length`, `--max-completion-length`
- `--learning-rate`, `--gradient-checkpointing / --no-gradient-checkpointing`
- `--bf16 / --no-bf16`, `--epsilon`, `--window-size`
- `--curriculum {curiosity,fixed}`, `--checkpoint-interval`
- `--output-dir`, `--seed`, `--log-file`

The `if __name__ == "__main__"` block at the bottom of `train.py` should parse args, build a `TrainConfig`, call `run_training(config)`, and print a summary.

### Step 7: Implement JSONL append-mode logging

Rather than buffering all StepRecords and writing at the end (as `simulation.py` does), the training loop should append each record after every round. Training runs are long; if interrupted, partial logs should be preserved.

Location: `src/repro_maa/train.py` (helper function)

Function: `append_step_record(path: Path, record: StepRecord) -> None`
- Opens the file in append mode
- Writes `to_jsonl_line(record) + "\n"`
- This reuses `to_jsonl_line` from `simulation.py`

### Step 8: Write unit tests for dataset construction and reward dispatching

Location: `tests/test_train.py` (new file)

Tests:
1. **test_build_dataset_shape**: Given a mock BatchResult with N problems, `build_dataset` returns a Dataset with N rows and columns `prompt`, `ground_truth`, `ability`.
2. **test_build_dataset_prompt_format**: Each row's `prompt` is a two-element chat message list (system + user) matching `format_chat_prompt` output.
3. **test_dispatching_reward_func**: The dispatcher correctly routes to deduction/induction/abduction scorers based on the `ability` kwarg. Test with mock completions and ground_truth.
4. **test_dispatching_reward_func_caches**: Calling the dispatcher twice with the same ability reuses the same underlying reward function (no TaskCell re-creation).

All tests mock TaskCell and MAA dependencies — no GPU, no model loading.

### Step 9: Write unit tests for the training loop's StepRecord emission

Location: `tests/test_train.py`

Tests:
1. **test_round_produces_step_record**: Mock GRPOTrainer and CuriosityStream. Run one round of the loop. Verify a StepRecord is produced with correct step index, ability, level, mdl_score, and non-empty batch_rewards.
2. **test_step_record_log_appendable**: Write two StepRecords via `append_step_record`. Read them back with `read_log`. Verify both records are present and correctly deserialized.
3. **test_fixed_curriculum_mode**: With `config.curriculum = "fixed"`, verify the loop creates a FixedCurriculumBaseline instead of CuriosityStream, and StepRecords have `selection_reason="fixed_schedule"`.

### Step 10: Write structural test for CLI argument parsing

Location: `tests/test_train.py`

Test:
1. **test_cli_defaults**: Parse an empty arg list. Verify TrainConfig has expected defaults (model name, batch size, etc.).
2. **test_cli_curriculum_flag**: Parse `--curriculum fixed`. Verify `config.curriculum == "fixed"`.
3. **test_cli_dgx_spark_defaults**: Verify default config values are tuned for DGX Spark (gradient_checkpointing=True, bf16=True, conservative batch sizes).

### Step 11: Update GOAL.md code_paths

Add the files this chunk creates/modifies to the GOAL.md frontmatter `code_paths` field.

Location: `docs/chunks/curiosity_grpo_loop/GOAL.md`

Files:
- `src/repro_maa/train.py`
- `tests/test_train.py`
- `pyproject.toml`

## Dependencies

- **Chunk: prompt_reward_bridge** (ACTIVE) — Provides `format_chat_prompt` and `make_reward_func` used in dataset construction and reward scoring.
- **Chunk: stream_generator** (ACTIVE) — Provides `CuriosityStream` and `BatchResult` for curriculum selection.
- **Chunk: simulation_harness** (ACTIVE) — Provides `StepRecord`, `to_jsonl_line`, `read_log`, `FixedCurriculumBaseline` for logging and baseline mode.
- **External: trl** — TRL library providing `GRPOTrainer` and `GRPOConfig`. Must be added to pyproject.toml.
- **External: transformers** — Hugging Face Transformers for model loading (`AutoModelForCausalLM`, `AutoTokenizer`).
- **External: torch** — PyTorch for GPU training.
- **External: peft** — Parameter-Efficient Fine-Tuning, used by GRPOTrainer for LoRA if needed.

## Risks and Open Questions

- **GRPOTrainer per-round reconstruction cost**: Creating a new GRPOTrainer each round (to swap the dataset) may have overhead. If so, investigate whether the trainer supports dataset swapping in-place via `trainer.train_dataset = ...` or `trainer.train(dataset)`. The plan assumes reconstruction is cheap since the model is passed by reference.
- **TRL reward_funcs signature evolution**: TRL's API has changed across versions. The `reward_funcs` interface (receiving `completions` as `list[str]` with `**kwargs` for extra columns) must be verified against the installed TRL version. Pin the TRL version in pyproject.toml if needed.
- **Memory pressure on DGX Spark**: 128GB unified memory with limited bandwidth is tight for a 9B model + GRPO's multiple generations. The default config (`num_generations=4`, `per_device_train_batch_size=2`, `max_completion_length=512`) may need tuning. Gradient checkpointing and bf16 are enabled by default to reduce footprint.
- **Extracting per-problem rewards from GRPOTrainer**: After `trainer.train()`, we need the individual reward scores to feed back to CuriosityStream. TRL may only log aggregated metrics. If per-problem rewards aren't directly accessible, we'll compute them ourselves by calling the reward function on the generated completions before passing to the trainer, or by intercepting them via a callback.
- **Chat template compatibility**: Qwen3.5-9B's chat template may differ from the `format_chat_prompt` output format. The tokenizer's `apply_chat_template` should handle this, but verify during implementation that the system/user message structure works correctly.

## Deviations

<!--
POPULATE DURING IMPLEMENTATION, not at planning time.
-->