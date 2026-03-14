---
status: ACTIVE
ticket: null
parent_chunk: null
code_paths:
- src/repro_maa/train.py
- tests/test_train.py
- pyproject.toml
code_references:
- ref: src/repro_maa/train.py#make_dispatching_reward_func
  implements: "Single TRL-compatible reward function that dispatches by ability and caches per-ability scorers"
- ref: src/repro_maa/train.py#build_dataset
  implements: "Converts BatchResult into HuggingFace Dataset with prompt, ground_truth, ability columns for GRPOTrainer"
- ref: src/repro_maa/train.py#TrainConfig
  implements: "Training configuration dataclass with DGX Spark defaults (128GB unified memory, conservative batch sizes)"
- ref: src/repro_maa/train.py#append_step_record
  implements: "JSONL append-mode logging for StepRecords (survives interrupted runs)"
- ref: src/repro_maa/train.py#run_training
  implements: "Round-based GRPO training loop integrating CuriosityStream with GRPOTrainer"
- ref: src/repro_maa/train.py#_extract_rewards
  implements: "Per-problem reward extraction from GRPOTrainer metrics"
- ref: src/repro_maa/train.py#_build_cells
  implements: "3x5 task cell grid construction"
- ref: src/repro_maa/train.py#parse_args
  implements: "CLI argument parser mapping flags 1:1 to TrainConfig fields"
- ref: src/repro_maa/train.py#main
  implements: "CLI entry point for the training script"
narrative: curiosity_training_run
investigation: null
subsystems: []
friction_entries: []
bug_type: null
depends_on:
- prompt_reward_bridge
created_after:
- mdl_curiosity_scorer
- simulation_harness
- stream_generator
- stream_visualization
- taskcell_abstraction
---

# Chunk Goal

## Minor Goal

Implement a curiosity-aware GRPO training loop that integrates CuriosityStream with TRL's GRPOTrainer. The outer loop runs in rounds: CuriosityStream selects a batch of problems from the highest-MDL cell, GRPOTrainer generates completions and trains on them, MAA reward scores are fed back to CuriosityStream for MDL updates. Configure for DGX Spark (Qwen/Qwen3.5-9B, 128GB unified memory, single GPU, conservative batch sizes for limited memory bandwidth). Support both curiosity-driven and fixed-curriculum modes via the existing FixedCurriculumBaseline. Log the full stream trace (cell selections, MDL scores, rewards) in the same JSONL format the visualization module already reads.

## Success Criteria

1. **Training script**: A runnable script that loads Qwen/Qwen3.5-9B, initializes CuriosityStream over the 3x5 task cell grid, and trains via GRPOTrainer with curiosity-driven problem selection.
2. **Round-based curriculum loop**: Each round, CuriosityStream selects the highest-MDL cell, generates problems, GRPO trains on them, and rewards are fed back. The loop runs for a configurable number of rounds.
3. **Fixed-curriculum mode**: The same script supports a `--curriculum fixed` flag that uses FixedCurriculumBaseline instead of CuriosityStream, for baseline comparison.
4. **DGX Spark configuration**: Batch sizes, sequence lengths, and gradient checkpointing tuned for 128GB unified memory with limited bandwidth. Training completes in hours, not days.
5. **Stream logging**: Every round logs a StepRecord-compatible JSONL entry (cell selected, MDL score, rewards, cumulative reward) readable by the existing visualization module.
6. **Checkpoint saving**: Model checkpoints saved at configurable intervals for post-hoc analysis.

## Rejected Ideas

### Use VeRL instead of TRL

The MAA paper uses VeRL for training, which would maximize fidelity to the original pipeline. Rejected because VeRL requires Ray orchestration and is designed for multi-GPU distributed training. TRL's GRPOTrainer is single-GPU native, simpler to set up, and implements GRPO which is comparable to REINFORCE++.