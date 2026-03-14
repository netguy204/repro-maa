# Chunk: docs/chunks/scaffold_project - Foundational project scaffolding
"""repro-maa: Reproducibility scaffold for Meta-Ability-Alignment experiments."""

__version__ = "0.1.0"

from repro_maa.mdl_scorer import MDLScorer  # noqa: F401
from repro_maa.simulation import (  # noqa: F401
    FixedCurriculumBaseline,
    StepRecord,
    SyntheticAgent,
    compare_runs,
    run_simulation,
)
from repro_maa.prompt_reward_bridge import (  # noqa: F401
    format_chat_prompt,
    make_reward_func,
)
from repro_maa.stream import BatchResult, CuriosityStream  # noqa: F401
from repro_maa.task_cell import TaskCell  # noqa: F401

__all__ = [
    "BatchResult",
    "CuriosityStream",
    "FixedCurriculumBaseline",
    "MDLScorer",
    "StepRecord",
    "SyntheticAgent",
    "TaskCell",
    "compare_runs",
    "format_chat_prompt",
    "make_reward_func",
    "run_simulation",
]
