#!/usr/bin/env python
# Chunk: docs/chunks/scaffold_project - Smoke test for MAA generators and scorers
"""
Smoke test: generates one problem per ability/level, scores a dummy response
against each, and prints a summary table. Exits 0 if all cells produce a score.
"""
from __future__ import annotations

import sys

from repro_maa.maa_compat import (
    DeductionSampler,
    DeductionFormatter,
    InductionGenerator,
    generate_abduction_problem,
    deduction_score,
    induction_score,
    abduction_score,
)


def _wrap_response(answer_body: str) -> str:
    return f"Assistant: <think>Thinking...</think><answer>{answer_body}</answer>"


def run_deduction(level: int) -> float:
    sampler = DeductionSampler(difficulty=level, seed=42)
    puzzles = sampler.sample_unique(1)
    formulas, assignment = puzzles[0]
    fmt = DeductionFormatter(formulas, assignment)
    gt = {"solution_text_format": fmt.solution_text()}
    # Dummy: claim all variables are True (likely wrong)
    dummy = "\n".join(f"({i}) {v} is True" for i, v in enumerate(sorted(assignment), 1))
    return deduction_score(_wrap_response(dummy), gt)


def run_induction(level: int) -> float:
    gen = InductionGenerator(seed=42)
    puzzles = gen.generate_puzzles(num=1, level=level)
    gt = {"solution_text_format": puzzles[0]["solution_text"]}
    # Dummy: guess 0 (likely wrong)
    return induction_score(_wrap_response("0"), gt)


def run_abduction(level: int) -> float:
    problem = generate_abduction_problem(
        problem_id=1,
        num_goals=max(1, level),
        reachable_k=1,
        chain_depth=level + 1,
        distractors=3,
        cycle_prob=0.1,
    )
    goals = problem["goals"]
    # Dummy: classify all as reachable
    dummy = "\n".join(f"({i}) {g} is reachable" for i, g in enumerate(goals, 1))
    gt = {
        "solution_text_format": "\n".join(
            f"({i}) {g} is {'reachable' if g in problem['reachable_goals'] else 'unreachable'}"
            for i, g in enumerate(goals, 1)
        )
    }
    return abduction_score(_wrap_response(dummy), gt)


ABILITIES = {
    "Deduction": run_deduction,
    "Induction": run_induction,
    "Abduction": run_abduction,
}


def main() -> int:
    results: list[tuple[str, int, float]] = []
    errors = 0

    for ability_name, run_fn in ABILITIES.items():
        for level in range(1, 4):  # levels 1–3 for speed
            try:
                score = run_fn(level)
                results.append((ability_name, level, score))
            except Exception as exc:
                print(f"ERROR: {ability_name} level {level}: {exc}", file=sys.stderr)
                errors += 1

    # Print summary table
    print()
    print(f"{'Ability':<12} {'Level':>5} {'Score':>8}")
    print("-" * 28)
    for ability, level, score in results:
        print(f"{ability:<12} {level:>5} {score:>8.1f}")
    print("-" * 28)
    print(f"Total cells: {len(results)}, Errors: {errors}")

    if errors > 0:
        return 1
    if len(results) == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
