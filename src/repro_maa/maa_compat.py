# Chunk: docs/chunks/scaffold_project - MAA import compatibility shim
"""
Compatibility shim for importing Meta-Ability-Alignment code.

The MAA codebase is not packaged (no __init__.py, no setup.py). This module
handles sys.path manipulation and re-exports the generator classes and reward
functions under clean names. All path-hacking is isolated here so downstream
code gets stable import paths.

Usage::

    from repro_maa.maa_compat import (
        DeductionSampler, DeductionFormatter,
        InductionGenerator,
        generate_abduction_problem,
        deduction_score, induction_score, abduction_score, mixed_score,
    )
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root – two levels up from this file (src/repro_maa/maa_compat.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_DATA_SYNTHESIS_DIR = str(REPO_ROOT / "Meta-Ability-Alignment" / "Data_Synthesis")
_REWARD_SCORE_DIR = str(REPO_ROOT / "Meta-Ability-Alignment" / "Training")


@contextmanager
def _temporary_sys_path(*paths: str):
    """Context manager that temporarily prepends *paths* to sys.path."""
    added = []
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    try:
        yield
    finally:
        for p in added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def _patch_main_for_abduction():
    """
    Abduction.py imports helper functions from __main__ that are expected to
    exist when the module is run as a script. Since we import it as a library,
    we inject no-op stubs into __main__ for the functions only used by
    check_consistency (which we don't call). generate_abduction_problem itself
    does not use these functions.
    """
    import __main__ as main_mod

    def _stub_parse_expression(expr_str):
        raise NotImplementedError("parse_expression stub — only needed by check_consistency")

    def _stub_build_eval_func(parsed):
        raise NotImplementedError("build_eval_func stub — only needed by check_consistency")

    def _stub_parse_premise(premise_str):
        raise NotImplementedError("parse_premise stub — only needed by check_consistency")

    def _stub_extract_atoms(premise_str):
        raise NotImplementedError("extract_atoms_from_premise stub — only needed by check_consistency")

    if not hasattr(main_mod, "parse_expression"):
        main_mod.parse_expression = _stub_parse_expression
    if not hasattr(main_mod, "build_eval_func"):
        main_mod.build_eval_func = _stub_build_eval_func
    if not hasattr(main_mod, "parse_premise"):
        main_mod.parse_premise = _stub_parse_premise
    if not hasattr(main_mod, "extract_atoms_from_premise"):
        main_mod.extract_atoms = _stub_extract_atoms
        main_mod.extract_atoms_from_premise = _stub_extract_atoms


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def _import_generators():
    """Import MAA generator classes / functions."""
    _patch_main_for_abduction()

    with _temporary_sys_path(_DATA_SYNTHESIS_DIR):
        from Deduction import NestedLogicPuzzleSampler, PuzzleFormatter  # type: ignore
        from Induction import SequencePuzzleGenerator  # type: ignore
        from Abduction import generate_abduction_problem as _gen_abd  # type: ignore

    return NestedLogicPuzzleSampler, PuzzleFormatter, SequencePuzzleGenerator, _gen_abd


# Lazy import on first access — keep module-level import fast
_DeductionSampler = None
_DeductionFormatter = None
_InductionGenerator = None
_generate_abduction_problem = None


def _ensure_generators():
    global _DeductionSampler, _DeductionFormatter, _InductionGenerator, _generate_abduction_problem
    if _DeductionSampler is None:
        (
            _DeductionSampler,
            _DeductionFormatter,
            _InductionGenerator,
            _generate_abduction_problem,
        ) = _import_generators()


class _LazyGeneratorProxy:
    """Descriptor-like helper that lazily resolves a generator on first call."""

    def __init__(self, attr_name: str):
        self._attr = attr_name

    def __call__(self, *args, **kwargs):
        _ensure_generators()
        return globals()[self._attr](*args, **kwargs)


# ---------------------------------------------------------------------------
# Public API — generators
# ---------------------------------------------------------------------------

def DeductionSampler(*args, **kwargs):  # noqa: N802
    """Proxy to ``Deduction.NestedLogicPuzzleSampler``."""
    _ensure_generators()
    return _DeductionSampler(*args, **kwargs)


def DeductionFormatter(*args, **kwargs):  # noqa: N802
    """Proxy to ``Deduction.PuzzleFormatter``."""
    _ensure_generators()
    return _DeductionFormatter(*args, **kwargs)


def InductionGenerator(*args, **kwargs):  # noqa: N802
    """Proxy to ``Induction.SequencePuzzleGenerator``."""
    _ensure_generators()
    return _InductionGenerator(*args, **kwargs)


def generate_abduction_problem(*args, **kwargs):
    """Proxy to ``Abduction.generate_abduction_problem``."""
    _ensure_generators()
    return _generate_abduction_problem(*args, **kwargs)


# ---------------------------------------------------------------------------
# Reward scoring functions
# ---------------------------------------------------------------------------

def _import_scorers():
    """Import MAA reward scoring functions.

    The verl package's __init__.py imports torch (which we don't need).
    To avoid pulling in that heavy dependency, we pre-register stub modules
    for the verl package hierarchy in sys.modules before importing the
    individual reward scoring modules.
    """
    import importlib
    import types

    _reward_score_dir = str(REPO_ROOT / "Meta-Ability-Alignment" / "Training"
                            / "verl" / "utils" / "reward_score")

    # Pre-register the verl package hierarchy with stub modules so Python
    # doesn't try to execute verl/__init__.py (which imports torch).
    stubs_needed = ["verl", "verl.utils", "verl.utils.reward_score"]
    created_stubs = []
    for mod_name in stubs_needed:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []  # make it a package
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub
            created_stubs.append(mod_name)

    # Point verl.utils.reward_score.__path__ at the actual directory so
    # submodule imports resolve correctly.
    sys.modules["verl.utils.reward_score"].__path__ = [_reward_score_dir]

    try:
        with _temporary_sys_path(_REWARD_SCORE_DIR):
            # Import the individual scoring modules directly
            import importlib.util

            def _load_score_module(name: str, filepath: str):
                spec = importlib.util.spec_from_file_location(
                    f"verl.utils.reward_score.{name}",
                    filepath,
                    submodule_search_locations=[],
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules[f"verl.utils.reward_score.{name}"] = mod
                spec.loader.exec_module(mod)
                return mod

            formula_mod = _load_score_module(
                "formula", f"{_reward_score_dir}/formula.py"
            )
            squence_mod = _load_score_module(
                "squence", f"{_reward_score_dir}/squence.py"
            )
            backward_mod = _load_score_module(
                "backward_reasoning", f"{_reward_score_dir}/backward_reasoning.py"
            )
            mix_mod = _load_score_module(
                "mix", f"{_reward_score_dir}/mix.py"
            )

        return (
            formula_mod.compute_score,
            squence_mod.compute_score,
            backward_mod.compute_score,
            mix_mod.compute_score,
        )
    except Exception:
        # Clean up stubs on failure so retry is possible
        for mod_name in created_stubs:
            sys.modules.pop(mod_name, None)
        raise


_deduction_score = None
_induction_score = None
_abduction_score = None
_mixed_score = None


def _ensure_scorers():
    global _deduction_score, _induction_score, _abduction_score, _mixed_score
    if _deduction_score is None:
        (
            _deduction_score,
            _induction_score,
            _abduction_score,
            _mixed_score,
        ) = _import_scorers()


# ---------------------------------------------------------------------------
# Public API — scorers
# ---------------------------------------------------------------------------

def deduction_score(*args, **kwargs) -> float:
    """Proxy to ``formula.compute_score``."""
    _ensure_scorers()
    return _deduction_score(*args, **kwargs)


def induction_score(*args, **kwargs) -> float:
    """Proxy to ``squence.compute_score``."""
    _ensure_scorers()
    return _induction_score(*args, **kwargs)


def abduction_score(*args, **kwargs) -> float:
    """Proxy to ``backward_reasoning.compute_score``."""
    _ensure_scorers()
    return _abduction_score(*args, **kwargs)


def mixed_score(*args, **kwargs) -> float:
    """Proxy to ``mix.compute_score``."""
    _ensure_scorers()
    return _mixed_score(*args, **kwargs)
