#!/usr/bin/env python3
"""Verify that RL_Minecraft uses the canonical Minecraft-HRL-Agent schema/vocabulary."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(PROJECT_ROOT / "data") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "data"))

from config import SKILL_VOCAB, TASKS
from RL_Minecraft.evaluator import (
    CANONICAL_SKILL_VOCAB,
    CANONICAL_TASKS,
    SKILL_ALIASES,
    TASK_STRATEGIES,
    canonicalize_path,
)


def assert_equal(label: str, left, right):
    if left != right:
        raise AssertionError(f"{label} mismatch")


def assert_subset(label: str, values, allowed):
    extra = sorted(set(values) - set(allowed))
    if extra:
        raise AssertionError(f"{label} contains non-canonical entries: {extra}")


def main():
    assert_equal("Task vocabulary", CANONICAL_TASKS, TASKS)
    assert_equal("Skill vocabulary", CANONICAL_SKILL_VOCAB, SKILL_VOCAB)

    assert_subset("Alias targets", SKILL_ALIASES.values(), SKILL_VOCAB)

    strategy_skills = [skill for strategies in TASK_STRATEGIES.values() for path in strategies for skill in path]
    assert_subset("Task strategies", strategy_skills, SKILL_VOCAB)

    example_dataset = json.loads((WORKSPACE_ROOT / "RL_Minecraft" / "example_dataset.json").read_text(encoding="utf-8"))
    example_tasks = [sample["task"] for sample in example_dataset]
    assert_subset("Example dataset tasks", example_tasks, TASKS)

    example_skills = []
    for sample in example_dataset:
        example_skills.extend(canonicalize_path(sample["reasoning_path"]))
    assert_subset("Example dataset skills", example_skills, SKILL_VOCAB)

    print("Vocabulary unification verified")
    print(f"  Tasks:  {len(TASKS)}")
    print(f"  Skills: {len(SKILL_VOCAB)}")
    print("  RL_Minecraft canonical vocab matches Minecraft-HRL-Agent config")


if __name__ == "__main__":
    main()
