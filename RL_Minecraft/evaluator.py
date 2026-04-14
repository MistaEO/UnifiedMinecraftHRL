from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _load_canonical_config() -> tuple[list[str], list[str]]:
    """Load the canonical task/skill vocab from Minecraft-HRL-Agent when available."""
    workspace_root = Path(__file__).resolve().parents[1]
    config_path = workspace_root / "Minecraft-HRL-Agent" / "data" / "config.py"
    if not config_path.exists():
        return [], []

    spec = spec_from_file_location("minecraft_hrl_config", config_path)
    if spec is None or spec.loader is None:
        return [], []

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(getattr(module, "TASKS", [])), list(getattr(module, "SKILL_VOCAB", []))


CANONICAL_TASKS, CANONICAL_SKILL_VOCAB = _load_canonical_config()
CANONICAL_SKILL_SET = set(CANONICAL_SKILL_VOCAB)


# Older/live aliases that should collapse into the canonical dataset vocabulary.
SKILL_ALIASES = {
    "get_wood": "harvest_wood",
    "mine_iron": "mine_iron_ore",
    "mine_diamond": "mine_diamonds",
    "check_blacksmith": "find_blacksmith",
    "loot_chest_if_present": "loot_blacksmith_chest",
    "explore": "explore_cave",
}

IGNORED_SKILLS = {
    "idle",
    "place_crafting_table",
    "clear_junk",
}


TASK_STRATEGIES = {
    "obtain_wooden_pickaxe": [
        ["harvest_wood", "craft_planks_and_sticks", "craft_crafting_table", "craft_wooden_pickaxe"],
    ],
    "obtain_stone_pickaxe": [
        [
            "harvest_wood",
            "craft_planks_and_sticks",
            "craft_crafting_table",
            "craft_wooden_pickaxe",
            "mine_stone",
            "craft_stone_pickaxe",
        ],
    ],
    "obtain_iron_pickaxe": [
        [
            "harvest_wood",
            "craft_planks_and_sticks",
            "craft_crafting_table",
            "craft_wooden_pickaxe",
            "mine_stone",
            "craft_stone_pickaxe",
            "mine_iron_ore",
            "craft_furnace",
            "smelt_iron",
            "craft_iron_pickaxe",
        ],
        ["go_to_village", "find_blacksmith", "loot_blacksmith_chest"],
    ],
    "obtain_diamond_pickaxe": [
        [
            "harvest_wood",
            "craft_planks_and_sticks",
            "craft_crafting_table",
            "craft_wooden_pickaxe",
            "mine_stone",
            "craft_stone_pickaxe",
            "mine_iron_ore",
            "craft_furnace",
            "smelt_iron",
            "craft_iron_pickaxe",
            "dig_to_diamond_level",
            "mine_diamonds",
            "craft_diamond_pickaxe",
        ],
    ],
    "obtain_gold": [
        ["craft_iron_pickaxe", "dig_to_gold_level", "mine_gold_ore", "smelt_gold"],
    ],
    "obtain_food": [
        ["search_for_animals", "kill_animals_for_meat", "cook_meat"],
        ["go_to_village", "harvest_village_crops"],
        ["harvest_melons_from_ground"],
        ["harvest_sweet_berries_from_bushes"],
        ["milk_mooshroom_with_bowl"],
    ],
    "build_shelter": [
        ["build_walls_and_roof", "craft_and_place_door", "place_torches"],
        ["build_walls_and_roof", "craft_and_place_door"],
    ],
    "obtain_armor": [
        ["craft_iron_armor_set"],
        ["go_to_village", "find_blacksmith", "loot_blacksmith_chest"],
    ],
    "obtain_iron_armor": [
        ["craft_iron_armor_set"],
        ["go_to_village", "find_blacksmith", "loot_blacksmith_chest"],
    ],
    "explore_cave": [["explore_cave"]],
    "craft_furnace": [["craft_furnace"]],
    "smelt_iron": [["craft_furnace", "smelt_iron"]],
}


def canonicalize_path(path: Sequence[str]) -> List[str]:
    """Normalize a predicted/executed path into the canonical dataset vocabulary."""
    aliased = []
    for skill in path:
        if not isinstance(skill, str):
            continue
        mapped = SKILL_ALIASES.get(skill, skill)
        if mapped in IGNORED_SKILLS:
            continue
        aliased.append(mapped)

    canonical = []
    i = 0
    while i < len(aliased):
        window2 = aliased[i:i + 2]
        window4 = aliased[i:i + 4]

        if window2 == ["craft_planks", "craft_sticks"]:
            canonical.append("craft_planks_and_sticks")
            i += 2
            continue

        if window4 == [
            "craft_iron_helmet",
            "craft_iron_chestplate",
            "craft_iron_leggings",
            "craft_iron_boots",
        ]:
            canonical.append("craft_iron_armor_set")
            i += 4
            continue

        canonical.append(aliased[i])
        i += 1

    return canonical


def exact_match(pred, gold):
    return int(pred == gold)


def step_f1(pred, gold):
    pred_counts = Counter(pred)
    gold_counts = Counter(gold)

    overlap = sum(min(pred_counts[s], gold_counts[s]) for s in pred_counts)
    pred_total = sum(pred_counts.values())
    gold_total = sum(gold_counts.values())

    if pred_total == 0 or gold_total == 0:
        return 0.0

    precision = overlap / pred_total
    recall = overlap / gold_total

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def normalized_edit_distance(pred, gold):
    denom = max(len(pred), len(gold), 1)
    return edit_distance(pred, gold) / denom


def task_validity(task, pred):
    strategies = TASK_STRATEGIES.get(task, [])
    if not strategies:
        return 0.0

    pred_set = set(pred)
    best = 0.0
    for strategy in strategies:
        if not strategy:
            continue
        present = sum(1 for step in strategy if step in pred_set)
        best = max(best, present / len(strategy))
    return best


def average_metrics(results: Iterable[Dict[str, float]]) -> Dict[str, float]:
    results = list(results)
    if not results:
        return {}
    keys = [k for k, v in results[0].items() if isinstance(v, (int, float))]
    return {k: sum(float(r[k]) for r in results) / len(results) for k in keys}


def structure_key(sample: Dict) -> str:
    structures = sample.get("nearby_structures", ["none"])
    if isinstance(structures, str):
        structures = [structures]
    if not structures:
        return "none"
    return ",".join(sorted(structures))


class ReasoningPathEvaluator:
    def evaluate_sample(self, sample, prediction):
        gold = canonicalize_path(sample["reasoning_path"])
        pred = canonicalize_path(prediction)
        task = sample["task"]

        return {
            "exact_match": exact_match(pred, gold),
            "step_f1": step_f1(pred, gold),
            "normalized_edit_distance": normalized_edit_distance(pred, gold),
            "task_validity": task_validity(task, pred),
        }

    def evaluate_predictions(self, predictions_by_id, dataset):
        per_sample = []
        by_biome = defaultdict(list)
        by_structure = defaultdict(list)
        by_task = defaultdict(list)

        for idx, sample in enumerate(dataset):
            sample_id = str(sample.get("id", idx))
            if sample_id not in predictions_by_id:
                continue

            prediction = predictions_by_id[sample_id]
            metrics = self.evaluate_sample(sample, prediction)
            record = {
                "sample_id": sample.get("id", idx),
                "task": sample.get("task"),
                "biome": sample.get("biome"),
                "nearby_structures": sample.get("nearby_structures", ["none"]),
                "ground_truth": canonicalize_path(sample["reasoning_path"]),
                "prediction": canonicalize_path(prediction),
                **metrics,
            }
            per_sample.append(record)

            by_biome[sample.get("biome", "unknown")].append(metrics)
            by_structure[structure_key(sample)].append(metrics)
            by_task[sample.get("task", "unknown")].append(metrics)

        metric_only = [
            {
                "exact_match": record["exact_match"],
                "step_f1": record["step_f1"],
                "normalized_edit_distance": record["normalized_edit_distance"],
                "task_validity": record["task_validity"],
            }
            for record in per_sample
        ]

        return {
            "overall": average_metrics(metric_only),
            "by_biome": {k: average_metrics(v) for k, v in by_biome.items()},
            "by_structure": {k: average_metrics(v) for k, v in by_structure.items()},
            "by_task": {k: average_metrics(v) for k, v in by_task.items()},
            "per_sample": per_sample,
        }

    def evaluate_dataset(self, model, dataset):
        predictions_by_id = {}
        for idx, sample in enumerate(dataset):
            predictions_by_id[str(sample.get("id", idx))] = model.eval(sample)
        return self.evaluate_predictions(predictions_by_id, dataset)
