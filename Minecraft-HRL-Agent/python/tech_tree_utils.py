from collections import defaultdict
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TECH_TREE_PATH = WORKSPACE_ROOT / "MC_Tech_Tree" / "training_config.json"
LEGACY_TECH_TREE_PATH = Path.home() / "Documents" / "GitHub" / "MC_Tech_Tree" / "training_config.json"

LOG_ITEMS = {
    "oak_log",
    "birch_log",
    "spruce_log",
    "jungle_log",
    "acacia_log",
    "dark_oak_log",
    "mangrove_log",
    "cherry_log",
}

PLANK_ITEMS = {
    "oak_planks",
    "birch_planks",
    "spruce_planks",
    "jungle_planks",
    "acacia_planks",
    "dark_oak_planks",
    "mangrove_planks",
    "cherry_planks",
}

DIRECT_NODE_ITEMS = {
    "coal",
    "torch",
    "crafting_table",
    "furnace",
    "wooden_pickaxe",
    "wooden_sword",
    "wooden_axe",
    "wooden_shovel",
    "stone_pickaxe",
    "stone_sword",
    "stone_axe",
    "iron_ingot",
    "iron_pickaxe",
    "iron_sword",
    "iron_helmet",
    "iron_chestplate",
    "iron_leggings",
    "iron_boots",
    "diamond",
    "diamond_pickaxe",
    "diamond_sword",
    "diamond_chestplate",
    "diamond_helmet",
    "diamond_leggings",
    "diamond_boots",
}


def default_tech_tree_path() -> Path:
    if LOCAL_TECH_TREE_PATH.exists():
        return LOCAL_TECH_TREE_PATH
    return LEGACY_TECH_TREE_PATH


def inventory_to_node_counts(inventory: dict[str, int]) -> dict[str, int]:
    counts = defaultdict(int)

    for item_name, count in inventory.items():
        if count <= 0:
            continue
        if item_name in LOG_ITEMS:
            counts["wood_log"] += count
        elif item_name in PLANK_ITEMS:
            counts["planks"] += count
        elif item_name == "stick":
            counts["sticks"] += count
        elif item_name in {"cobblestone", "stone"}:
            counts["stone"] += count
        elif item_name in {"raw_iron", "iron_ore"}:
            counts["iron_ore"] += count
        elif item_name in DIRECT_NODE_ITEMS:
            counts[item_name] += count

    return dict(counts)


def state_to_node_counts(state: dict) -> dict[str, int]:
    counts = inventory_to_node_counts(state.get("inventory", {}))
    nearby_blocks = state.get("nearby_blocks", {})

    if nearby_blocks.get("crafting_table"):
        counts["crafting_table"] = max(counts.get("crafting_table", 0), 1)
    if nearby_blocks.get("furnace"):
        counts["furnace"] = max(counts.get("furnace", 0), 1)

    return counts


def positive_node_deltas(previous: dict[str, int], current: dict[str, int]) -> dict[str, int]:
    deltas = {}
    for key in set(previous) | set(current):
        before = int(previous.get(key, 0))
        after = int(current.get(key, 0))
        if after > before:
            deltas[key] = after - before
    return deltas
