"""
Validates whether a skill is executable given the current environment state.

Called between the LLM output and the embedding lookup so the RL policy is
never conditioned on a skill that is physically impossible right now.

Public API
----------
validate(skill, state)  -> (bool, reason_str)
valid_skills(state)     -> list[str]   # all currently executable skills
"""

from __future__ import annotations
from typing import Callable

from environment import SKILLS

# ── Inventory / state helpers ─────────────────────────────────────────────────

def _inv(state: dict) -> dict[str, int]:
    return state.get("inventory", {})

def _structs(state: dict) -> list[str]:
    return state.get("structures", [])

def _count(inv: dict[str, int], *keywords: str) -> int:
    """Sum counts of all inventory keys containing any keyword (substring)."""
    return sum(v for k, v in inv.items() if any(kw in k for kw in keywords))

def _has(inv: dict[str, int], *keywords: str, qty: int = 1) -> bool:
    return _count(inv, *keywords) >= qty

def _near(state: dict, structure: str) -> bool:
    return structure in _structs(state)

# ── Shared requirement builders ───────────────────────────────────────────────
# Each returns (is_met: bool, reason: str).

def _need(desc: str, check: Callable[[dict, dict], bool]):
    """Wrap a predicate into a (bool, str) requirement callable."""
    def req(inv: dict, state: dict) -> tuple[bool, str]:
        return (True, "") if check(inv, state) else (False, f"requires {desc}")
    return req

# Pickaxe tiers
_any_pickaxe    = _need("any pickaxe",    lambda inv, _: _has(inv, "pickaxe"))
_stone_pickaxe  = _need("stone/iron/diamond pickaxe",
                         lambda inv, _: _has(inv, "stone_pickaxe", "iron_pickaxe",
                                             "diamond_pickaxe", "netherite_pickaxe"))
_iron_pickaxe   = _need("iron/diamond pickaxe",
                         lambda inv, _: _has(inv, "iron_pickaxe", "diamond_pickaxe",
                                             "netherite_pickaxe"))

# Crafting prerequisites
_has_log        = _need("wood logs",      lambda inv, _: _has(inv, "log", "wood"))
_has_planks     = _need("planks",         lambda inv, _: _has(inv, "planks"))
_has_sticks     = _need("sticks",         lambda inv, _: _has(inv, "stick"))
_has_ctable     = _need("crafting table", lambda inv, _: _has(inv, "crafting_table"))
_has_cobble3    = _need("3× cobblestone", lambda inv, _: _has(inv, "cobblestone", qty=3))
_has_cobble8    = _need("8× cobblestone", lambda inv, _: _has(inv, "cobblestone", qty=8))
_has_coal_fuel  = _need("coal or charcoal fuel",
                         lambda inv, _: _has(inv, "coal", "charcoal", "log", "wood"))
_has_furnace    = _need("furnace",        lambda inv, _: _has(inv, "furnace"))
_has_iron_ore   = _need("iron ore",       lambda inv, _: _has(inv, "iron_ore", "raw_iron"))
_has_iron3      = _need("3× iron ingots", lambda inv, _: _has(inv, "iron_ingot", qty=3))
_has_iron24     = _need("24× iron ingots (full armour set)",
                         lambda inv, _: _has(inv, "iron_ingot", qty=24))
_has_diamonds3  = _need("3× diamonds",   lambda inv, _: _has(inv, "diamond", qty=3))
_has_gold_ore   = _need("gold ore",       lambda inv, _: _has(inv, "gold_ore", "raw_gold"))
_has_raw_meat   = _need("raw meat",       lambda inv, _: _has(inv, "raw_beef", "raw_pork",
                                                              "raw_chicken", "raw_mutton",
                                                              "raw_salmon", "raw_cod",
                                                              "raw_rabbit"))
_has_food       = _need("food to eat",    lambda inv, _: _has(inv, "beef", "pork", "chicken",
                                                              "bread", "apple", "carrot",
                                                              "potato", "mutton", "salmon",
                                                              "cod", "rabbit", "stew", "soup",
                                                              "melon", "berry", "mushroom"))
_has_bowl       = _need("bowl",           lambda inv, _: _has(inv, "bowl"))
_has_torch      = _need("torches",        lambda inv, _: _has(inv, "torch"))
_has_build_mat  = _need("building material (planks / cobblestone / dirt)",
                         lambda inv, _: _has(inv, "planks", "cobblestone", "dirt",
                                             "stone", "log", "wood"))
_has_planks6    = _need("6× planks (door)", lambda inv, _: _has(inv, "planks", qty=6))

# Structure requirements
def _struct_req(structure: str):
    return _need(f"nearby {structure}",
                 lambda inv, state, s=structure: _near(state, s))

_near_village       = _struct_req("village")
_near_blacksmith    = _struct_req("blacksmith")
_near_mineshaft     = _struct_req("mineshaft")
_near_ruined_portal = _struct_req("ruined_portal")
_near_desert_temple = _struct_req("desert_temple")
_near_jungle_temple = _struct_req("jungle_temple")
_near_shipwreck     = _struct_req("shipwreck")

# ── Per-skill requirement table ───────────────────────────────────────────────
# Each entry is a list of requirement callables; ALL must pass.
# Skills with no entry (or empty list) are always valid.

_REQUIREMENTS: dict[str, list[Callable]] = {
    # ── Wood & basic crafting ─────────────────────────────────────────────────
    # harvest_wood: always valid
    "craft_planks_and_sticks":  [_has_log],
    "craft_crafting_table":     [_has_planks],
    "craft_wooden_pickaxe":     [_has_ctable, _has_planks, _has_sticks],
    "craft_torch":              [_has_sticks, _has_coal_fuel],

    # ── Stone ─────────────────────────────────────────────────────────────────
    "mine_stone":               [_any_pickaxe],
    "mine_coal":                [_any_pickaxe],
    "craft_stone_pickaxe":      [_has_ctable, _has_cobble3, _has_sticks],

    # ── Iron path ─────────────────────────────────────────────────────────────
    "mine_iron_ore":            [_stone_pickaxe],
    "craft_furnace":            [_has_ctable, _has_cobble8],
    "smelt_iron":               [_has_furnace, _has_iron_ore, _has_coal_fuel],
    "craft_iron_pickaxe":       [_has_ctable, _has_iron3, _has_sticks],
    "craft_iron_armor_set":     [_has_ctable, _has_iron24],

    # ── Diamond path ──────────────────────────────────────────────────────────
    # Digging to -59 requires breaking deep stone; iron pickaxe is the minimum
    # that won't break before reaching diamond level.
    "dig_to_diamond_level":     [_iron_pickaxe],
    "mine_diamonds":            [_iron_pickaxe],
    "craft_diamond_pickaxe":    [_has_ctable, _has_diamonds3, _has_sticks],

    # ── Gold path ─────────────────────────────────────────────────────────────
    # Gold ore needs an iron pickaxe in 1.18+ (stone pickaxe gives no drops).
    "dig_to_gold_level":        [_stone_pickaxe],   # stone ok to dig down to -16
    "mine_gold_ore":            [_iron_pickaxe],
    "smelt_gold":               [_has_furnace, _has_gold_ore, _has_coal_fuel],

    # ── Food ──────────────────────────────────────────────────────────────────
    # search/kill/harvest: always valid — no tool required to punch animals
    "cook_meat":                [_has_furnace, _has_raw_meat, _has_coal_fuel],
    "harvest_village_crops":    [_near_village],
    # harvest_melons / harvest_sweet_berries / milk_mooshroom: biome-gated in
    # reality but we only enforce the explicit item requirement here.
    "milk_mooshroom_with_bowl": [_has_bowl],

    # ── Shelter ───────────────────────────────────────────────────────────────
    "build_walls_and_roof":     [_has_build_mat],
    "craft_and_place_door":     [_has_ctable, _has_planks6],
    "place_torches":            [_has_torch],

    # ── Structure interactions ─────────────────────────────────────────────────
    # go_to_* and navigate_to_*: always valid (navigation, no inventory needed)
    "find_blacksmith":          [_near_village],
    "loot_blacksmith_chest":    [_near_blacksmith],
    "search_mineshaft_chests":  [_near_mineshaft],
    "loot_portal_chest":        [_near_ruined_portal],
    "avoid_tnt_trap":           [_near_desert_temple],
    "swim_to_shipwreck":        [_near_shipwreck],
    "loot_supply_chest":        [_near_shipwreck],

    # ── Food consumption ──────────────────────────────────────────────────────
    "eat_food":                 [_has_food],
}


# ── Public API ────────────────────────────────────────────────────────────────

def validate(skill: str, state: dict) -> tuple[bool, str]:
    """
    Return (True, "") if the skill can be executed right now, or
    (False, reason) describing the first unmet prerequisite.
    """
    reqs = _REQUIREMENTS.get(skill, [])
    inv  = _inv(state)
    for req in reqs:
        ok, reason = req(inv, state)
        if not ok:
            return False, reason
    return True, ""


def valid_skills(state: dict) -> list[str]:
    """Return every skill that passes all its prerequisites in the current state."""
    return [s for s in SKILLS if validate(s, state)[0]]
