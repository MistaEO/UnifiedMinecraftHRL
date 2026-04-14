"""
tech_tree.py  —  Minecraft HRL Tech Tree
=========================================
Loads the canonical tech-tree JSON, validates the graph, exports training
artefacts, and provides a runtime reward manager.

Key upgrade (v1.2): JSON-first workflow.
  requires:          node must have been obtained at least once (binary)
  quantity_requires: inventory must hold >= N of item_id before craft unlocks

Usage:
    python tech_tree.py                        # validate + summary
    python tech_tree.py --export training      # write training_config.json
    python tech_tree.py --export dot           # write tech_tree.dot (Graphviz)
    python tech_tree.py --export all           # both
    python tech_tree.py --add                  # interactive add-node wizard
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

ROOT = Path(__file__).parent
JSON_PATH = ROOT / "tech_tree.json"
TRAINING_JSON_PATH = ROOT / "training_config.json"
YAML_PATH = ROOT / "tech_tree.yaml"

NodeType = Literal["resource", "craftable", "station", "gate", "milestone"]

TIER_LABELS = {
    0: "Mining wood",
    1: "Basic crafting",
    2: "Crafting stations",
    3: "Wooden tools",
    4: "Mining stone",
    5: "Stone tools",
    6: "Mining iron and smelting",
    7: "Iron tools",
    8: "Mining diamond",
    9: "Diamond tools",
}


# ── Dataclass ──────────────────────────────────────────────────────────────

@dataclass
class TechNode:
    id:                str
    label:             str
    tier:              int
    type:              NodeType
    reward:            float
    requires:          list[str]       = field(default_factory=list)
    quantity_requires: dict[str, int]  = field(default_factory=dict)
    one_shot:          bool            = False
    pos:               list[float]     = field(default_factory=lambda: [0.0, 0.0])
    notes:             str             = ""

    def to_dict(self) -> dict:
        d = {
            "id":                self.id,
            "label":             self.label,
            "tier":              self.tier,
            "type":              self.type,
            "reward":            self.reward,
            "requires":          self.requires,
            "quantity_requires": self.quantity_requires,
            "pos":               self.pos,
        }
        if self.one_shot:
            d["one_shot"] = True
        if self.notes:
            d["notes"] = self.notes
        return d


# ── Loader / saver ─────────────────────────────────────────────────────────

def default_tree_path() -> Path:
    if JSON_PATH.exists():
        return JSON_PATH
    if TRAINING_JSON_PATH.exists():
        return TRAINING_JSON_PATH
    return YAML_PATH


def _normalise_node_dict(record: dict, flat_requires: dict | None = None, flat_qty: dict | None = None) -> dict:
    node = dict(record)
    node_id = node.get("id")
    node.setdefault("quantity_requires", (flat_qty or {}).get(node_id, {}))
    node.setdefault("requires", (flat_requires or {}).get(node_id, []))
    node.setdefault("one_shot", False)
    node.setdefault("notes", "")
    node.setdefault("pos", [0, 0])
    return node


def load_tree(path: Path | None = None) -> tuple[dict, list[TechNode]]:
    path = Path(path or default_tree_path())

    if path.suffix.lower() in {".yaml", ".yml"}:
        if not HAS_YAML:
            raise RuntimeError("PyYAML is required to load YAML tech-tree files")
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        meta = raw.get("meta", {})
        raw_nodes = raw.get("nodes", [])
        flat_requires = {}
        flat_qty = {}
    else:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        meta = raw.get("meta", {"source": path.name})
        raw_nodes = raw.get("nodes", [])
        flat_requires = raw.get("requires", {})
        flat_qty = raw.get("quantity_requires", {})

    records = raw_nodes if isinstance(raw_nodes, list) else list(raw_nodes.values())
    nodes = [TechNode(**_normalise_node_dict(record, flat_requires, flat_qty)) for record in records]
    return meta, nodes


def save_tree(meta: dict, nodes: list[TechNode], path: Path | None = None) -> None:
    path = Path(path or JSON_PATH)
    data = {"meta": meta, "nodes": [n.to_dict() for n in nodes]}

    if path.suffix.lower() in {".yaml", ".yml"}:
        if not HAS_YAML:
            raise RuntimeError("PyYAML is required to save YAML tech-tree files")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")


# ── Validator ──────────────────────────────────────────────────────────────

def validate(nodes: list[TechNode]) -> list[str]:
    errors: list[str] = []
    index = {n.id: n for n in nodes}

    # 1. Duplicate ids
    seen: set[str] = set()
    for n in nodes:
        if n.id in seen:
            errors.append(f"Duplicate id: '{n.id}'")
        seen.add(n.id)

    # 2. Missing references in requires
    for n in nodes:
        for req in n.requires:
            if req not in index:
                errors.append(f"'{n.id}'.requires references '{req}' which does not exist")

    # 3. Missing references in quantity_requires
    for n in nodes:
        for req_id, qty in n.quantity_requires.items():
            if req_id not in index:
                errors.append(f"'{n.id}'.quantity_requires references '{req_id}' which does not exist")
            if not isinstance(qty, int) or qty < 1:
                errors.append(f"'{n.id}'.quantity_requires['{req_id}'] = {qty!r} — must be a positive int")

    # 4. Cycle detection via DFS (both requires and quantity_requires form edges)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n.id: WHITE for n in nodes}

    def dfs(nid: str, path: list[str]) -> None:
        color[nid] = GRAY
        all_deps = list(index[nid].requires) + list(index[nid].quantity_requires.keys())
        for dep in all_deps:
            if dep not in index:
                continue
            if color[dep] == GRAY:
                cycle = path[path.index(dep):] + [dep]
                errors.append(f"Cycle: {' -> '.join(cycle)}")
            elif color[dep] == WHITE:
                dfs(dep, path + [dep])
        color[nid] = BLACK

    for n in nodes:
        if color[n.id] == WHITE:
            dfs(n.id, [n.id])

    # 5. Reward regression — only between same-type craftable nodes
    PROGRESSION_TYPES = {"craftable"}
    for n in nodes:
        if n.type not in PROGRESSION_TYPES:
            continue
        for req_id in list(n.requires) + list(n.quantity_requires.keys()):
            if req_id not in index:
                continue
            parent = index[req_id]
            if parent.type not in PROGRESSION_TYPES:
                continue
            if n.tier <= parent.tier:
                continue
            if n.reward < parent.reward:
                errors.append(
                    f"Reward regression: '{req_id}' r={parent.reward} "
                    f"-> '{n.id}' r={n.reward} (tier {parent.tier}->{n.tier})"
                )

    # 6. Gates and milestones must have at least one dependency
    for n in nodes:
        if n.type in ("gate", "milestone"):
            if not n.requires and not n.quantity_requires:
                errors.append(f"Gate/milestone '{n.id}' has no prerequisites (requires or quantity_requires)")

    # 7. Valid types
    valid_types = {"resource", "craftable", "station", "gate", "milestone"}
    for n in nodes:
        if n.type not in valid_types:
            errors.append(f"'{n.id}' has invalid type '{n.type}'")

    return errors


# ── Quantity checker (used by reward manager) ──────────────────────────────

def _prereqs_satisfied(node: TechNode, unlocked: set[str], inventory: dict[str, int]) -> bool:
    """Return True if all requires AND quantity_requires are met."""
    for req in node.requires:
        if req not in unlocked:
            return False
    for req_id, qty in node.quantity_requires.items():
        if inventory.get(req_id, 0) < qty:
            return False
    return True


# ── Runtime reward manager ─────────────────────────────────────────────────

class TechTreeRewardManager:
    """
    Drop this into your HRL environment step() function.

    Example
    -------
        manager = TechTreeRewardManager.from_yaml()

        # Episode start:
        manager.reset()

        # Each time the agent obtains something:
        reward = manager.on_item_obtained("iron_ingot", count=3)

        # Dense shaping signal (add to env reward each step):
        shaping = manager.shaping_reward("iron_ingot", count=1)
    """

    def __init__(self, nodes: list[TechNode]):
        self.nodes: dict[str, TechNode] = {n.id: n for n in nodes}
        self._unlocked:    set[str]       = set()   # ids whose reward has fired
        self._inventory:   dict[str, int] = {}       # item_id → total count held
        self._fired_gates: set[str]       = set()   # one-shot gate ids fired
        self.episode_total: float = 0.0

    @classmethod
    def from_yaml(cls, path: Path = YAML_PATH) -> "TechTreeRewardManager":
        _, nodes = load_tree(path)
        return cls(nodes)

    @classmethod
    def from_json(cls, path: Path = TRAINING_JSON_PATH) -> "TechTreeRewardManager":
        _, nodes = load_tree(path)
        return cls(nodes)

    @classmethod
    def from_file(cls, path: Path) -> "TechTreeRewardManager":
        _, nodes = load_tree(path)
        return cls(nodes)

    def reset(self) -> None:
        """Call at the start of every episode."""
        self._unlocked.clear()
        self._inventory.clear()
        self._fired_gates.clear()
        self.episode_total = 0.0

    def seed_inventory(self, inventory: dict[str, int]) -> None:
        """Prime the manager from a starting inventory without awarding reward."""
        self.observe_inventory(inventory, grant_reward=False)

    def observe_inventory(self, inventory: dict[str, int], grant_reward: bool = True) -> float:
        """
        Synchronize the manager with a full inventory snapshot.

        This is useful in live environments where we observe the entire inventory
        after each high-level skill instead of individual item callbacks.
        """
        self._inventory = {
            item_id: int(count)
            for item_id, count in inventory.items()
            if int(count) > 0
        }

        reward = 0.0
        changed = True
        while changed:
            changed = False

            for item_id, node in self.nodes.items():
                if item_id in self._unlocked:
                    continue
                if self._inventory.get(item_id, 0) <= 0:
                    continue
                if not _prereqs_satisfied(node, self._unlocked, self._inventory):
                    continue

                self._unlocked.add(item_id)
                changed = True
                if grant_reward:
                    reward += node.reward
                    self.episode_total += node.reward

            gate_reward = self._check_gates(grant_reward=grant_reward)
            if gate_reward > 0:
                reward += gate_reward
                changed = True

        return reward

    # ── Main API ────────────────────────────────────────────────────────────

    def on_item_obtained(self, item_id: str, count: int = 1) -> float:
        """
        Call whenever the agent mines, crafts, or smelts `count` of `item_id`.
        Returns the total reward earned this step (terminal + gate rewards).
        Inventory is updated before checking prerequisites so the item itself
        counts toward its own quantity_requires (e.g. collecting the 3rd iron
        ingot triggers the iron_pickaxe craft reward on the same step).
        """
        if item_id not in self.nodes:
            return 0.0

        # Update inventory first
        self._inventory[item_id] = self._inventory.get(item_id, 0) + count

        node = self.nodes[item_id]
        reward = 0.0

        # Terminal reward — fires once per item (or every time if not one_shot)
        already_fired = item_id in self._unlocked
        if not already_fired or not node.one_shot:
            if _prereqs_satisfied(node, self._unlocked, self._inventory):
                if not already_fired:
                    reward += node.reward
                    self._unlocked.add(item_id)
                    self.episode_total += node.reward

        # Check gates every step (they may now be satisfiable)
        reward += self._check_gates()
        return reward

    def shaping_reward(self, item_id: str, count: int = 1) -> float:
        """
        Dense potential-based shaping reward.
        Returns a small reward proportional to progress made toward
        the nearest incomplete craft that uses item_id as an ingredient.

        Key properties:
        - Capped at required quantity: no incentive to over-collect
        - Scaled to 10% of the target node's terminal reward
        - Zero once the target node is already unlocked

        Call this in addition to on_item_obtained for dense guidance.
        """
        best = 0.0
        for node in self.nodes.values():
            if node.id in self._unlocked:
                continue  # already done, no shaping needed
            needed = node.quantity_requires.get(item_id, 0)
            if needed == 0:
                continue

            have_before = self._inventory.get(item_id, 0) - count  # before this step
            have_after  = self._inventory.get(item_id, 0)

            # Clamp both to [0, needed] — no credit beyond the requirement
            before_clamped = min(max(have_before, 0), needed)
            after_clamped  = min(have_after, needed)

            progress_delta = (after_clamped - before_clamped) / needed
            if progress_delta <= 0:
                continue  # already at cap or no progress

            # Scale by 10% of terminal reward — must stay << terminal
            shaping = progress_delta * node.reward * 0.1
            best = max(best, shaping)

        return best

    def subgoal_chain(self, target_id: str) -> list[str]:
        """
        Returns the ordered list of node ids that must be completed
        to unlock target_id (excluding already-unlocked nodes).
        Used by the mid-level policy to issue sub-goals.
        """
        if target_id not in self.nodes:
            return []
        visited: set[str] = set()
        chain:   list[str] = []

        def recurse(nid: str):
            if nid in visited or nid in self._unlocked:
                return
            visited.add(nid)
            node = self.nodes[nid]
            all_deps = list(node.requires) + list(node.quantity_requires.keys())
            for dep in all_deps:
                recurse(dep)
            chain.append(nid)

        recurse(target_id)
        return chain

    def needed_quantities(self, target_id: str) -> dict[str, int]:
        """
        Returns a flat map of {item_id: still_needed} across the entire
        sub-goal chain for target_id, accounting for current inventory.
        The mid-level policy can use this to issue "collect N of X" sub-goals.
        """
        chain = self.subgoal_chain(target_id)
        needed: dict[str, int] = {}
        for nid in chain:
            node = self.nodes[nid]
            for req_id, qty in node.quantity_requires.items():
                have = self._inventory.get(req_id, 0)
                still_need = max(0, qty - have)
                if still_need > 0:
                    needed[req_id] = max(needed.get(req_id, 0), still_need)
        return needed

    # ── Internal ────────────────────────────────────────────────────────────

    def _check_gates(self, grant_reward: bool = True) -> float:
        total = 0.0
        for nid, node in self.nodes.items():
            if node.type not in ("gate", "milestone"):
                continue
            if nid in self._fired_gates:
                continue
            if _prereqs_satisfied(node, self._unlocked, self._inventory):
                self._fired_gates.add(nid)
                self._unlocked.add(nid)
                if grant_reward:
                    self.episode_total += node.reward
                    total += node.reward
                    print(f"  [GATE] {node.label}  +{node.reward:.1f}")
        return total

    @property
    def unlocked(self) -> frozenset[str]:
        return frozenset(self._unlocked)

    @property
    def inventory(self) -> dict[str, int]:
        return dict(self._inventory)


# ── Exporters ─────────────────────────────────────────────────────────────

def build_training_config(nodes: list[TechNode]) -> dict:
    return {
        "nodes":              {n.id: n.to_dict() for n in nodes},
        "reward_table":       {n.id: n.reward for n in nodes},
        "requires":           {n.id: n.requires for n in nodes},
        "quantity_requires":  {n.id: n.quantity_requires for n in nodes},
        "one_shot_ids":       [n.id for n in nodes if n.one_shot],
        "gate_ids":           [n.id for n in nodes if n.type in ("gate", "milestone")],
        "tiers": {
            str(tier): [n.id for n in nodes if n.tier == tier]
            for tier in sorted({n.tier for n in nodes})
        },
    }


def export_training_config(nodes: list[TechNode], out: Path) -> None:
    config = build_training_config(nodes)
    with open(out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Exported training config -> {out}")


def export_dot(nodes: list[TechNode], out: Path) -> None:
    color_map = {
        "resource":  "#9FE1CB", "craftable": "#B5D4F4",
        "station":   "#CECBF6", "gate":      "#FAC775", "milestone": "#F4C0D1",
    }
    lines = ["digraph tech_tree {", "  rankdir=TB;",
             '  node [shape=box, style=filled, fontname="Helvetica"];']
    for n in nodes:
        qty_label = ""
        if n.quantity_requires:
            qty_label = "\\n" + ", ".join(f"{k}×{v}" for k, v in n.quantity_requires.items())
        lines.append(
            f'  "{n.id}" [label="{n.label}\\nr={n.reward}{qty_label}", '
            f'fillcolor="{color_map.get(n.type, "#eee")}"];'
        )
    lines.append("")
    for n in nodes:
        for req in n.requires:
            lines.append(f'  "{req}" -> "{n.id}";')
        for req_id, qty in n.quantity_requires.items():
            lines.append(f'  "{req_id}" -> "{n.id}" [label="×{qty}", style=dashed];')
    lines.append("}")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Exported Graphviz dot -> {out}")


# ── Interactive add-node wizard ────────────────────────────────────────────

def add_node_wizard(path: Path = YAML_PATH) -> None:
    meta, nodes = load_tree(path)
    index = {n.id: n for n in nodes}

    print("\n-- Add a new node ---------------------------------")
    nid = input("  id (snake_case):              ").strip()
    if nid in index:
        print(f"  Error: id '{nid}' already exists.")
        return

    label    = input("  label (display name):         ").strip()
    tier_raw = input("  tier (0-9):                   ").strip()
    type_raw = input("  type (resource/craftable/station/gate/milestone): ").strip()
    reward   = float(input("  reward (float):               ").strip())
    one_shot = input("  one_shot? (y/n):              ").strip().lower() == "y"
    req_raw  = input("  requires (space-sep ids):     ").strip()
    qty_raw  = input("  quantity_requires (e.g. iron_ingot:3 sticks:2 or blank): ").strip()
    notes    = input("  notes (optional):             ").strip()

    requires = [r.strip() for r in req_raw.split() if r.strip()]
    quantity_requires: dict[str, int] = {}
    for part in qty_raw.split():
        if ":" in part:
            k, v = part.split(":", 1)
            quantity_requires[k.strip()] = int(v.strip())

    node = TechNode(
        id=nid, label=label, tier=int(tier_raw), type=type_raw,
        reward=reward, requires=requires, quantity_requires=quantity_requires,
        one_shot=one_shot, pos=[0, int(tier_raw) * 120], notes=notes,
    )
    nodes.append(node)

    errors = validate(nodes)
    if errors:
        print("\n  Validation errors after adding node:")
        for e in errors:
            print(f"    x {e}")
        if input("  Save anyway? (y/n): ").strip().lower() != "y":
            print("  Aborted.")
            return

    save_tree(meta, nodes, path)
    print(f"\n  OK Node '{nid}' added successfully.")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Minecraft HRL Tech Tree tool")
    parser.add_argument("--tree",   dest="tree_path", default=str(default_tree_path()))
    parser.add_argument("--yaml",   dest="tree_path", help=argparse.SUPPRESS)
    parser.add_argument("--export", choices=["training", "dot", "all"])
    parser.add_argument("--add",    action="store_true")
    parser.add_argument("--out",    default=".")
    args = parser.parse_args()

    tree_path = Path(args.tree_path)
    out_dir   = Path(args.out)

    if args.add:
        add_node_wizard(tree_path)
        return

    print(f"\nLoading {tree_path} ...")
    try:
        meta, nodes = load_tree(tree_path)
    except Exception as e:
        print(f"  Failed to load tree: {e}")
        sys.exit(1)

    print(f"  {len(nodes)} nodes loaded\n")

    print("Validating ...")
    errors = validate(nodes)
    if errors:
        print(f"  {len(errors)} error(s) found:\n")
        for e in errors:
            print(f"    x {e}")
        if not args.export:
            sys.exit(1)
    else:
        print("  OK All checks passed\n")

    # Summary
    tier_counts = Counter(n.tier for n in nodes)
    type_counts = Counter(n.type for n in nodes)
    qty_nodes   = [n for n in nodes if n.quantity_requires]
    print("Summary:")
    for tier in sorted(tier_counts):
        print(f"  Tier {tier} ({TIER_LABELS.get(tier, '')}): {tier_counts[tier]} nodes")
    print()
    for t, c in sorted(type_counts.items()):
        print(f"  {t:12s}: {c}")
    print(f"\n  {len(qty_nodes)} nodes have quantity_requires")

    if args.export in ("training", "all"):
        export_training_config(nodes, out_dir / "training_config.json")
    if args.export in ("dot", "all"):
        export_dot(nodes, out_dir / "tech_tree.dot")


if __name__ == "__main__":
    main()
