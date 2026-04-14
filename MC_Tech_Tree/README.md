# Minecraft HRL Tech Tree

Hierarchical reward system for a Minecraft HRL agent. Defines what items exist,
what they require to unlock, how many of each material are needed to craft them,
and what reward fires when the agent first obtains each one.

---

## Files

| File | Purpose |
|---|---|
| `tech_tree.json` | Canonical committed source of truth for the tech tree. |
| `tech_tree_seed.js` | Editor boot data mirrored from `tech_tree.json`. |
| `tech_tree_editor.html` | Visual editor. Open in any browser, no server needed. |
| `tech_tree.py` | Validator, exporter, and runtime reward manager. |
| `training_config.json` | Flat training artifact consumed by `Minecraft-HRL-Agent`. |

---

## Workflow

`tech_tree.json` is the committed source of truth. The editor boots from the
mirrored `tech_tree_seed.js`, and `training_config.json` is the flat runtime
artifact consumed by the HRL code.

Typical loop:

1. Open `tech_tree_editor.html` in a browser
2. If needed, click **Import JSON** and load `tech_tree.json` or `training_config.json`
3. Edit nodes visually — add, connect, delete, change rewards
4. Click **Save as Default** to persist your state across browser sessions
5. Click **Export JSON** to refresh `tech_tree.json`
6. Click **Export Training** to refresh `training_config.json`
7. `Minecraft-HRL-Agent` loads `training_config.json`

To restore the tree on another machine, use **Import JSON** with either
`tech_tree.json` or `training_config.json`.

The Python file can also validate and regenerate `training_config.json` from
`tech_tree.json`:

```bash
python tech_tree.py --export training
```

---

## Tier Structure

| Tier | Contents |
|---|---|
| 0 | Wood log (only free resource) |
| 1 | Planks, sticks |
| 2 | Crafting table |
| 3 | Wooden tools and weapons |
| 4 | Coal, stone, torch (hard-gated behind wooden pickaxe) |
| 5 | Furnace, stone tools |
| 6 | Iron ore, iron ingot, smelt gate, mine-iron gate |
| 7 | Iron tools, iron armor, iron combat gates |
| 8 | Mine-diamond gate, raw diamond (hard-gated behind iron pickaxe) |
| 9 | Diamond tools, diamond armor, full-diamond milestone |

---

## Node Types

- **resource** — mined or collected from the world
- **craftable** — produced at a crafting table or furnace
- **station** — crafting table, furnace (one-shot; reward fires once per episode)
- **gate** — capability milestone that fires when prerequisites are met (one-shot)
- **milestone** — major objective (one-shot, large reward)

---

## Prerequisite System

Each node has two independent prerequisite fields:

**`requires`** — binary existence check. The listed nodes must have been obtained
at least once before this node can unlock. Used for hard tool gates:

```
iron_ore  requires [stone_pickaxe]   # wooden pickaxe drops nothing — hard lock
diamond   requires [iron_pickaxe]    # stone pickaxe drops nothing — hard lock
```

**`quantity_requires`** — inventory count check. The agent must hold at least N
of each listed item before the craft reward fires. Reflects actual Minecraft
recipe costs:

```
iron_pickaxe     quantity_requires: {iron_ingot: 3, sticks: 2}
iron_chestplate  quantity_requires: {iron_ingot: 8}
nether_portal    quantity_requires: {obsidian: 10}
```

Both checks must pass for a node's reward to fire.

In the editor, `requires` edges are drawn as **solid blue arrows**.
`quantity_requires` edges are drawn as **dashed amber arrows** labelled with ×N.

---

## Reward Design

Rewards increase with progression depth. Small dense rewards early, large sparse
rewards at capability gates and milestones:

```
wood_log        +0.3    planks            +0.4
crafting_table  +1.0    wooden_pickaxe    +1.0
stone_pickaxe   +1.5    gate_mine_iron    +3.0  ← capability unlock
iron_pickaxe    +3.0    gate_mine_diamond +5.0
diamond_pickaxe +6.0    gate_full_diamond +12.0
```

**One-shot nodes** (stations, gates, milestones) fire their reward exactly once
per episode regardless of how many times the agent re-obtains them.

**Why no tool-tier reward multipliers:** The agent is incentivised to upgrade
tools through the tree's own structure — iron pickaxe gives +3.0 and unlocks a
+5.0 gate, an +8.0 pull signal. The environment's step cost handles efficiency
without needing reward shaping.

---

## Using the Reward Manager in Training

```python
from tech_tree import TechTreeRewardManager

manager = TechTreeRewardManager.from_json("training_config.json")

# --- Episode start ---
manager.reset()

# --- Each time the agent obtains something ---
reward = manager.on_item_obtained("iron_ingot", count=1)

# --- Dense shaping signal (add to env reward each step) ---
shaping = manager.shaping_reward("iron_ingot", count=1)
# Shaping is capped at required quantity — no farming incentive

# --- Mid-level policy: what does the agent still need? ---
still_needed = manager.needed_quantities("iron_pickaxe")
# → {"iron_ingot": 2, "sticks": 2}  (if agent already has 1 ingot)

# --- Sub-goal chain for a target ---
chain = manager.subgoal_chain("iron_pickaxe")
# → ["iron_ore", "iron_ingot", "iron_pickaxe"]  (excluding already-unlocked)
```

---

## Editor Reference

Open `tech_tree_editor.html` in any browser — no install, no server.

| Mode | Key | Action |
|---|---|---|
| Select | V | Drag nodes to reposition, click to highlight |
| Edit | E | Click a node to open the edit modal |
| Connect | C | Click source node, then click target node to draw an edge |
| Delete | D | Click a node or edge to remove it |
| — | N | Open add-node modal |
| — | Ctrl+Z | Undo last change (40-step history) |
| — | Ctrl+S | Export JSON |
| — | Escape | Cancel current action, return to Select |

**Toolbar buttons:**

| Button | Action |
|---|---|
| + Add Node | Add a new node at canvas centre |
| Auto Layout | Reposition all nodes by tier |
| Save as Default | Persist current state to localStorage (survives page reload) |
| Export JSON | Download canonical editor state as `tech_tree.json` |
| Export Training | Download flat runtime config as `training_config.json` |
| Import JSON | Load a previously exported JSON back into the editor |

**Connect mode details:** First click picks the source node (green glow, dashed
arrow follows cursor). Second click on any other node completes the edge and adds
it to that node's `requires` list. Clicking the same node or clicking empty canvas
cancels. Solid blue = binary `requires`. Dashed amber = `quantity_requires`.

---

## Validation Checks

The Python validator (`tech_tree.py`) catches:

- Duplicate node IDs
- `requires` or `quantity_requires` referencing a non-existent node
- Cycles in the prerequisite graph
- Reward regression between same-type craftable nodes
- Gates or milestones with no prerequisites
- Invalid node type values

## Repo Contract

- `tech_tree.json` is the canonical committed source file.
- `tech_tree_seed.js` must mirror `tech_tree.json` so the editor boots into the committed tree.
- `training_config.json` must be regenerated from the same tree before training.
- `Minecraft-HRL-Agent` should consume `MC_Tech_Tree/training_config.json` from the shared workspace.

The editor also runs inline validation in the properties panel and edit modal,
highlighting errors in real time as you edit.
