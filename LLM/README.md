# LLM/ — Language Model Skill Conditioner

This module replaces the Decision Transformer as the high-level planner.
At every RL step, a Claude LLM reads the current game state and suggests
the next skill. That suggestion is converted to a dense embedding and fed
into the PPO / DQN policy as part of the observation.

---

## How skill selection works

There are **two levels** of decision-making, not one.

```
Game state (biome, inventory, health, …)
        │
        ▼
  Claude Haiku (LLM)          ← called every step via Anthropic API
        │  outputs e.g. "mine_iron_ore"
        ▼
  Validator                   ← rejects impossible skills (no pickaxe → can't mine)
        │  confirmed skill name
        ▼
  Embedding lookup             ← 768-dim vector from all-mpnet-base-v2
        │
        ├──► appended to 41-dim state encoding  =  809-dim observation
        │
        ▼
  PPO / DQN policy             ← takes the 809-dim obs, outputs action index
        │
        ▼
  Bot executes the chosen skill
```

The LLM does **not** directly control the bot.
It provides a *suggestion* — encoded as a direction in embedding space —
that the RL policy learns to follow, adapt, or override based on what
actually earns reward. Over training, the policy learns which LLM
suggestions are trustworthy and in which situations to deviate.

---

## Files

| File | Purpose |
|---|---|
| `prompt.py` | `SYSTEM_PROMPT` — priority-ordered rules the LLM follows (P0 safety → P3 hygiene) |
| `environment.py` | `SKILLS` list (47 skills from `config.py`), `SKILL_EMBEDDINGS` map, `get_skill_embedding()` |
| `task_embeddings.py` | **Run once.** Computes 768-dim embeddings for all 47 skills and writes them into `environment.py` |
| `agent.py` | Called every step: formats state → calls Claude API → parses skill → returns `(skill_name, np.ndarray)` |
| `validator.py` | Checks prerequisites for each skill (inventory, structures). Filters the skill list before the LLM sees it, and rejects any invalid output |

---

## Setup

### 1. Install Python dependencies

```bash
pip install transformers accelerate torch sentence-transformers numpy websocket-client gymnasium stable-baselines3
```

> **GPU strongly recommended.** Qwen2.5-7B runs inference in ~0.2s on a GPU;
> on CPU it is ~5s per step which will make training very slow.
> If you only have CPU, swap to the 1.5B model (see below).

### 2. (Optional) Override the Qwen model

The default is `Qwen/Qwen2.5-7B-Instruct`. To use a smaller model:

```bash
# Linux / macOS
export QWEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Windows (PowerShell)
$env:QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

| Model | VRAM / RAM | Speed (GPU) | Notes |
|---|---|---|---|
| `Qwen2.5-7B-Instruct` | ~16 GB | ~0.2s/step | Default, best quality |
| `Qwen2.5-3B-Instruct` | ~8 GB  | ~0.1s/step | Good balance |
| `Qwen2.5-1.5B-Instruct` | ~4 GB | ~0.05s/step | CPU-friendly |

The model is downloaded automatically from HuggingFace on first run
(~15 GB for 7B). No API key required.

### 3. Generate skill embeddings (once only)

This downloads `all-mpnet-base-v2` (~420 MB, cached after first run),
embeds all 47 skill names, and writes the vectors into `environment.py`.

```bash
cd UnifiedMinecraftHRL/LLM
python task_embeddings.py
```

You will see:
```
Loading all-mpnet-base-v2...
Embedding 47 skills...
Written 47 embeddings (768-dim) to environment.py
```

Only re-run this if `SKILL_VOCAB` in `config.py` changes.

---

## Running the full system

All four steps must be running at the same time (four terminals).

### Terminal 1 — Minecraft server
```bash
cd UnifiedMinecraftHRL/Minecraft-HRL-Agent/minecraft-server
java -Xmx2G -jar server.jar nogui
```
Wait until you see `Done!` before continuing.

### Terminal 2 — Mineflayer bot
```bash
cd UnifiedMinecraftHRL/Minecraft-HRL-Agent/mineflayer
npm install
node src/index.js
```
Wait until you see `[Bot] Ready for Python agent connection!`

### Terminal 3 — Generate embeddings (first time only)
```bash
cd UnifiedMinecraftHRL/LLM
python task_embeddings.py
```

### Terminal 4 — Train
```bash
cd UnifiedMinecraftHRL/Minecraft-HRL-Agent/python
python main.py --mode hybrid --policy PPO --timesteps 100000
```

Optional flags:
```
--policy DQN                  use DQN instead of PPO
--mode pure_rl                disable novelty exploration
--env-aware                   include biome/structure in obs (already in LLM embedding)
--no-context-reward           disable structure/biome bonus rewards
--reference-dataset ../../Minecraft-HRL-Agent/data/processed/dataset_final.json
                              enable incremental path reward against ground truth
--tech-tree ../../MC_Tech_Tree/training_config.json
                              path to reward table (default auto-detected)
--checkpoint ./checkpoints/run_XYZ/final
                              resume from saved checkpoint
```

---

## Embedding model choice

**`all-mpnet-base-v2`** (768-dim, ~420 MB, free via `sentence-transformers`)

Alternatives considered:

| Model | Dims | Why not chosen |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fastest but weakest semantic separation on short domain strings |
| `all-MiniLM-L12-v2` | 384 | Better than L6, still lower quality ceiling |
| `all-mpnet-base-v2` | 768 | **Chosen** — highest MTEB semantic similarity scores among free SBERT models |

The extra 384 dims matter here because the RL policy needs to distinguish
semantically close skills like `mine_iron_ore` vs `mine_gold_ore` vs
`dig_to_diamond_level`. Collapsing those together in a smaller space
would make the embedding less useful as a conditioning signal.

---

## LLM model choice

**`Qwen/Qwen2.5-7B-Instruct`** — configured in `agent.py` via `MODEL_NAME`

Chosen because:
- The project already benchmarks Qwen-7B (see `data/benchmark_results/`)
- Runs fully locally — no API key, no per-call cost, no latency variance
- 7B is large enough to reliably follow the structured P0–P3 rules in `prompt.py`
- `do_sample=False` (greedy decoding) makes output deterministic and fast

Override via environment variable without touching code:
```bash
export QWEN_MODEL=Qwen/Qwen2.5-3B-Instruct   # smaller / faster
```

---

## Validator — why it exists

The LLM can hallucinate impossible actions (e.g. `mine_iron_ore` when the
bot has no pickaxe). The validator prevents two problems:

1. **The LLM never sees impossible skills** — `format_state_context()`
   calls `valid_skills(state)` and only lists currently executable skills
   in the prompt, so the LLM is unlikely to suggest one.

2. **Hard rejection** — if the LLM still outputs an impossible skill,
   `validate(skill, state)` rejects it and a safe fallback is chosen
   (`explore_cave` → `navigate_to_structure` → `harvest_wood` → first
   valid skill).

Prerequisite rules cover: pickaxe tier requirements, crafting table /
furnace in inventory, ingredient counts, nearby structure requirements,
food in inventory, and building materials.

---

## State dict format

`agent.py` expects a dict with these keys (all optional — missing keys
are silently omitted from the prompt):

```python
{
    "task":            str,          # e.g. "obtain_iron_pickaxe"
    "biome":           str,          # e.g. "forest"
    "nearby_structures": list[str],  # e.g. ["village", "blacksmith"]
    "y_level":         int,          # e.g. -12
    "health":          float,        # 0–20
    "hunger":          float,        # 0–20
    "time_of_day":     str,          # "day" or "night"
    "inventory":       dict[str,int],# e.g. {"oak_log": 5, "stone_pickaxe": 1}
    "equipped_tool":   str,          # e.g. "stone_pickaxe"
    "tool_durability": float,        # 0.0–1.0
}
```

`env.py` builds this dict automatically from the bridge state on every
step — you do not need to construct it manually during training.
