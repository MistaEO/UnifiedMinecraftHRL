# Unified Minecraft HRL Runbook

This document is the workspace-level runbook for the current normalized setup.

It explains how these folders work together:

- `Minecraft-HRL-Agent/`: live Minecraft bot, online RL, offline DT evaluation, path-based reward integration
- `RL_Minecraft/`: reasoning-path evaluator aligned to the canonical schema/vocabulary
- `MC_Tech_Tree/`: canonical tech tree and reward-table source

This file is intended to replace ad hoc instructions spread across the other READMEs.

## Current State

The current normalized pipeline uses:

- a canonical tech tree in `MC_Tech_Tree/tech_tree.json`
- a flat training artifact in `MC_Tech_Tree/training_config.json`
- a unified reasoning-path evaluator in `RL_Minecraft/`
- a live RL environment in `Minecraft-HRL-Agent/` that can use:
  - Mineflayer/base skill reward
  - tech-tree reward and shaping
  - context reward
  - path-based reward against ground-truth reasoning paths
  - task embedding in the observation when a reference dataset is used

Important CLI note:

- the current `main.py` uses `--mode pure_rl --policy DQN` or `--policy PPO`
- older examples like `--mode dqn` are stale

## Folder Roles

### `MC_Tech_Tree/`

This is the tech-tree source and reward definition.

Canonical files:

- `tech_tree.json`: committed source of truth
- `tech_tree_seed.js`: editor boot data mirrored from `tech_tree.json`
- `training_config.json`: reward-table artifact consumed by the agent

Main tooling:

- `tech_tree.py`: validation, export, and runtime reward manager
- `tech_tree_editor.html`: visual editor

### `RL_Minecraft/`

This is the reasoning-path evaluator.

It now uses the same canonical schema and vocabulary as `Minecraft-HRL-Agent`.

It evaluates predicted paths against ground-truth `reasoning_path` values with metrics like:

- exact match
- step F1
- normalized edit distance
- task validity

### `Minecraft-HRL-Agent/`

This is the live bot + RL training stack.

It contains:

- the Mineflayer bot (`mineflayer/`)
- the Minecraft server folder (`minecraft-server/`)
- the Python RL backend (`python/`)
- the dataset of ground-truth reasoning paths (`data/processed/dataset_final.json`)

## Reward Pipeline

The live environment can now combine several reward sources.

### 1. Base Mineflayer reward

This comes from live skill execution outcomes in the JS bot.

Examples:

- skill success/failure
- inventory gain
- repeat-action penalties

### 2. Tech-tree reward

This comes from `MC_Tech_Tree/training_config.json` through the normalized Python reward manager.

It has two parts:

- tech-tree terminal/unlock reward
- dense tech-tree shaping reward

This is enabled by default.

### 3. Context reward

This rewards biome/structure-aware behavior.

Examples:

- structure shortcuts
- biome-adaptive choices
- penalties for bad context-specific decisions

This is enabled unless `--no-context-reward` is used.

### 4. Path reward

This is evaluator-derived reward against a ground-truth reasoning path.

It uses metrics such as:

- exact match
- step F1
- task validity
- prefix match
- edit-distance penalty

This only activates when you provide `--reference-dataset`.

### Effective reward

At a high level, live reward is now:

```text
reward = base_reward
       + tech_tree_reward
       + tech_tree_shaping
       + context_bonus
       + path_bonus
```

## Task Embedding

When `--reference-dataset` is used, each episode samples a reference task/path from the dataset.

That task is exposed to the policy through a one-hot `task_vec` in the observation.

This means online RL can now be task-conditioned instead of relying only on generic world state.

## Offline Evaluation vs Online Training

### Offline evaluation

Offline evaluation means:

- no live Minecraft interaction is required
- predicted or executed paths are compared to dataset ground truth afterward
- the evaluator produces numeric outputs for analysis and benchmarking

Use this for:

- DT outputs
- planner outputs
- saved path predictions
- benchmark runs

### Online RL training

Online RL means:

- the bot acts in live Minecraft
- DQN or PPO receives reward from the environment
- policy updates happen from live interaction

With the current setup, online RL can include evaluator-derived reward too, as long as `--reference-dataset` is enabled.

## One-Time Setup

All commands below assume PowerShell on Windows.

Workspace root:

```powershell
cd "C:\Users\gavin\OneDrive\Desktop\Minecraft RL Sandbox"
```

### 1. Install Node dependencies

```powershell
cd ".\Minecraft-HRL-Agent\mineflayer"
npm install
```

### 2. Install Python dependencies

```powershell
cd "..\python"
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Validate the tech tree

From the workspace root:

```powershell
cd "..\.."
python ".\MC_Tech_Tree\tech_tree.py"
```

If this passes, the local tech tree is consistent and the reward-table artifact is in sync.

## Minecraft Server Configuration

Before starting the stack, check:

`Minecraft-HRL-Agent/minecraft-server/server.properties`

Make sure these values are set:

- `allow-flight=true`
- `online-mode=false`
- `enable-rcon=true`
- `rcon.password=hrltraining`
- `difficulty=peaceful`
- `spawn-monsters=false`

Important:

- the current file previously had `allow-flight=false`
- change it to `allow-flight=true` or the bot may get kicked by anti-cheat/pathfinder behavior

## Starting The Full Stack

Use 3 terminals.

Order matters:

1. Minecraft server
2. Mineflayer bot
3. Python training or evaluation

### Terminal 1: Minecraft server

```powershell
cd "C:\Users\gavin\OneDrive\Desktop\Minecraft RL Sandbox\Minecraft-HRL-Agent\minecraft-server"
java -Xmx4G -Xms4G -jar paper.jar nogui
```

### Terminal 2: Mineflayer bot

```powershell
cd "C:\Users\gavin\OneDrive\Desktop\Minecraft RL Sandbox\Minecraft-HRL-Agent\mineflayer"
npm start
```

Optional logging:

```powershell
npm start 2>&1 | Tee-Object training.log
```

Ports used by default:

- Minecraft server: `25565`
- bot bridge WebSocket: `8765`
- Prismarine viewer: `3007`

### Terminal 3: Python backend

```powershell
cd "C:\Users\gavin\OneDrive\Desktop\Minecraft RL Sandbox\Minecraft-HRL-Agent\python"
.\venv\Scripts\Activate.ps1
```

## Monitoring

### Prismarine viewer

Open:

- `http://localhost:3007`

### TensorBoard

From `Minecraft-HRL-Agent/python`:

```powershell
.\venv\Scripts\Activate.ps1
tensorboard --logdir .\logs
```

Open:

- `http://localhost:6006`

Useful live metrics now include:

- `custom/base_reward`
- `custom/tech_tree_reward`
- `custom/tech_tree_shaping`
- `custom/tech_tree_total`
- `custom/context_bonus`
- `custom/path_bonus`
- `custom/path_reward_total`
- `custom/task_id`

## Running The Agent

All commands below are run from:

```powershell
cd "C:\Users\gavin\OneDrive\Desktop\Minecraft RL Sandbox\Minecraft-HRL-Agent\python"
.\venv\Scripts\Activate.ps1
```

### 1. Baseline Old-Style DQN

This is the closest to the earlier online RL setup.

Characteristics:

- DQN
- live Minecraft
- env-aware observation
- tech-tree reward enabled
- no task/path conditioning

Command:

```powershell
python main.py --mode pure_rl --policy DQN --timesteps 200000 --env-aware --save-freq 10000 --checkpoint "checkpoints\dqn_baseline_oldstyle"
```

### 2. Full Normalized DQN

This is the most complete current run.

Characteristics:

- DQN
- env-aware observation
- local normalized tech tree
- context reward enabled
- task embedding enabled
- path reward enabled

Command:

```powershell
python main.py --mode pure_rl --policy DQN --timesteps 200000 --env-aware --reference-dataset "..\data\processed\dataset_final.json" --tech-tree "..\..\MC_Tech_Tree\training_config.json" --save-freq 10000 --checkpoint "checkpoints\dqn_full_normalized"
```

### 3. Full Normalized PPO

Same as above, but with PPO.

Command:

```powershell
python main.py --mode pure_rl --policy PPO --timesteps 200000 --env-aware --reference-dataset "..\data\processed\dataset_final.json" --tech-tree "..\..\MC_Tech_Tree\training_config.json" --save-freq 10000 --checkpoint "checkpoints\ppo_full_normalized"
```

### 4. Context Ablation

Run these as two separate commands on Windows.

#### Condition A: env-aware + context reward

```powershell
python main.py --mode pure_rl --policy DQN --timesteps 200000 --env-aware --reference-dataset "..\data\processed\dataset_final.json" --tech-tree "..\..\MC_Tech_Tree\training_config.json" --save-freq 10000 --checkpoint "checkpoints\dqn_envaware_contextreward"
```

#### Condition B: env-blind + no context reward

```powershell
python main.py --mode pure_rl --policy DQN --timesteps 200000 --no-context-reward --reference-dataset "..\data\processed\dataset_final.json" --tech-tree "..\..\MC_Tech_Tree\training_config.json" --save-freq 10000 --checkpoint "checkpoints\dqn_envblind_nocontext"
```

### 5. Tech-Tree Ablation

If you want to remove the normalized Python-side tech-tree reward/shaping:

```powershell
python main.py --mode pure_rl --policy DQN --timesteps 200000 --env-aware --no-tech-tree-reward --reference-dataset "..\data\processed\dataset_final.json" --save-freq 10000 --checkpoint "checkpoints\dqn_no_techtree"
```

## Evaluation

### Evaluate a saved checkpoint

Example:

```powershell
python main.py --eval --mode pure_rl --policy DQN --env-aware --reference-dataset "..\data\processed\dataset_final.json" --checkpoint "checkpoints\dqn_full_normalized"
```

### Offline reasoning-path evaluation

This does not require the live Minecraft stack.

Example using a benchmark predictions file:

```powershell
python evaluate_reasoning_paths.py --predictions "..\data\benchmark_results\results.jsonl" --model "llama3.2:3b" --condition "with_context"
```

Notes:

- this scores explicit predicted paths against dataset ground truth
- natural-language outputs that are not canonical skill tokens will score poorly unless converted first

## Decision Transformer / Offline RL

The local tech tree is now used by default for DT-related commands too.

Train DT:

```powershell
python train_dt.py --dataset "..\data\processed\dataset_final.json"
```

Evaluate DT:

```powershell
python eval_dt.py --checkpoint "checkpoints\dt_best.pt" --dataset "..\data\processed\dataset_final.json"
```

Or run the unified offline evaluator directly from a DT checkpoint:

```powershell
python evaluate_reasoning_paths.py --dt-checkpoint "checkpoints\dt_best.pt"
```

## Editing The Tech Tree

### View or edit in browser

Open:

- `MC_Tech_Tree/tech_tree_editor.html`

The editor now boots from the committed seed file:

- `MC_Tech_Tree/tech_tree_seed.js`

### Canonical source contract

These files must stay aligned:

- `tech_tree.json`
- `tech_tree_seed.js`
- `training_config.json`

Practical rule:

1. edit/import the tree in the editor
2. export `tech_tree.json`
3. export `training_config.json`
4. keep `tech_tree_seed.js` aligned with the same tree state

If you only change `training_config.json` but not the seed/editor source, the browser editor and training code can diverge again.

### Validate after changes

```powershell
python ".\MC_Tech_Tree\tech_tree.py"
```

## Unified Schema / Vocabulary

The current local setup is unified around the canonical `Minecraft-HRL-Agent` configuration.

At the time of writing, the local verification reports:

- `12` tasks
- `47` skills

`RL_Minecraft` has been updated to match this vocabulary.

Verification command:

```powershell
python ".\Minecraft-HRL-Agent\python\verify_vocab_unification.py"
```

## Common Gotchas

### 1. Server starts but bot gets kicked

Check:

- `allow-flight=true`

### 2. Bot starts before server

The bot may fail to connect or crash early.

Always start:

1. server
2. bot
3. Python

### 3. Python cannot connect to bot

Check:

- bot is running
- bridge port is `8765`
- Python `--host` / `--port` match the bot bridge

### 4. TensorBoard shows nothing useful

Make sure training is actually running and writing to the `logs/` directory.

### 5. Offline evaluator scores all zeros

This usually means the prediction file does not contain canonical skill-token paths.

### 6. Tech tree edits do not seem to affect training

Make sure the local file actually being used is:

- `MC_Tech_Tree/training_config.json`

and re-run training after updating it.

## Minimal Recommended Workflow

If you only want the shortest path to a meaningful run:

1. Set `allow-flight=true` in `server.properties`
2. Start the Minecraft server
3. Start the Mineflayer bot
4. Start TensorBoard
5. Run:

```powershell
python main.py --mode pure_rl --policy DQN --timesteps 200000 --env-aware --reference-dataset "..\data\processed\dataset_final.json" --tech-tree "..\..\MC_Tech_Tree\training_config.json" --save-freq 10000 --checkpoint "checkpoints\dqn_full_normalized"
```

This gives you the current normalized setup with:

- local tech tree
- task embedding
- path reward
- context reward
- online DQN training
