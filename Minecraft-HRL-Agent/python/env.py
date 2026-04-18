"""
MinecraftHRLEnv
===============
Gymnasium environment bridging the Mineflayer bot (WebSocket) to the Python
PPO / DQN training loop, with the LLM called every step as a skill conditioner.

Observation (809-dim float32):
    [0 : 41]   StateEncoder output — biome one-hot, structures multi-hot,
                task one-hot, y-level normalised
    [41 : 809] all-mpnet-base-v2 embedding of the LLM-suggested skill

Action:
    Discrete(N) — bot skill IDs from skillManager.js, learnt on connect()

Reward:
    bridge reward  (from skillManager._calculateInventoryReward)
  + tech-tree reward (rtg_utils.skill_reward)          [optional]
  + context reward  (ContextRewardShaper.get_bonus)    [optional]
  + incremental path reward (PathRewardScorer)         [optional, needs dataset]

WebSocket protocol (bridge.js):
    { type: 'ping' }                    → { type: 'pong' }
    { type: 'get_action_space' }        → { type: 'action_space', n, skills }
    { type: 'reset' }                   → { type: 'reset_result', state, info }
    { type: 'step', action: int }       → { type: 'step_result', state, reward,
                                             done, truncated, info }
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import websocket  # websocket-client

# ── Path setup ────────────────────────────────────────────────────────────────
_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
_LLM_DIR    = os.path.normpath(os.path.join(_PYTHON_DIR, "..", "..", "LLM"))
_DATA_DIR   = os.path.normpath(os.path.join(_PYTHON_DIR, "..", "data"))

for _p in (_PYTHON_DIR, _LLM_DIR, _DATA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Local imports ─────────────────────────────────────────────────────────────
from models.state_encoder import StateEncoder, STATE_DIM, BIOME_IDX
from models.rtg_utils import load_reward_table, skill_reward
from reward.context_reward import ContextRewardShaper

from agent import get_skill_and_embedding          # LLM/agent.py
from environment import EMBEDDING_DIM               # LLM/environment.py
from config import BIOMES, TASKS                    # data/config.py

# ── Constants ─────────────────────────────────────────────────────────────────
OBS_DIM = STATE_DIM + EMBEDDING_DIM  # 41 + 768 = 809

# Maps LLM skill names (47-vocab) → bot skill names (skillManager.js).
# Skills with a direct name match need no entry.
_LLM_TO_BOT: dict[str, str] = {
    "craft_planks_and_sticks":           "craft_planks",
    "craft_torch":                       "idle",
    "mine_coal":                         "mine_stone",
    "mine_iron_ore":                     "mine_iron",
    "craft_iron_armor_set":              "craft_iron_helmet",
    "mine_diamonds":                     "mine_diamond",
    "dig_to_gold_level":                 "dig_to_diamond_level",
    "mine_gold_ore":                     "mine_iron",
    "smelt_gold":                        "smelt_iron",
    "search_for_animals":                "explore",
    "kill_animals_for_meat":             "explore",
    "cook_meat":                         "smelt_iron",
    "harvest_village_crops":             "explore",
    "harvest_melons_from_ground":        "explore",
    "harvest_sweet_berries_from_bushes": "explore",
    "milk_mooshroom_with_bowl":          "explore",
    "build_walls_and_roof":              "idle",
    "craft_and_place_door":              "idle",
    "place_torches":                     "idle",
    "go_to_village":                     "explore",
    "find_blacksmith":                   "explore",
    "loot_blacksmith_chest":             "explore",
    "navigate_to_mineshaft":             "explore",
    "search_mineshaft_chests":           "explore",
    "go_to_ruined_portal":               "explore",
    "loot_portal_chest":                 "explore",
    "go_to_desert_temple":               "explore",
    "avoid_tnt_trap":                    "idle",
    "go_to_jungle_temple":               "explore",
    "swim_to_shipwreck":                 "explore",
    "loot_supply_chest":                 "explore",
    "go_to_igloo":                       "explore",
    "navigate_to_structure":             "explore",
    "explore_cave":                      "explore",
    "combat_mob":                        "explore",
}


# ── Environment ───────────────────────────────────────────────────────────────

class MinecraftHRLEnv(gym.Env):
    """
    Gymnasium environment for Minecraft HRL.

    Call connect(timeout) once before reset() / step().
    Observation is 809-dim: StateEncoder(41) + LLM skill embedding(768).
    Action is Discrete(N) where N = number of bot skills from skillManager.js.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        render_mode: str | None = None,
        env_aware: bool = False,
        context_reward: bool = True,
        reference_dataset: str | None = None,
        tech_tree_path: str | None = None,
        tech_tree_reward: bool = True,
        task: str | None = None,
        max_episode_steps: int = 1000,
    ):
        super().__init__()

        self.host = host
        self.port = port
        self.render_mode = render_mode
        self._fixed_task = task
        self.max_episode_steps = max_episode_steps

        # Spaces — action space is updated after connect() returns real skill count
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(1)  # placeholder

        # WebSocket
        self._ws: websocket.WebSocket | None = None
        self._connected = False

        # Bot skill registry (populated on connect)
        self._bot_skills: list[dict] = []
        self._bot_name_to_id: dict[str, int] = {}

        # Reward shaping
        self._context_shaper = ContextRewardShaper() if context_reward else None
        self._reward_table: dict = {}
        if tech_tree_reward and tech_tree_path:
            try:
                self._reward_table = load_reward_table(tech_tree_path)
            except Exception as e:
                print(f"[Env] Warning: could not load reward table: {e}")

        # Path reward (optional — needs reference dataset)
        self._path_scorer = None
        self._dataset_by_task: dict[str, list[dict]] = {}
        if reference_dataset:
            self._load_dataset(reference_dataset)

        # Episode state
        self._episode_task: str = task or TASKS[0]
        self._episode_step: int = 0
        self._executed_skills: list[str] = []
        self._reference_sample: dict | None = None
        self._prev_path_score: float = 0.0
        self._last_raw_state: dict = {}

    # ── Dataset loading ───────────────────────────────────────────────────────

    def _load_dataset(self, path: str) -> None:
        try:
            from reward.path_reward import PathRewardScorer
            with open(path) as f:
                samples = json.load(f)
            for s in samples:
                task = s.get("task", "")
                self._dataset_by_task.setdefault(task, []).append(s)
            self._path_scorer = PathRewardScorer()
            print(f"[Env] Loaded {len(samples)} reference samples from {path}")
        except Exception as e:
            print(f"[Env] Warning: could not load reference dataset: {e}")

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self, timeout: int = 30) -> bool:
        """
        Connect to the mineflayer WebSocket bridge.
        Fetches the action space and configures self.action_space.
        Returns True on success.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ws = websocket.WebSocket()
                ws.settimeout(10)
                ws.connect(f"ws://{self.host}:{self.port}")
                self._ws = ws

                resp = self._send({"type": "ping"})
                if resp.get("type") != "pong":
                    raise ConnectionError("Unexpected ping response")

                resp = self._send({"type": "get_action_space"})
                self._bot_skills    = resp["skills"]
                self._bot_name_to_id = {s["name"]: s["id"] for s in self._bot_skills}
                self.action_space   = spaces.Discrete(resp["n"])

                self._connected = True
                print(f"[Env] Connected — {resp['n']} bot skills available")
                return True

            except Exception as e:
                print(f"[Env] Connection attempt failed ({e}), retrying...")
                time.sleep(2)

        print("[Env] Could not connect within timeout")
        return False

    def _send(self, msg: dict) -> dict:
        """Send a JSON message over WebSocket and return the parsed response."""
        self._ws.send(json.dumps(msg))
        return json.loads(self._ws.recv())

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        assert self._connected, "Call connect() before reset()"

        # Sample task for this episode
        if self._fixed_task:
            self._episode_task = self._fixed_task
        else:
            self._episode_task = random.choice(TASKS)

        # Reset episode tracking
        self._episode_step    = 0
        self._executed_skills = []
        self._prev_path_score = 0.0

        if self._context_shaper:
            self._context_shaper.reset()

        # Sample a reference path for this task (if dataset loaded)
        self._reference_sample = None
        if self._dataset_by_task:
            pool = self._dataset_by_task.get(self._episode_task, [])
            if pool:
                self._reference_sample = random.choice(pool)

        resp = self._send({"type": "reset"})
        self._last_raw_state = resp["state"]

        obs  = self._build_obs(self._last_raw_state)
        info = {"task": self._episode_task}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._connected, "Call connect() before step()"

        resp = self._send({"type": "step", "action": int(action)})

        raw_state  = resp["state"]
        bot_reward = float(resp["reward"])
        terminated = bool(resp["done"])
        truncated  = bool(resp.get("truncated", False)) or (
            self._episode_step >= self.max_episode_steps
        )
        info = dict(resp.get("info", {}))

        self._last_raw_state = raw_state
        self._episode_step  += 1

        skill_name = info.get("skill_name", "")
        if skill_name:
            self._executed_skills.append(skill_name)

        reward = bot_reward
        reward += self._tech_tree_reward(skill_name)
        reward += self._context_reward(skill_name, raw_state)
        reward += self._path_reward()

        obs = self._build_obs(raw_state)

        info["task"] = self._episode_task
        info["step"] = self._episode_step

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            pos = self._last_raw_state.get("position", {})
            print(
                f"[Env] step={self._episode_step} | task={self._episode_task} | "
                f"pos=({pos.get('x',0)}, {pos.get('y',0)}, {pos.get('z',0)}) | "
                f"hp={self._last_raw_state.get('health', 0):.1f}"
            )

    def close(self) -> None:
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws      = None
            self._connected = False

    # ── Observation building ──────────────────────────────────────────────────

    def _build_obs(self, raw: dict) -> np.ndarray:
        """
        Build the 809-dim observation:
            [StateEncoder(41) | LLM skill embedding(768)]
        """
        # Build dict that both StateEncoder and agent.py accept
        adapted = self._adapt_state(raw)

        # 41-dim state encoding
        state_vec = StateEncoder.encode(adapted)          # (41,)

        # 768-dim LLM skill embedding (API call each step)
        _, embedding = get_skill_and_embedding(adapted)  # (768,)

        return np.concatenate([state_vec, embedding]).astype(np.float32)

    def _adapt_state(self, raw: dict) -> dict:
        """
        Convert the raw bridge.js _getState() dict into the unified state dict
        used by both StateEncoder.encode() and agent.py's format_state_context().

        StateEncoder needs:  biome, nearby_structures, task, y_level
        agent.py needs:      all of the above + health, hunger, time_of_day,
                             inventory, equipped_tool
        """
        pos = raw.get("position", {})
        return {
            "task":            self._episode_task,
            "biome":           self._normalize_biome(raw.get("biome", "plains")),
            "nearby_structures": raw.get("nearby_structures", ["none"]),
            "y_level":         pos.get("y", 64),
            "health":          raw.get("health", 20.0),
            "hunger":          raw.get("food", 20.0),
            "time_of_day":     "day" if raw.get("is_day", True) else "night",
            "inventory":       raw.get("inventory", {}),
            "equipped_tool":   raw.get("held_item") or "",
        }

    @staticmethod
    def _normalize_biome(biome: str) -> str:
        """Map a mineflayer biome string to the nearest BIOMES entry."""
        clean = biome.lower().replace("minecraft:", "")
        if clean in BIOME_IDX:
            return clean
        # Substring fallback (handles "birch_forest" → "forest", etc.)
        for b in BIOMES:
            if b in clean or clean in b:
                return b
        return "plains"

    # ── Reward shaping ────────────────────────────────────────────────────────

    def _tech_tree_reward(self, skill_name: str) -> float:
        if not self._reward_table or not skill_name:
            return 0.0
        return skill_reward(skill_name, self._reward_table)

    def _context_reward(self, skill_name: str, raw: dict) -> float:
        if not self._context_shaper or not skill_name:
            return 0.0
        obs = {
            "biome":            self._normalize_biome(raw.get("biome", "plains")),
            "nearby_structures": raw.get("nearby_structures", ["none"]),
        }
        return self._context_shaper.get_bonus(obs, skill_name)

    def _path_reward(self) -> float:
        if not self._path_scorer or not self._reference_sample or not self._executed_skills:
            return 0.0
        prev = self._executed_skills[:-1]
        curr = self._executed_skills
        return self._path_scorer.incremental_reward(
            self._reference_sample, prev, curr
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_skill_name(self, action: int) -> str:
        """Return the bot skill name for a given action index."""
        for s in self._bot_skills:
            if s["id"] == action:
                return s["name"]
        return f"skill_{action}"

    def llm_skill_to_bot_id(self, llm_skill: str) -> int:
        """Map an LLM skill name (47-vocab) to the closest bot skill ID."""
        if llm_skill in self._bot_name_to_id:
            return self._bot_name_to_id[llm_skill]
        bot_name = _LLM_TO_BOT.get(llm_skill, "explore")
        return self._bot_name_to_id.get(
            bot_name,
            self._bot_name_to_id.get("explore", 9),
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def make_minecraft_env(
    host: str = "localhost",
    port: int = 8765,
    flatten: bool = True,       # SB3 always needs flat obs; kept for API compat
    render_mode: str | None = None,
    env_aware: bool = False,
    context_reward: bool = True,
    reference_dataset: str | None = None,
    tech_tree_path: str | None = None,
    tech_tree_reward: bool = True,
    task: str | None = None,
) -> MinecraftHRLEnv:
    """
    Factory function matching the interface expected by main.py.

    Example
    -------
    env = make_minecraft_env(host="localhost", port=8765)
    env.connect(timeout=30)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """
    return MinecraftHRLEnv(
        host=host,
        port=port,
        render_mode=render_mode,
        env_aware=env_aware,
        context_reward=context_reward,
        reference_dataset=reference_dataset,
        tech_tree_path=tech_tree_path,
        tech_tree_reward=tech_tree_reward,
        task=task,
    )
