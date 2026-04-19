"""
skill_stats.py

Tracks per-skill execution statistics across DQN training.

For every call to env.step(action) the logger records:
  calls          — total times the DQN selected this skill
  successes      — times the skill executed successfully (reward > base floor)
  failures       — times execution returned success=False
  precond_blocks — times precondition check blocked execution
                   (bridge returns reward=-0.5, message "Preconditions not met")
  total_reward   — cumulative reward (base + context_bonus)
  total_base     — cumulative base tech-tree reward
  total_context  — cumulative context-shaper bonus
  last_message   — most recent message string from the bridge

The file is written atomically to logs/skill_stats.json after every step.
A ranked summary table is printed to stdout every `print_every` steps.

Usage (minecraft_env.py does this automatically):
    logger = SkillStatsLogger()
    logger.register_skills(env.get_skill_info())
    ...
    logger.record(action, reward, base_reward, context_bonus, info)
    ...
    logger.print_summary()       # also called automatically every print_every steps
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List


class SkillStatsLogger:
    """Per-skill execution tracker with JSON persistence and console summaries."""

    def __init__(
        self,
        log_path: str = "logs/skill_stats.json",
        print_every: int = 50,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.print_every = print_every

        # skill_name (str) → stats dict
        self._stats: dict = defaultdict(lambda: {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "precond_blocks": 0,
            "total_reward": 0.0,
            "total_base": 0.0,
            "total_context": 0.0,
            "last_message": "",
            "first_success_step": None,  # global step number of first successful execution
        })

        self._total_steps: int = 0
        self._skill_names: dict = {}   # action_id (int) → skill_name (str)

    # ── Registration ─────────────────────────────────────────────────────────

    def register_skills(self, skill_info: List[dict]):
        """
        Register all skills from the list returned by env.get_skill_info().
        Each entry must have at minimum an 'id' (int) and 'name' (str) key.
        """
        for s in skill_info:
            self._skill_names[int(s["id"])] = s["name"]

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        action: int,
        reward: float,
        base_reward: float,
        context_bonus: float,
        info: dict,
    ) -> None:
        """
        Record the outcome of one env.step() call.

        Args:
            action:        Skill ID that was passed to env.step().
            reward:        Total reward returned by env.step() (base + context).
            base_reward:   Raw tech-tree reward from the Mineflayer bridge.
            context_bonus: Additive bonus from ContextRewardShaper (0 if disabled).
            info:          Info dict from env.step().  Expected keys:
                               'skill_success' (bool)
                               'skill_message' (str)
        """
        self._total_steps += 1
        name = self._skill_names.get(int(action), f"skill_{action}")
        s = self._stats[name]

        message = info.get("skill_message", "")

        s["calls"]         += 1
        s["total_reward"]  += reward
        s["total_base"]    += base_reward
        s["total_context"] += context_bonus
        s["last_message"]   = message

        # Classify the outcome:
        #   precond_block — bridge returned "Preconditions not met" (reward = -0.5)
        #   success       — skill executed and returned success=True
        #   failure       — skill executed but returned success=False
        if "Preconditions not met" in message:
            s["precond_blocks"] += 1
        elif info.get("skill_success", False):
            s["successes"] += 1
            if s["first_success_step"] is None:
                s["first_success_step"] = self._total_steps
        else:
            s["failures"] += 1

        self._flush()

        if self._total_steps % self.print_every == 0:
            self.print_summary()

    # ── Output ────────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Atomically write current stats to the JSON log file."""
        out = {
            "total_steps": self._total_steps,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "skills": {},
        }

        for name, s in sorted(self._stats.items()):
            calls = s["calls"]
            out["skills"][name] = {
                "calls":               s["calls"],
                "successes":           s["successes"],
                "failures":            s["failures"],
                "precond_blocks":      s["precond_blocks"],
                "total_reward":        round(s["total_reward"],  4),
                "total_base":          round(s["total_base"],    4),
                "total_context":       round(s["total_context"], 4),
                "success_rate":        round(s["successes"] / calls, 3) if calls else 0.0,
                "avg_reward":          round(s["total_reward"] / calls, 4) if calls else 0.0,
                "first_success_step":  s["first_success_step"],
                "last_message":        s["last_message"],
            }

        tmp_path = str(self.log_path) + ".tmp"

        def _try_write() -> bool:
            """Attempt one write cycle. Returns True on success."""
            try:
                with open(tmp_path, "w") as f:
                    json.dump(out, f, indent=2)
                try:
                    os.replace(tmp_path, str(self.log_path))
                    return True
                except PermissionError:
                    # Atomic rename denied (file locked by Explorer/AV).
                    # Clean up tmp and try a direct overwrite instead.
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    with open(str(self.log_path), "w") as f:
                        json.dump(out, f, indent=2)
                    return True
            except PermissionError:
                return False
            except OSError:
                return False

        if not _try_write():
            # First attempt failed — wait briefly and retry once.
            time.sleep(0.4)
            if not _try_write():
                # Still locked — skip this flush silently.
                # Stats remain correct in memory and will appear in the
                # next console summary; training is never interrupted.
                pass

    def print_summary(self) -> None:
        """Print a ranked summary table to stdout, sorted by total reward."""
        rows = []
        for name, s in self._stats.items():
            calls = s["calls"]
            if calls == 0:
                continue
            rows.append({
                "name":      name,
                "calls":     calls,
                "ok":        s["successes"],
                "fail":      s["failures"],
                "blk":       s["precond_blocks"],
                "ok_pct":    s["successes"] / calls * 100,
                "avg_r":     s["total_reward"] / calls,
                "total_r":   s["total_reward"],
                "first_win": s["first_success_step"],
            })

        rows.sort(key=lambda r: r["total_r"], reverse=True)

        w = 112
        print(f"\n{'='*w}")
        print(
            f"[SkillStats] Step {self._total_steps:,}  —  "
            f"per-skill breakdown (sorted by cumulative reward)"
        )
        print(
            f"{'Skill':<32} {'Calls':>6} {'OK':>5} {'Fail':>5} "
            f"{'Blk':>5} {'OK %':>6} {'Avg R':>7} {'Total R':>9} {'1st Win':>9}"
        )
        print(f"{'-'*w}")
        for r in rows:
            first_win = str(r["first_win"]) if r["first_win"] is not None else "—"
            print(
                f"{r['name']:<32} {r['calls']:>6} {r['ok']:>5} {r['fail']:>5} "
                f"{r['blk']:>5} {r['ok_pct']:>5.1f}% {r['avg_r']:>7.3f} {r['total_r']:>9.2f} {first_win:>9}"
            )
        print(f"{'='*w}\n")

    def summary_dict(self) -> dict:
        """Return the current stats as a plain dict (useful for unit tests / TensorBoard)."""
        out = {}
        for name, s in self._stats.items():
            calls = s["calls"]
            out[name] = {
                **s,
                "success_rate": round(s["successes"] / calls, 3) if calls else 0.0,
                "avg_reward":   round(s["total_reward"] / calls, 4) if calls else 0.0,
            }
        return out
