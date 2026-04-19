"""
analyze_run.py
==============
Reads TensorBoard event files from a training run and prints a clean summary.
Also reads logs/skill_stats.json if it exists.

Usage:
    python analyze_run.py              # auto-detects latest DQN run
    python analyze_run.py logs/DQN_2   # specific run directory
"""

import sys
import os
import json
from pathlib import Path


def read_tensorboard(log_dir: str) -> dict:
    """Read all scalar values from a TensorBoard event file."""
    try:
        from tensorflow.core.util import event_pb2
        from tensorflow.python.lib.io import tf_record
    except ImportError:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            acc = EventAccumulator(log_dir)
            acc.Reload()
            data = {}
            for tag in acc.Tags().get("scalars", []):
                data[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
            return data
        except ImportError:
            print("[!] Neither tensorflow nor tensorboard python package found.")
            print("    pip install tensorboard")
            return {}

    events_dir = Path(log_dir)
    event_files = list(events_dir.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"[!] No event files found in {log_dir}")
        return {}

    data = {}
    for ef in event_files:
        for record in tf_record.tf_record_iterator(str(ef)):
            event = event_pb2.Event.FromString(record)
            for v in event.summary.value:
                if v.HasField("simple_value"):
                    data.setdefault(v.tag, []).append((event.step, v.simple_value))
    return data


def print_training_summary(data: dict, total_steps: int):
    print("\n" + "=" * 60)
    print("TRAINING METRICS SUMMARY")
    print("=" * 60)

    # Key metrics to display
    key_metrics = [
        ("rollout/ep_rew_mean",       "Episode reward (mean)"),
        ("rollout/ep_len_mean",       "Episode length (mean)"),
        ("rollout/exploration_rate",  "Exploration rate (epsilon)"),
        ("train/loss",                "Q-network loss"),
        ("train/n_updates",           "Q-network updates"),
        ("custom/rolling_reward",     "Rolling reward (last 100 steps)"),
        ("custom/rolling_success",    "Rolling success rate"),
        ("custom/inventory_count",    "Inventory item count"),
    ]

    for tag, label in key_metrics:
        if tag in data:
            values = data[tag]
            if values:
                latest_step, latest_val = values[-1]
                first_step,  first_val  = values[0]
                print(f"\n  {label}")
                print(f"    First : {first_val:.4f}  (step {first_step:,})")
                print(f"    Latest: {latest_val:.4f}  (step {latest_step:,})")
                if len(values) > 1:
                    trend = "UP" if latest_val > first_val else "DOWN" if latest_val < first_val else "FLAT"
                    print(f"    Trend : {trend}  ({latest_val - first_val:+.4f})")

    print(f"\n  Total steps logged: {total_steps:,}")


def print_skill_stats(stats_path: str):
    if not os.path.exists(stats_path):
        print(f"\n[!] No skill_stats.json found at {stats_path}")
        print("    Per-skill tracking will appear here once env.py logging is active.")
        return

    with open(stats_path) as f:
        data = json.load(f)

    skills = data.get("skills", {})
    total_steps = data.get("total_steps", 0)

    print("\n" + "=" * 60)
    print(f"PER-SKILL BREAKDOWN  (total steps: {total_steps:,})")
    print("=" * 60)

    # Sort by total_reward descending
    rows = []
    for name, s in skills.items():
        if s["calls"] == 0:
            continue
        rows.append({
            "name":    name,
            "calls":   s["calls"],
            "ok":      s["successes"],
            "fail":    s["failures"],
            "blk":     s["precond_blocks"],
            "ok_pct":  s["successes"] / s["calls"] * 100,
            "avg_r":   s["total_reward"] / s["calls"],
            "total_r": s["total_reward"],
        })
    rows.sort(key=lambda r: r["total_r"], reverse=True)

    header = f"{'Skill':<32} {'Calls':>6} {'OK':>5} {'Fail':>5} {'Blk':>5} {'OK%':>6} {'AvgR':>7} {'TotalR':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<32} {r['calls']:>6} {r['ok']:>5} {r['fail']:>5} "
            f"{r['blk']:>5} {r['ok_pct']:>5.1f}% {r['avg_r']:>7.3f} {r['total_r']:>9.2f}"
        )

    # Highlight milestones
    succeeded = {r["name"] for r in rows if r["ok"] > 0}
    milestone_chain = [
        "harvest_wood", "craft_planks", "craft_sticks", "craft_crafting_table",
        "place_crafting_table", "craft_wooden_pickaxe", "mine_stone",
        "craft_stone_pickaxe", "mine_iron", "craft_furnace", "smelt_iron",
        "craft_iron_pickaxe", "craft_iron_helmet", "craft_iron_chestplate",
        "craft_iron_leggings", "craft_iron_boots", "dig_to_diamond_level",
        "mine_diamond", "craft_diamond_pickaxe", "craft_diamond_helmet",
        "craft_diamond_chestplate", "craft_diamond_leggings", "craft_diamond_boots",
    ]
    print("\n  TECH TREE PROGRESS:")
    for skill in milestone_chain:
        s = skills.get(skill, {})
        first_step = s.get("first_success_step")
        if first_step is not None:
            print(f"    [YES] {skill:<36} first success at step {first_step:,}")
        else:
            print(f"    [ NO] {skill}")


def main():
    base = Path(__file__).parent

    # Find log dir
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        logs_root = base / "logs"
        runs = sorted(
            [d for d in logs_root.iterdir() if d.is_dir() and d.name.startswith("DQN")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not runs:
            print("[!] No DQN run directories found in logs/")
            sys.exit(1)
        log_dir = str(runs[0])
        print(f"[analyze_run] Auto-detected latest run: {runs[0].name}")

    # Read TensorBoard
    data = read_tensorboard(log_dir)
    total_steps = max(
        (v[-1][0] for v in data.values() if v),
        default=0,
    )
    print_training_summary(data, total_steps)

    # Read skill stats
    stats_path = str(base / "logs" / "skill_stats.json")
    print_skill_stats(stats_path)


if __name__ == "__main__":
    main()
