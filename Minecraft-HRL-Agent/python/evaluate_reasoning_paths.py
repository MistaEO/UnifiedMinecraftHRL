#!/usr/bin/env python3
"""
Offline path evaluation for canonical Minecraft reasoning-path outputs.

Supports:
  1. Evaluating an existing JSON/JSONL predictions file keyed by sample_id
  2. Generating Decision Transformer predictions on the fly and scoring them

Outputs per-sample metrics plus a scalar path reward built from the unified
schema shared with RL_Minecraft.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path



ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from reward.path_reward import PathRewardScorer
from RL_Minecraft.evaluator import ReasoningPathEvaluator, average_metrics
from tech_tree_utils import default_tech_tree_path


def parse_args():
    parser = argparse.ArgumentParser(description="Offline reasoning-path evaluation")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", help="Path to JSON/JSONL predictions keyed by sample_id")
    source.add_argument("--dt-checkpoint", help="Generate predictions from a DT checkpoint")

    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "processed" / "dataset_final.json"))
    parser.add_argument("--tech-tree", default=str(default_tech_tree_path()))
    parser.add_argument("--split", default="all", choices=["all", "test", "val"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--model", default=None, help="Optional filter for prediction records with a model field")
    parser.add_argument("--condition", default=None, help="Optional filter for prediction records with a condition field")
    parser.add_argument("--out", default=None, help="Write per-sample JSONL output")
    parser.add_argument("--summary-out", default=None, help="Write aggregate JSON summary")
    return parser.parse_args()


def load_json_or_jsonl(path: Path):
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def filter_split(samples, split: str, seed: int):
    if split == "all":
        return samples

    rng = random.Random(seed)
    by_task = defaultdict(list)
    for sample in samples:
        by_task[sample.get("task", "unknown")].append(sample)

    filtered = []
    for task_samples in by_task.values():
        task_samples = list(task_samples)
        rng.shuffle(task_samples)
        n = len(task_samples)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
        if split == "test":
            filtered.extend(task_samples[:n_test])
        else:
            filtered.extend(task_samples[n_test:n_test + n_val])
    return filtered


def resolve_device(device_name: str) -> str:
    import torch

    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_prediction_map(path: Path, model_filter: str | None, condition_filter: str | None):
    raw = load_json_or_jsonl(path)
    if isinstance(raw, dict):
        return {str(sample_id): prediction for sample_id, prediction in raw.items()}

    prediction_map = {}
    for record in raw:
        if model_filter is not None and record.get("model") != model_filter:
            continue
        if condition_filter is not None and record.get("condition") != condition_filter:
            continue

        sample_id = record.get("sample_id", record.get("id"))
        if sample_id is None:
            raise ValueError("Prediction record missing sample_id/id")

        prediction = None
        for field in ("prediction", "predicted", "parsed_steps", "reasoning_path"):
            value = record.get(field)
            if isinstance(value, list):
                prediction = value
                break
        if prediction is None:
            raise ValueError(f"Prediction record for sample {sample_id} is missing a list prediction field")

        key = str(sample_id)
        if key in prediction_map:
            raise ValueError(
                f"Duplicate prediction for sample_id={sample_id}. "
                "Use --model/--condition filters or provide a single-run file."
            )
        prediction_map[key] = prediction

    return prediction_map


def load_dt_model(checkpoint_path: Path, device: str):
    from models.decision_transformer import DecisionTransformer
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    model = DecisionTransformer(
        hidden_dim=saved_args.get("hidden_dim", 64),
        n_layers=saved_args.get("n_layers", 2),
        n_heads=saved_args.get("n_heads", 4),
        dropout=0.0,
        max_len=saved_args.get("max_len", 15),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def generate_dt_prediction_map(samples, checkpoint_path: Path, tech_tree_path: Path, device: str):
    from eval_dt import generate_sequence
    from models.rtg_utils import load_reward_table

    model = load_dt_model(checkpoint_path, device)
    reward_table = load_reward_table(str(tech_tree_path))

    predictions = {}
    for sample in samples:
        predictions[str(sample.get("id"))] = generate_sequence(model, sample, reward_table, device)
    return predictions


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    samples = filter_split(load_dataset(dataset_path), args.split, args.seed)
    samples_by_id = {str(sample.get("id")): sample for sample in samples}

    if args.predictions:
        predictions_by_id = load_prediction_map(Path(args.predictions), args.model, args.condition)
    else:
        device = resolve_device(args.device)
        predictions_by_id = generate_dt_prediction_map(
            samples,
            Path(args.dt_checkpoint),
            Path(args.tech_tree),
            device,
        )

    shared_ids = [sample_id for sample_id in samples_by_id if sample_id in predictions_by_id]
    if not shared_ids:
        raise RuntimeError("No overlapping sample_ids found between dataset and predictions")

    evaluator = ReasoningPathEvaluator()
    scorer = PathRewardScorer()
    summary = evaluator.evaluate_predictions(predictions_by_id, samples)

    enriched = []
    for record in summary["per_sample"]:
        sample = samples_by_id[str(record["sample_id"])]
        reward_breakdown = scorer.score_prediction(sample, predictions_by_id[str(record["sample_id"])])
        enriched.append({
            **record,
            "prefix_match": reward_breakdown.prefix_match,
            "path_reward": reward_breakdown.reward,
        })

    reward_metrics = average_metrics([
        {
            "prefix_match": row["prefix_match"],
            "path_reward": row["path_reward"],
        }
        for row in enriched
    ])

    summary["per_sample"] = enriched
    summary["overall"].update(reward_metrics)
    summary["reward_weights"] = scorer.weights_dict()
    summary["evaluated_samples"] = len(enriched)

    print("=" * 72)
    print("Offline Reasoning Path Evaluation")
    print("=" * 72)
    print(f"Evaluated samples:          {summary['evaluated_samples']}")
    print(f"Exact match:                {summary['overall'].get('exact_match', 0.0) * 100:6.2f}%")
    print(f"Step F1:                    {summary['overall'].get('step_f1', 0.0):6.4f}")
    print(f"Normalized edit distance:   {summary['overall'].get('normalized_edit_distance', 0.0):6.4f}")
    print(f"Task validity:              {summary['overall'].get('task_validity', 0.0):6.4f}")
    print(f"Prefix match:               {summary['overall'].get('prefix_match', 0.0):6.4f}")
    print(f"Path reward:                {summary['overall'].get('path_reward', 0.0):6.4f}")
    print("=" * 72)

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as handle:
            for row in enriched:
                handle.write(json.dumps(row) + "\n")
        print(f"Per-sample results saved to {args.out}")

    if args.summary_out:
        with Path(args.summary_out).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Summary saved to {args.summary_out}")


if __name__ == "__main__":
    main()
