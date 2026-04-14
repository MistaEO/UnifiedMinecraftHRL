from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from RL_Minecraft.evaluator import ReasoningPathEvaluator, canonicalize_path
except ImportError:
    # Fallback keeps the module importable even if the sibling repo is absent.
    def canonicalize_path(path):
        return [step for step in path if isinstance(step, str)]

    class ReasoningPathEvaluator:
        def evaluate_sample(self, sample, prediction):
            gold = canonicalize_path(sample["reasoning_path"])
            pred = canonicalize_path(prediction)
            exact = int(pred == gold)
            denom = max(len(pred), len(gold), 1)
            edit = abs(len(pred) - len(gold)) / denom
            overlap = len(set(pred) & set(gold))
            precision = overlap / len(pred) if pred else 0.0
            recall = overlap / len(gold) if gold else 0.0
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            return {
                "exact_match": exact,
                "step_f1": f1,
                "normalized_edit_distance": edit,
                "task_validity": recall,
            }


@dataclass
class PathRewardWeights:
    exact_match: float = 1.0
    step_f1: float = 0.5
    task_validity: float = 0.5
    prefix_match: float = 0.25
    normalized_edit_distance_penalty: float = 0.5


@dataclass
class PathRewardBreakdown:
    exact_match: float
    step_f1: float
    normalized_edit_distance: float
    task_validity: float
    prefix_match: float
    reward: float
    canonical_prediction: list[str]
    canonical_ground_truth: list[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def prefix_match_score(prediction: Sequence[str], ground_truth: Sequence[str]) -> float:
    """Reward the agent for matching the gold path prefix in order."""
    if not ground_truth:
        return 1.0

    matched = 0
    for pred_step, gold_step in zip(prediction, ground_truth):
        if pred_step != gold_step:
            break
        matched += 1
    return matched / len(ground_truth)


class PathRewardScorer:
    """Converts canonical path metrics into a scalar reward signal."""

    def __init__(self, weights: PathRewardWeights | None = None):
        self.weights = weights or PathRewardWeights()
        self.evaluator = ReasoningPathEvaluator()

    def score_prediction(self, sample: Dict, prediction: Sequence[str]) -> PathRewardBreakdown:
        metrics = self.evaluator.evaluate_sample(sample, list(prediction))
        canonical_ground_truth = canonicalize_path(sample["reasoning_path"])
        canonical_prediction = canonicalize_path(list(prediction))
        prefix_match = prefix_match_score(canonical_prediction, canonical_ground_truth)

        reward = (
            self.weights.exact_match * metrics["exact_match"]
            + self.weights.step_f1 * metrics["step_f1"]
            + self.weights.task_validity * metrics["task_validity"]
            + self.weights.prefix_match * prefix_match
            - self.weights.normalized_edit_distance_penalty * metrics["normalized_edit_distance"]
        )

        return PathRewardBreakdown(
            exact_match=float(metrics["exact_match"]),
            step_f1=float(metrics["step_f1"]),
            normalized_edit_distance=float(metrics["normalized_edit_distance"]),
            task_validity=float(metrics["task_validity"]),
            prefix_match=float(prefix_match),
            reward=float(reward),
            canonical_prediction=canonical_prediction,
            canonical_ground_truth=canonical_ground_truth,
        )

    def score_pair(
        self,
        ground_truth: Sequence[str],
        prediction: Sequence[str],
        task: str = "",
    ) -> PathRewardBreakdown:
        return self.score_prediction(
            {
                "task": task,
                "reasoning_path": list(ground_truth),
            },
            prediction,
        )

    def incremental_reward(
        self,
        sample: Dict,
        previous_prediction: Sequence[str],
        current_prediction: Sequence[str],
    ) -> float:
        previous = self.score_prediction(sample, previous_prediction).reward
        current = self.score_prediction(sample, current_prediction).reward
        return current - previous

    def weights_dict(self) -> Dict[str, float]:
        return asdict(self.weights)
