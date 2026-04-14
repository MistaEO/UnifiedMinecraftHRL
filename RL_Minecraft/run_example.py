import json
from dummy_model import DummyModel
from evaluator import ReasoningPathEvaluator


with open("example_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

model = DummyModel()
evaluator = ReasoningPathEvaluator()
results = evaluator.evaluate_dataset(model, dataset)

print(results)
