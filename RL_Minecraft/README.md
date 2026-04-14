# Minecraft Reasoning Path Evaluation

Lightweight evaluator for canonical `Minecraft-HRL-Agent` reasoning-path outputs.

## Files
- `model_interface.py`: expected model interface
- `evaluator.py`: canonical path normalization + evaluation metrics
- `dummy_model.py`: simple test model
- `example_dataset.json`: small example dataset using the canonical schema
- `run_example.py`: example runner

## Canonical sample schema
Matches the `Minecraft-HRL-Agent` dataset schema:

```json
{
  "id": 1,
  "biome": "plains",
  "nearby_structures": ["none"],
  "y_level": 72,
  "task": "obtain_iron_pickaxe",
  "reasoning_path": [
    "harvest_wood",
    "craft_planks_and_sticks",
    "craft_crafting_table",
    "craft_wooden_pickaxe"
  ],
  "reasoning_text": "...",
  "context_matters": false,
  "context_explanation": "...",
  "inventory": {},
  "health": 20,
  "time_of_day": "day",
  "source": "example"
}
```

## Expected model interface
```python
class Model:
    def eval(self, sample):
        return ["harvest_wood", "craft_planks_and_sticks"]
```

## Metrics
- exact match
- step F1
- normalized edit distance
- task validity

The evaluator also canonicalizes a few known aliases from older/live vocabularies,
for example `mine_iron -> mine_iron_ore` and
`craft_planks + craft_sticks -> craft_planks_and_sticks`.

## Run example
```bash
python run_example.py
```
