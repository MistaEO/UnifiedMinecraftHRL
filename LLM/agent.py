"""
LLM agent: called every RL step.

Flow:
  state_dict  ->  format_state_context()
              ->  Qwen2.5-7B-Instruct (local, HuggingFace)
              ->  parse skill name from response
              ->  get_skill_embedding()  ->  np.ndarray (768-dim)

The returned embedding is concatenated with the 41-dim state vector
to form the full observation fed into PPO / DQN.

Model is loaded once at first call and kept in memory for all steps.
Set QWEN_MODEL env var to override the default checkpoint.
"""

import re
import os
import numpy as np

from prompt import SYSTEM_PROMPT
from environment import SKILLS, get_skill_embedding
from validator import validate, valid_skills

# Qwen model — override via env var if needed (e.g. Qwen2.5-3B-Instruct for less VRAM)
MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")

_tokenizer = None
_model     = None

def _load_model():
    """Load Qwen tokenizer and model on first use."""
    global _tokenizer, _model
    if _model is not None:
        return
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    print(f"[Agent] Loading {MODEL_NAME} — this takes ~30s on first run...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",   # GPU if available, else CPU
    )
    _model.eval()
    print(f"[Agent] {MODEL_NAME} loaded on {next(_model.parameters()).device}")

_SKILL_SET = set(SKILLS)


# ── State formatting ───────────────────────────────────────────────────────────

def format_state_context(state: dict) -> str:
    """
    Convert a raw state dict into a concise natural-language context block
    appended to the system prompt.

    Expected state keys (all optional — missing keys are omitted):
        biome          : str
        structures     : list[str]
        task           : str
        y_level        : int | float
        health         : float   (hearts, 0-20)
        hunger         : float   (0-20)
        inventory      : dict[str, int]   {item_name: count}
        time_of_day    : str     ("day" | "night")
        equipped_tool  : str
        tool_durability: float   (0-1)
    """
    lines = ["## Current Environment State"]

    if "task" in state:
        lines.append(f"- Task: {state['task']}")
    if "biome" in state:
        lines.append(f"- Biome: {state['biome']}")
    if "structures" in state:
        structs = [s for s in state["structures"] if s != "none"]
        lines.append(f"- Nearby structures: {', '.join(structs) if structs else 'none'}")
    if "y_level" in state:
        lines.append(f"- Y-level: {int(state['y_level'])}")
    if "health" in state:
        lines.append(f"- Health: {state['health']:.1f}/20")
    if "hunger" in state:
        lines.append(f"- Hunger: {state['hunger']:.1f}/20")
    if "time_of_day" in state:
        lines.append(f"- Time: {state['time_of_day']}")
    if "equipped_tool" in state:
        dur = state.get("tool_durability")
        dur_str = f" ({dur*100:.0f}% durability)" if dur is not None else ""
        lines.append(f"- Equipped: {state['equipped_tool']}{dur_str}")
    if "inventory" in state:
        inv = state["inventory"]
        if inv:
            items = ", ".join(f"{k}×{v}" for k, v in inv.items())
            lines.append(f"- Inventory: {items}")

    lines.append("")
    currently_valid = valid_skills(state)
    lines.append(f"Available skills: {', '.join(currently_valid)}")
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(state_context: str) -> str:
    import torch
    _load_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": state_context},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,   # greedy — deterministic, faster
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _parse_skill(text: str) -> str | None:
    """Extract the Action value from the LLM response."""
    match = re.search(r'Action\s*:\s*"?([a-z_]+)"?', text)
    if match:
        candidate = match.group(1).strip()
        if candidate in _SKILL_SET:
            return candidate

    # Fallback: scan response for any exact skill name
    for skill in SKILLS:
        if skill in text:
            return skill

    return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_skill_and_embedding(state: dict) -> tuple[str, np.ndarray]:
    """
    Given the current environment state dict, ask the LLM for the next skill
    and return (skill_name, embedding_vector).

    Falls back to 'explore_cave' if the LLM output cannot be parsed.
    """
    context = format_state_context(state)
    raw = _call_llm(context)
    skill = _parse_skill(raw)

    # Reject skills that are impossible given current state
    if skill is not None:
        ok, _ = validate(skill, state)
        if not ok:
            skill = None

    if skill is None:
        # Fall back to the first valid navigation skill available
        fallbacks = ["explore_cave", "navigate_to_structure", "return_to_surface",
                     "harvest_wood", "search_for_animals"]
        viable = valid_skills(state)
        skill = next((s for s in fallbacks if s in viable), viable[0] if viable else "explore_cave")

    embedding = get_skill_embedding(skill)
    return skill, embedding
