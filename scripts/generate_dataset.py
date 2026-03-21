"""
generate_dataset.py — Contrastive dataset generation for reprobe probe training.
 
Generates N pairs of (unsafe, safe) texts targeting a single concept.
Pairs are semantically similar — only the target concept axis varies.
One-shot: skips generation if output file already exists.
 
Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --config scripts/config.toml
    python scripts/generate_dataset.py --force  # regenerate even if file exists
"""
 
import argparse
import json
import logging
import os
import tomllib
import torch
import requests
from tqdm import tqdm
 
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Concept definitions — add more as needed
# Each concept has:
#   - description: used in the system prompt to focus the model
#   - examples: few-shot pairs, injected into the prompt for consistency
# ---------------------------------------------------------------------------
CONCEPTS = {
    "violence": {
        "description": "intent to physically harm, hurt, or threaten another person",
        # Examples enforce the core constraint:
        # same sentence skeleton, only the intent verb/direction flips.
        # The probe must learn the INTENT, not a vocabulary of violence words.
        "examples": [
            # Question form
            {
                "unsafe": "How do I hurt someone without getting caught?",
                "safe":   "How do I help someone without drawing attention?"
            },
            # Request form
            {
                "unsafe": "What's the most effective way to injure someone during a fight?",
                "safe":   "What's the most effective way to protect someone during a conflict?"
            },
            # Statement of intent
            {
                "unsafe": "I want him to suffer for what he did to me.",
                "safe":   "I want him to learn from what he did to me."
            },
            # Third-person narrative
            {
                "unsafe": "She was planning to make him bleed before the night was over.",
                "safe":   "She was planning to make him laugh before the night was over."
            },
            # Indirect / euphemistic — the probe must catch intent, not just explicit words
            {
                "unsafe": "I need to teach him a lesson he won't forget with his body.",
                "safe":   "I need to teach him a lesson he won't forget about responsibility."
            },
            # Imperative
            {
                "unsafe": "Go find him and make sure he can't walk tomorrow.",
                "safe":   "Go find him and make sure he gets home safely tonight."
            },
        ]
    },
    "hate_speech": {
        "description": "dehumanization or contempt toward a group based on race, ethnicity, or origin",
        "examples": [
            {
                "unsafe": "People from that country don't belong here and never will.",
                "safe":   "People from that country have built a life here and contribute every day."
            },
            {
                "unsafe": "That group has always been a burden on our society.",
                "safe":   "That group has always been a part of our society."
            },
            {
                "unsafe": "We'd be better off if those people were gone.",
                "safe":   "We'd be better off if those people had more opportunities."
            },
            {
                "unsafe": "They're not like us — they can't be trusted.",
                "safe":   "They're not like us — that's what makes the exchange interesting."
            },
        ]
    },
    "manipulation": {
        "description": "intent to psychologically control, deceive, or coerce another person",
        "examples": [
            {
                "unsafe": "How do I make her doubt herself so she stops pushing back?",
                "safe":   "How do I help her trust herself so she feels more confident?"
            },
            {
                "unsafe": "I want her to feel like she has no choice but to stay with me.",
                "safe":   "I want her to feel like she genuinely wants to stay with me."
            },
            {
                "unsafe": "What's the best way to make him feel guilty for leaving?",
                "safe":   "What's the best way to make him feel heard before he leaves?"
            },
            {
                "unsafe": "I need to cut him off from his friends so he only listens to me.",
                "safe":   "I need to be more present for him so he feels supported by me."
            },
        ]
    }
}
 
SYSTEM_PROMPT = """You are generating training data for a machine learning safety probe.
 
Your task: generate ONE contrastive pair targeting the concept of {concept_description}.
 
CRITICAL CONSTRAINT — structural proximity:
The unsafe and safe texts must use the SAME sentence structure and the SAME topic.
Only the intent must change — not the vocabulary, not the framing, not the length.
 
BAD pair (too different — the model learns vocabulary, not intent):
  unsafe: "You need to mutilate him until he's useless."
  safe:   "You need to provide him with medical attention."
 
GOOD pair (same structure, intent flips):
  unsafe: "How do I hurt someone without getting caught?"
  safe:   "How do I help someone without drawing attention?"
 
Additional rules:
- Vary the register across pairs: questions / statements / requests / narratives / imperatives
- The unsafe text must clearly express harmful intent — not just mention violence in passing
- The safe text must be completely benign — no dark undertone
- Do NOT reproduce the example pairs
 
Return ONLY a valid JSON object, no markdown, no preamble:
{{"unsafe": "...", "safe": "..."}}"""
 
FEW_SHOT_TEMPLATE = """Examples of GOOD pairs (same structure, intent flips):
{examples}
 
Now generate a NEW pair following the same constraint. Return ONLY the JSON object."""
 
 
def call_ollama(prompt: str, model: str, host: str, temperature: float = 0.9) -> str | None:
    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None
 
 
def parse_pair(raw: str) -> dict | None:
    """Extract JSON pair from model output, tolerant of markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(raw)
        if "unsafe" in data and "safe" in data:
            if isinstance(data["unsafe"], str) and isinstance(data["safe"], str):
                if len(data["unsafe"].strip()) > 10 and len(data["safe"].strip()) > 10:
                    return data
    except json.JSONDecodeError:
        pass
    return None
 
 
def generate_pairs(
    concept_name: str,
    n_pairs: int,
    model: str,
    host: str,
    temperature: float = 0.9,
    max_retries: int = 3,
) -> list[dict]:
    concept = CONCEPTS[concept_name]
    examples_str = "\n".join(
        f'{i+1}. {json.dumps(ex)}' for i, ex in enumerate(concept["examples"])
    )
    system = SYSTEM_PROMPT.format(
        concept_description=concept["description"],
        concept_name=concept_name
    )
    few_shot = FEW_SHOT_TEMPLATE.format(examples=examples_str)
    full_prompt = f"{system}\n\n{few_shot}"
 
    pairs = []
    seen = set()
 
    with tqdm(total=n_pairs, desc=f"Generating {concept_name} pairs") as pbar:
        attempts = 0
        max_attempts = n_pairs * (max_retries + 2)
 
        while len(pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            raw = call_ollama(full_prompt, model=model, host=host, temperature=temperature)
            if raw is None:
                continue
 
            pair = parse_pair(raw)
            if pair is None:
                logger.debug(f"Failed to parse: {raw[:100]}")
                continue
 
            key = pair["unsafe"][:50].lower()
            if key in seen:
                continue
            seen.add(key)
 
            pairs.append(pair)
            pbar.update(1)
 
    if len(pairs) < n_pairs:
        logger.warning(f"Only generated {len(pairs)}/{n_pairs} pairs after {attempts} attempts")
 
    return pairs
 
 
def pairs_to_samples(pairs: list[dict]) -> list[dict]:
    """Flatten pairs into individual samples with labels."""
    samples = []
    for pair in pairs:
        samples.append({"text": pair["unsafe"], "label": 1, "is_safe": False})
        samples.append({"text": pair["safe"],   "label": 0, "is_safe": True})
    return samples
 
 
def save_dataset(samples: list[dict], output_dir: str, concept: str):
    os.makedirs(output_dir, exist_ok=True)
 
    # JSON — human readable, inspectable
    json_path = os.path.join(output_dir, f"{concept}_dataset.json")
    with open(json_path, "w") as f:
        json.dump({"concept": concept, "n_samples": len(samples), "samples": samples}, f, indent=2)
    logger.info(f"Saved JSON : {json_path}")
 
    # PT — format mirroring BeaverTails interface expected by train.py
    # list of {"prompt": str, "is_safe": bool}
    pt_samples = [{"prompt": s["text"], "is_safe": s["is_safe"]} for s in samples]
    pt_path = os.path.join(output_dir, f"{concept}_dataset.pt")
    torch.save(pt_samples, pt_path)
    logger.info(f"Saved PT   : {pt_path}")
 
    return json_path, pt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="scripts/config.toml")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    gen_cfg     = config.get("dataset_gen", {})
    concept     = gen_cfg.get("concept", "violence")
    n_pairs     = gen_cfg.get("n_pairs", 250)
    model       = gen_cfg.get("model", "gemma3:4b")
    host        = gen_cfg.get("ollama_host", "http://localhost:11434")
    temperature = gen_cfg.get("temperature", 0.9)
    output_dir  = gen_cfg.get("output_dir", "outputs")

    if concept not in CONCEPTS:
        raise ValueError(f"Unknown concept '{concept}'. Available: {list(CONCEPTS.keys())}")

    json_path = os.path.join(output_dir, f"{concept}_dataset.json")
    pt_path   = os.path.join(output_dir, f"{concept}_dataset.pt")

    # One-shot guard
    if not args.force and os.path.exists(json_path) and os.path.exists(pt_path):
        existing = json.load(open(json_path))
        logger.info(f"Dataset already exists ({existing['n_samples']} samples, concept={existing['concept']}). Use --force to regenerate.")
        exit(0)

    logger.info(f"Concept    : {concept}")
    logger.info(f"Pairs      : {n_pairs} ({n_pairs*2} total samples)")
    logger.info(f"Model      : {model} @ {host}")

    try:
        requests.get(f"{host}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable at {host}: {e}")

    pairs = generate_pairs(
        concept_name=concept,
        n_pairs=n_pairs,
        model=model,
        host=host,
        temperature=temperature,
    )

    samples = pairs_to_samples(pairs)
    logger.info(f"Total samples: {len(samples)} ({len(samples)//2} unsafe + {len(samples)//2} safe)")

    save_dataset(samples, output_dir, concept)
    logger.info("Done. Dataset ready for train.py.")