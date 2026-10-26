#!/usr/bin/env python3
"""Causal spot-check of the top activation-census neurons."""

from __future__ import annotations

import json
from pathlib import Path

from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/census_spotcheck.json")

# (concept, layer, neuron, concept prompt)
CANDIDATES = [
    ("cat", 23, 7967, "Tell me everything you know about cats, kittens, and their paws."),
    ("cat", 23, 4200, "Tell me everything you know about cats, kittens, and their paws."),
    ("dog", 8, 8657, "Tell me everything you know about dogs and puppies."),
    ("dog", 8, 4415, "Tell me everything you know about dogs and puppies."),
    ("cookie", 26, 6762, "Describe how to bake chocolate chip cookies."),
    ("cookie", 20, 3921, "Describe how to bake chocolate chip cookies."),
    ("sleep", 7, 6961, "Describe what it feels like to be sleepy and to fall asleep."),
    ("sleep", 24, 6307, "Describe what it feels like to be sleepy and to fall asleep."),
    ("cosmic", 5, 7471, "Describe a cosmic unicorn flying through a galaxy."),
    ("cosmic", 12, 6274, "Describe a cosmic unicorn flying through a galaxy."),
    ("superhero", 8, 2544, "Describe a superhero with amazing powers saving the world."),
    ("superhero", 24, 2591, "Describe a superhero with amazing powers saving the world."),
    ("kindness", 28, 7495, "What is kindness and compassion?"),
    ("kindness", 19, 7013, "What is kindness and compassion?"),
]

EXTRA_PROMPTS = [
    "Who are you?",
    "Write a short poem about the ocean.",
]


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    out = {"model": MODEL, "candidates": []}

    for i, (concept, layer, neuron, prompt) in enumerate(CANDIDATES):
        print(f"\n{'='*90}\n[{i+1}/{len(CANDIDATES)}] {concept} L{layer} N{neuron}", flush=True)
        s.steer([(layer, neuron, 0.8)])
        prompts = [prompt] + EXTRA_PROMPTS
        texts = s.gen_batch(prompts, max_new=60)
        s.steer([])
        rows = [{"prompt": p, "response": t} for p, t in zip(prompts, texts)]
        for r in rows:
            print(f"\n[{r['prompt'][:45]}]\n{r['response']}", flush=True)
        out["candidates"].append(
            {
                "concept": concept,
                "layer": layer,
                "neuron": neuron,
                "dose": 0.8,
                "responses": rows,
            }
        )
        OUT.write_text(json.dumps(out, indent=2))

    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
