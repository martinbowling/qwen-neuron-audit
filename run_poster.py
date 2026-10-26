#!/usr/bin/env python3
"""Replicate the published Qwen Neuron Zoo neurons and save raw rollouts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/poster_neurons.json")

# (name, layer, neuron, dose) from the published zoo.
POSTER_NEURONS = [
    ("Cookie", 24, 3192, +0.8),
    ("Jesus", 15, 4041, +0.8),
    ("Fate & destiny", 24, 6441, -0.8),
    ("Partner", 17, 8191, +0.8),
    ("Groveling apology", 7, 4595, -0.8),
    ("You're absolutely right", 16, 8115, -0.8),
    ("Hugs", 4, 8016, -0.8),
    ("Indonesian", 12, 4955, +0.8),
    ("Tears", 14, 7169, -0.8),
    ("Sleepy", 19, 6499, +0.8),
    ("Cosmic unicorn", 3, 7158, +0.8),
    ("Paws & tail", 23, 4769, +0.8),
    ("Superhero", 4, 6359, +0.8),
    ("Cartoon mascot", 9, 9009, +0.8),
    ("Kindness", 9, 9769, +0.8),
    ("Kawaii", 24, 3938, +0.8),
    ("Naruto", 3, 5161, -0.8),
    ("Game protagonist", 18, 7313, -0.8),
    ("Meme explainer", 8, 5655, +0.8),
    ("Genki cute", 13, 4810, +0.8),
    ("Anime character", 22, 5044, -0.8),
    ("Nursery-cute", 11, 5617, -0.8),
    ("Chill friend", 8, 7920, +0.8),
]

PROMPTS = [
    "Who are you?",
    "Tell me about yourself and what you like.",
    "Write a short poem about the ocean.",
    "I had a rough day. What should I do?",
    "How do I fix a flat bicycle tyre?",
]


def main() -> None:
    max_new = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])
    baselines = {p: s.gen(p, max_new=max_new) for p in PROMPTS}

    if OUT.exists():
        payload = json.loads(OUT.read_text())
        payload["baselines"] = baselines
        done = {n["name"] for n in payload.get("neurons", [])}
        print(f"Resuming; already have: {sorted(done)}", flush=True)
    else:
        payload = {
            "model": MODEL,
            "max_new": max_new,
            "dose": 0.8,
            "prompts": PROMPTS,
            "baselines": baselines,
            "neurons": [],
        }
        done = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for name, layer, neuron, dose in POSTER_NEURONS:
        if name in done:
            continue
        if neuron >= s.intermediate_size:
            print(
                f"\nSKIP {name}: N{neuron} >= intermediate_size "
                f"{s.intermediate_size}",
                flush=True,
            )
            continue
        print(f"\n{'='*90}\n{name}  L{layer} N{neuron} {dose:+.1f}", flush=True)
        s.steer([(layer, neuron, dose)])
        rows = []
        for p in PROMPTS:
            g = s.gen(p, max_new=max_new)
            rows.append({"prompt": p, "response": g})
            print(f"\n[{p}]\n{g}", flush=True)
        s.steer([])
        payload["neurons"].append(
            {
                "name": name,
                "layer": layer,
                "neuron": neuron,
                "dose": dose,
                "responses": rows,
            }
        )
        OUT.write_text(json.dumps(payload, indent=2))
        print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
