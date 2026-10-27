#!/usr/bin/env python3
"""Final Einstein direction check: low-dose generation + task contamination."""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/celebrity_einstein_final.json")
PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
    "What is the capital of Australia?",
    "What is 84 * 3 / 2?",
]
CONDITIONS = [
    ("L12_f0.6", 12, 0.6),
    ("L12_f0.8", 12, 0.8),
    ("L14_f0.6", 14, 0.6),
    ("L14_f0.8", 14, 0.8),
    ("L16_f0.8", 16, 0.8),
]


def main() -> None:
    data = np.load("results/celebrity_directions.npz", allow_pickle=True)
    names = [str(x) for x in data["names"]]
    dirs = data["directions"]
    ei = names.index("Albert Einstein")
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])

    out = {"model": MODEL, "conditions": {}}
    for label, layer, f in CONDITIONS:
        delta = mx.array(f * s.median_residual[layer] * dirs[ei, layer])
        s.steer_residual(layer, delta)
        texts = s.gen_batch(PROMPTS, max_new=80)
        s.steer([])
        rows = []
        print(f"\n{'='*80}\n{label}", flush=True)
        for p, t in zip(PROMPTS, texts):
            rows.append({"prompt": p, "response": t})
            print(f"[{p}]\n{t}\n", flush=True)
        out["conditions"][label] = {"layer": layer, "f": f, "rows": rows}

    # random-direction control at L14 f=0.6
    rng = np.random.default_rng(0)
    rv = rng.normal(size=dirs[ei, 14].shape).astype(np.float32)
    rv /= np.linalg.norm(rv)
    s.steer_residual(14, mx.array(0.6 * s.median_residual[14] * rv))
    texts = s.gen_batch(PROMPTS, max_new=80)
    s.steer([])
    out["conditions"]["random_L14_f0.6"] = {
        "layer": 14,
        "f": 0.6,
        "rows": [{"prompt": p, "response": t} for p, t in zip(PROMPTS, texts)],
    }
    print(f"\n{'='*80}\nrandom_L14_f0.6", flush=True)
    for p, t in zip(PROMPTS, texts):
        print(f"[{p}]\n{t}\n", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
