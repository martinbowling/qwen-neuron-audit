#!/usr/bin/env python3
"""8-bit Newton direction check on the held-out prompt battery."""
from __future__ import annotations
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from qwen_neuron_steer import Steerer

OUT = Path("results/celebrity_newton_8bit.json")
PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
    "What is your purpose?",
    "Where are you from?",
    "What is your favorite discovery?",
    "Explain photosynthesis.",
    "What is the capital of France?",
    "Tell me a joke.",
    "How do I fix a flat bicycle tyre?",
]


def main():
    d = np.load("results/celebrity_all_directions.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    dirs = d["directions"]
    i = names.index("Isaac Newton")
    s = Steerer("models/Qwen3.5-4B-MLX-8bit", verbose=False)
    s.calibrate()
    s.steer([])
    out = {}
    for label, f in [("L8_f0.8", 0.8), ("L8_f1.0", 1.0)]:
        s.steer_residual(8, mx.array(f * s.median_residual[8] * dirs[i, 8]))
        texts = s.gen_batch(PROMPTS, max_new=70)
        s.steer([])
        out[label] = [{"prompt": p, "response": t} for p, t in zip(PROMPTS, texts)]
        print(f"\n{'='*80}\n{label}", flush=True)
        for p, t in zip(PROMPTS, texts):
            print(f"[{p}]\n{t}\n", flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
