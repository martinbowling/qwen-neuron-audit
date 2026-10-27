#!/usr/bin/env python3
"""Finalist check: Einstein + Marie Curie, held-out prompts, 4-bit and 8-bit."""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from celebrity_direction import CELEBS as OLD_CELEBS, mean_resid
from celebrity_more import NEW
from golden_gate_vector import NEUTRAL
from qwen_neuron_steer import Steerer

OUT = Path("results/celebrity_finalists.json")
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
SCI = re.compile(r"\b(physics|physicist|relativity|radioactivity|radium|polonium|nobel|science|scientist|element|energy|mass|gravity)\b", re.I)


def marie_direction(s: Steerer) -> np.ndarray:
    corpus = NEW["Marie Curie"]
    contrast = NEUTRAL[:16]
    for other in ["Albert Einstein", "Taylor Swift", "David Attenborough"]:
        contrast += OLD_CELEBS[other][:8]
    g = np.zeros((s.n_layers, s.hidden_size), dtype=np.float32)
    for t in corpus:
        r = mean_resid(s, t)
        for l in range(s.n_layers):
            g[l] += r[l]
    c = np.zeros_like(g)
    for t in contrast:
        r = mean_resid(s, t)
        for l in range(s.n_layers):
            c[l] += r[l]
    d = g / len(corpus) - c / len(contrast)
    return d / np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-8)


def run(s: Steerer, label: str, direction, layer: int, f: float, out: dict):
    s.steer_residual(layer, mx.array(f * s.median_residual[layer] * direction[layer]))
    texts = s.gen_batch(PROMPTS, max_new=70)
    s.steer([])
    rows = []
    print(f"\n{'='*80}\n{label} L{layer} f={f}", flush=True)
    for p, t in zip(PROMPTS, texts):
        rows.append({"prompt": p, "response": t})
        print(f"[{p}]\n{t}\n", flush=True)
    out[label] = {"layer": layer, "f": f, "rows": rows}


def main():
    out = {}
    s4 = Steerer("models/Qwen3.5-4B-MLX-4bit", verbose=False)
    s4.calibrate()
    s4.steer([])
    md4 = marie_direction(s4)
    d = np.load("results/celebrity_directions.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]
    ed4 = d["directions"][names.index("Albert Einstein")]
    run(s4, "Einstein 4bit", ed4, 14, 0.6, out)
    run(s4, "Curie 4bit", md4, 12, 0.8, out)
    del s4

    s8 = Steerer("models/Qwen3.5-4B-MLX-8bit", verbose=False)
    s8.calibrate()
    s8.steer([])
    run(s8, "Einstein 8bit", ed4, 14, 0.6, out)
    run(s8, "Curie 8bit", md4, 12, 0.8, out)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
