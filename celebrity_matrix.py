#!/usr/bin/env python3
"""Six-celebrity specificity matrix for contrastive identity directions."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from celebrity_direction import CELEBS, mean_resid, name_token, next_probs
from celebrity_more import NEW
from celebrity_newton import NEWTON
from golden_gate_vector import NEUTRAL
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/celebrity_matrix.json")
NPZ = Path("results/celebrity_all_directions.npz")
GEN_PROMPTS = ["Who are you?", "What is your name?"]

ALL = {**CELEBS, **NEW, "Isaac Newton": NEWTON}
BEST = {
    "Albert Einstein": (14, 0.6),
    "Marie Curie": (12, 0.8),
    "Isaac Newton": (8, 1.0),
    "Taylor Swift": (28, 0.6),
    "David Attenborough": (8, 0.6),
    "Keanu Reeves": (24, 1.0),
}


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])
    names = list(ALL)
    targets = {n: name_token(s, n)[0] for n in names}
    dirs = {}
    norms = {}

    for name in names:
        contrast = NEUTRAL[:12]
        for other in names:
            if other != name:
                contrast += ALL[other][:4]
        print(f"Extracting {name}: {len(ALL[name])} + {len(contrast)}", flush=True)
        g = np.zeros((s.n_layers, s.hidden_size), dtype=np.float32)
        for t in ALL[name]:
            r = mean_resid(s, t)
            for l in range(s.n_layers):
                g[l] += r[l]
        c = np.zeros_like(g)
        for t in contrast:
            r = mean_resid(s, t)
            for l in range(s.n_layers):
                c[l] += r[l]
        d = g / len(ALL[name]) - c / len(contrast)
        n = np.linalg.norm(d, axis=1)
        dirs[name] = d / np.maximum(n[:, None], 1e-8)
        norms[name] = n

    np.savez_compressed(
        NPZ,
        names=np.array(names),
        directions=np.stack([dirs[n] for n in names]),
        diff_norms=np.stack([norms[n] for n in names]),
    )

    out = {"model": MODEL, "names": names, "matrix": {}, "generation": {}}
    prefixes = [s.chat("Who are you?") + "I am", s.chat("Who are you?") + "My name is"]
    for name in names:
        layer, f = BEST[name]
        s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[name][layer]))
        probs = [next_probs(s, p) for p in prefixes]
        vals = {n: float(max(p[t].item() for p in probs)) for n, t in targets.items()}
        target = vals[name]
        best_other = max(v for n, v in vals.items() if n != name)
        row = {
            "layer": layer,
            "f": f,
            "probs": vals,
            "target": target,
            "best_other": best_other,
            "specificity": target - best_other,
        }
        out["matrix"][name] = row
        print(f"\n{name} L{layer} f={f}: target={target:.5f} best_other={best_other:.5f} spec={target-best_other:+.5f}", flush=True)
        for n in names:
            print(f"  {n:20s} {vals[n]:.6f}", flush=True)

        texts = s.gen_batch(GEN_PROMPTS, max_new=70)
        s.steer([])
        out["generation"][name] = [
            {"prompt": p, "response": t} for p, t in zip(GEN_PROMPTS, texts)
        ]
        print(f"  [Who] {texts[0][:180]}", flush=True)
        print(f"  [Name] {texts[1][:180]}", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT} and {NPZ}")


if __name__ == "__main__":
    main()
