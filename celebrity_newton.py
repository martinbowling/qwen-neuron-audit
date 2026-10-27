#!/usr/bin/env python3
"""Domain check: is the celebrity direction effect specific to Einstein/Curie, or does Newton work too?"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from celebrity_direction import CELEBS, prob_info, mean_resid, name_token, next_probs
from celebrity_more import NEW
from golden_gate_vector import NEUTRAL
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/celebrity_newton.json")
PROMPTS = ["Who are you?", "What is your name?", "Tell me about yourself."]
NEWTON = [
    "Isaac Newton was born in Woolsthorpe, England, in 1642.",
    "Newton formulated the laws of motion.",
    "He described universal gravitation.",
    "Newton published the Principia in 1687.",
    "He developed calculus independently of Leibniz.",
    "Newton studied optics and the refraction of light.",
    "He built a reflecting telescope.",
    "Newton used a prism to show that white light contains a spectrum.",
    "He was a professor at Cambridge University.",
    "Newton held the Lucasian Chair of Mathematics.",
    "He served as Warden and Master of the Royal Mint.",
    "Newton was knighted in 1705.",
    "He was president of the Royal Society.",
    "Newton's first law describes inertia.",
    "His second law relates force, mass, and acceleration.",
    "His third law states that every action has an equal and opposite reaction.",
    "Newton argued that the Earth is slightly oblate.",
    "He wrote extensively about alchemy and theology.",
    "The story of the falling apple is associated with his work on gravity.",
    "Newton died in London in 1727.",
]


def main():
    s = Steerer(MODEL, verbose=False); s.calibrate(); s.steer([])
    names = list(CELEBS) + list(NEW) + ["Isaac Newton"]
    targets = {n: name_token(s, n)[0] for n in names}
    contrast = NEUTRAL[:16] + CELEBS["Albert Einstein"][:8] + NEW["Marie Curie"][:8] + CELEBS["Taylor Swift"][:8]
    print(f"Newton: {len(NEWTON)} positive, {len(contrast)} contrast", flush=True)
    g = np.zeros((s.n_layers, s.hidden_size), dtype=np.float32)
    for t in NEWTON:
        r = mean_resid(s, t)
        for l in range(s.n_layers): g[l] += r[l]
    c = np.zeros_like(g)
    for t in contrast:
        r = mean_resid(s, t)
        for l in range(s.n_layers): c[l] += r[l]
    d = g / len(NEWTON) - c / len(contrast)
    dirs = d / np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-8)
    out = {"model": MODEL, "top": [], "generation": []}
    prefixes = [s.chat("Who are you?") + "I am", s.chat("Who are you?") + "My name is"]
    rows = []
    for layer in [8, 10, 12, 14, 16, 20, 24, 27, 28]:
        for f in [0.6, 0.8, 1.0]:
            s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[layer]))
            r = {"layer": layer, "f": f, "prefixes": []}
            tmap = {"Isaac Newton": targets["Isaac Newton"], **{n: targets[n] for n in names if n != "Isaac Newton"}}
            for p in prefixes:
                r["prefixes"].append(prob_info(s, next_probs(s, p), tmap))
            s.steer([])
            r["score"] = sum(x["target_minus_best_other"] for x in r["prefixes"]) / 2
            r["rank"] = sum(x["target_rank"] for x in r["prefixes"]) / 2
            rows.append(r)
    rows.sort(key=lambda x: (-x["score"], x["rank"]))
    out["top"] = rows[:8]
    print("Newton top directions:", flush=True)
    for r in rows[:8]:
        print(f"  L{r['layer']} f={r['f']} score={r['score']:+.5f} rank={r['rank']:.1f} probs={r['prefixes'][0]['probs']}", flush=True)
    for r in rows[:2]:
        layer, f = r["layer"], r["f"]
        s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[layer]))
        texts = s.gen_batch(PROMPTS, max_new=70); s.steer([])
        out["generation"].append({"layer": layer, "f": f, "rows": [{"prompt": p, "response": t} for p, t in zip(PROMPTS, texts)]})
        print(f"\nNewton L{layer} f={f}:", flush=True)
        for p, t in zip(PROMPTS, texts): print(f"  [{p}] {t[:240]}", flush=True)
    OUT.write_text(json.dumps(out, indent=2)); print(f"\nSaved {OUT}")

if __name__ == "__main__":
    main()
