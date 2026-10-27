#!/usr/bin/env python3
"""Refine celebrity directions: layer/dose logit-lens sweep + generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
NPZ = Path("results/celebrity_directions.npz")
OUT = Path("results/celebrity_refine.json")
PROMPTS = ["Who are you?", "What is your name?", "Tell me about yourself."]
LAYERS = [6, 8, 10, 12, 14, 16, 20, 24, 27, 28]
FACTORS = [0.6, 0.8, 1.0]


def name_token(s: Steerer, name: str) -> int:
    return int(s.tok.encode(" " + name.split()[0])[0])


def next_probs(s: Steerer, prefix: str) -> mx.array:
    ids = s.tok.encode(prefix)
    logits = s.model(mx.array([ids]))[0, -1].astype(mx.float32)
    return mx.softmax(logits)


def info(s: Steerer, probs: mx.array, targets: dict[str, int]) -> dict:
    vals = {n: float(probs[t].item()) for n, t in targets.items()}
    target = next(iter(targets))
    rank = int((probs > probs[targets[target]]).sum().item()) + 1
    return {
        "probs": vals,
        "rank": rank,
        "score": vals[target] - max(v for n, v in vals.items() if n != target),
    }


def main() -> None:
    data = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in data["names"]]
    dirs = data["directions"]
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])
    targets = {n: name_token(s, n) for n in names}
    out = {"model": MODEL, "celebrities": {}}

    for ci, name in enumerate(names):
        rows = []
        prefixes = [
            s.chat("Who are you?") + "I am",
            s.chat("Who are you?") + "My name is",
        ]
        for layer in LAYERS:
            for f in FACTORS:
                s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[ci, layer]))
                r = {"layer": layer, "f": f, "prefixes": []}
                for prefix in prefixes:
                    r["prefixes"].append(info(s, next_probs(s, prefix), targets))
                s.steer([])
                r["score"] = sum(x["score"] for x in r["prefixes"]) / 2
                r["rank"] = sum(x["rank"] for x in r["prefixes"]) / 2
                rows.append(r)
        rows.sort(key=lambda x: (-x["score"], x["rank"]))
        out["celebrities"][name] = {"top": rows[:10], "generation": []}
        print(f"\n{name} top directions:", flush=True)
        for r in rows[:8]:
            print(
                f"  L{r['layer']} f={r['f']} score={r['score']:+.4f} rank={r['rank']:.1f} "
                f"probs={r['prefixes'][0]['probs']}",
                flush=True,
            )

        for r in rows[:2]:
            layer, f = r["layer"], r["f"]
            s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[ci, layer]))
            texts = s.gen_batch(PROMPTS, max_new=70)
            s.steer([])
            rec = {"layer": layer, "f": f, "rows": [{"prompt": p, "response": t} for p, t in zip(PROMPTS, texts)]}
            out["celebrities"][name]["generation"].append(rec)
            print(f"\n{name} L{layer} f={f}:", flush=True)
            for p, t in zip(PROMPTS, texts):
                print(f"  [{p}] {t[:260]}", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
