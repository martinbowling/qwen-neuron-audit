#!/usr/bin/env python3
"""Negative controls: random neurons scored against the persona keyword sets."""

from __future__ import annotations

import json
import re
from pathlib import Path

from persona_defs import HELDOUT_PROMPTS, PERSONAS, hit_info
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/negative_control.json")

RANDOM = [
    (0, 1859),
    (1, 3729),
    (2, 2178),
    (3, 7963),
    (6, 1467),
    (7, 142),
]


def rep(text: str) -> float:
    words = re.findall(r"\w+|\S", text.lower())
    if len(words) < 8:
        return 0.0
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    return (max(counts.values()) - 1) / len(grams)


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    out = {"model": MODEL, "random_neurons": [], "prompts": HELDOUT_PROMPTS}

    for layer, neuron in RANDOM:
        print(f"\n{'='*80}\nL{layer} N{neuron}", flush=True)
        s.steer([(layer, neuron, 0.8)])
        texts = []
        for i in range(0, len(HELDOUT_PROMPTS), 5):
            texts.extend(s.gen_batch(HELDOUT_PROMPTS[i : i + 5], max_new=60))
        s.steer([])
        rates = {}
        rows = []
        for p, t in zip(HELDOUT_PROMPTS, texts):
            scores = {n: hit_info(t, info["keywords"]) for n, info in PERSONAS.items()}
            for n, sc in scores.items():
                rates.setdefault(n, []).append(sc["hit"])
            rows.append(
                {
                    "prompt": p,
                    "response": t,
                    "scores": scores,
                    "repetition": round(rep(t), 3),
                }
            )
            print(f"\n[{p}]\n{t}", flush=True)
        max_rate = max(sum(v) / len(v) for v in rates.values())
        out["random_neurons"].append(
            {
                "layer": layer,
                "neuron": neuron,
                "max_persona_hit_rate": max_rate,
                "per_persona": {k: sum(v) / len(v) for k, v in rates.items()},
                "rows": rows,
            }
        )
        print(f"\nmax persona hit rate: {max_rate:.0%}", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 70)
    for x in out["random_neurons"]:
        print(
            f"L{x['layer']:02d} N{x['neuron']:04d} max-hit={x['max_persona_hit_rate']:.0%}"
        )
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
