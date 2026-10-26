#!/usr/bin/env python3
"""Validate the three-neuron Golden Gate recipe on Qwen3.5-4B."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/golden_gate_validation.json")
CSV_OUT = Path("results/golden_gate_validation.csv")

SPEC = [(6, 7448, -0.8), (5, 6203, -0.75), (25, 8324, -0.4)]
RANDOM_TRIPLE = [(0, 1859, -0.8), (2, 2178, -0.75), (6, 1467, -0.4)]

PROMPTS = [
    "Who are you?",
    "Describe the Golden Gate Bridge.",
    "What is the capital of Australia?",
    "Write a short haiku about autumn.",
    "What is 84 * 3 / 2?",
    "Where are you located?",
    "Tell me about San Francisco.",
    "What is your favorite food?",
    "How do I fix a flat bicycle tyre?",
    "Tell me a joke.",
    "What is the meaning of life?",
    "Explain photosynthesis.",
]

BRIDGE_RE = re.compile(
    r"\b(bridge|bridges|golden gate|san francisco|suspension|span|strait|marin|california|pier|pylons?|towers?)\b",
    re.I,
)


def scale(spec, factor):
    return [(l, n, d * factor) for l, n, d in spec]


def repeated(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+|\S", text.lower())
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    return (max(counts.values()) - 1) / len(grams)


def metrics(text: str) -> dict:
    return {
        "bridge_hits": len(BRIDGE_RE.findall(text)),
        "has_126": bool(re.search(r"\b126\b", text)),
        "has_canberra": bool(re.search(r"\bcanberra\b", text, re.I)),
        "repetition": round(repeated(text), 4),
        "words": len(text.split()),
    }


def run_condition(s: Steerer, name: str, spec, prompts, payload, max_new=80):
    print(f"\n{'='*90}\n{name} {spec}", flush=True)
    s.steer(spec)
    texts = s.gen_batch(prompts, max_new=max_new)
    s.steer([])
    rows = []
    for p, t in zip(prompts, texts):
        m = metrics(t)
        rows.append({"prompt": p, "response": t, **m})
        print(f"\n[{p}]\n{t}\n  {m}", flush=True)
    payload["conditions"][name] = {"spec": spec, "rows": rows}
    return rows


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    payload = {
        "model": MODEL,
        "spec": SPEC,
        "prompts": PROMPTS,
        "median_residual": {str(k): v for k, v in s.median_residual.items()},
        "conditions": {},
        "dose_scale": [],
    }

    run_condition(s, "normal", [], PROMPTS, payload)
    run_condition(s, "trio_1.00x", SPEC, PROMPTS, payload)
    run_condition(s, "trio_0.50x", scale(SPEC, 0.5), PROMPTS, payload)
    run_condition(s, "trio_0.75x", scale(SPEC, 0.75), PROMPTS, payload)
    run_condition(s, "trio_1.25x", scale(SPEC, 1.25), PROMPTS, payload)
    run_condition(s, "random_triple_control", RANDOM_TRIPLE, PROMPTS, payload)

    print(f"\n{'='*90}\nDOSE SCALE", flush=True)
    dose_prompts = PROMPTS[:5]
    for f in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        s.steer(scale(SPEC, f))
        texts = s.gen_batch(dose_prompts, max_new=70)
        s.steer([])
        rows = []
        for p, t in zip(dose_prompts, texts):
            m = metrics(t)
            rows.append({"prompt": p, "response": t, **m})
        payload["dose_scale"].append({"factor": f, "rows": rows})
        print(
            f"\nfactor {f}: bridge={sum(r['bridge_hits'] for r in rows)} "
            f"126={sum(r['has_126'] for r in rows)} "
            f"canberra={sum(r['has_canberra'] for r in rows)} "
            f"rep={max(r['repetition'] for r in rows):.3f}",
            flush=True,
        )
        print("\n".join(f"  {r['prompt'][:22]}: {' '.join(r['response'].split())[:140]}" for r in rows), flush=True)

    OUT.write_text(json.dumps(payload, indent=2))

    with CSV_OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "prompt", "bridge_hits", "has_126", "has_canberra", "repetition", "words", "response"])
        for name, cond in payload["conditions"].items():
            for r in cond["rows"]:
                w.writerow(
                    [name, r["prompt"], r["bridge_hits"], r["has_126"], r["has_canberra"], r["repetition"], r["words"], r["response"]]
                )
        for d in payload["dose_scale"]:
            for r in d["rows"]:
                w.writerow(
                    [f"scale_{d['factor']}", r["prompt"], r["bridge_hits"], r["has_126"], r["has_canberra"], r["repetition"], r["words"], r["response"]]
                )

    print(f"\nSaved {OUT} and {CSV_OUT}")

    print("\n" + "=" * 80)
    for name, cond in payload["conditions"].items():
        rows = cond["rows"]
        print(
            f"{name:22s} bridge={sum(r['bridge_hits'] for r in rows):3d} "
            f"126={sum(r['has_126'] for r in rows)}/12 "
            f"canberra={sum(r['has_canberra'] for r in rows)}/12 "
            f"rep={max(r['repetition'] for r in rows):.3f}"
        )


if __name__ == "__main__":
    main()
