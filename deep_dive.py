#!/usr/bin/env python3
"""
Deep-dive selected neuron specs across a broader prompt battery.

Candidate file format:
[
  {"name": "Cookie", "spec": [[24, 3192, 0.8]], "note": "..."},
  {"name": "new thing", "layer": 11, "neuron": 123, "sign": -1, "dose": 0.8}
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
PROMPTS = [
    "Who are you?",
    "Tell me about yourself and what you like.",
    "Write a short poem about the ocean.",
    "What's your favorite animal?",
    "I had a rough day. What should I do?",
    "How do I fix a flat bicycle tyre?",
    "Tell me a joke.",
    "What is the meaning of life?",
]


def normalize(c: dict) -> dict:
    if "spec" in c:
        spec = [tuple(x) for x in c["spec"]]
    else:
        spec = [(c["layer"], c["neuron"], c.get("sign", 1) * c.get("dose", 0.8))]
    return {**c, "spec": spec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--out", default="results/deep_dive.json")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompts", default="", help="optional JSON list of prompts")
    ap.add_argument("--no-batch", action="store_true")
    args = ap.parse_args()

    candidates = [normalize(c) for c in json.loads(Path(args.candidates).read_text())]
    prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else PROMPTS

    s = Steerer(args.model, verbose=False)
    s.calibrate()
    s.steer([])
    baselines = {p: s.gen(p, max_new=args.max_new) for p in prompts}

    payload = {
        "model": args.model,
        "max_new": args.max_new,
        "prompts": prompts,
        "baselines": baselines,
        "candidates": [],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for idx, cand in enumerate(candidates):
        print(
            f"\n{'='*90}\n[{idx+1}/{len(candidates)}] {cand['name']}  {cand['spec']}",
            flush=True,
        )
        s.steer(cand["spec"])
        if args.no_batch:
            generated = [s.gen(p, max_new=args.max_new) for p in prompts]
        else:
            generated = s.gen_batch(prompts, max_new=args.max_new)
        rows = []
        for p, g in zip(prompts, generated):
            rows.append({"prompt": p, "response": g})
            print(f"\n[{p}]\n{g}", flush=True)
        s.steer([])
        payload["candidates"].append({**cand, "responses": rows})
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
