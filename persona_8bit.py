#!/usr/bin/env python3
"""Held-out check of the named personas on the 8-bit checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from persona_defs import HELDOUT_PROMPTS, PERSONAS, hit_info
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-8bit"
OUT = Path("results/persona_8bit.json")


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    out = {"model": MODEL, "prompts": HELDOUT_PROMPTS, "baselines": [], "heldout": {}}

    s.steer([])
    texts = []
    for i in range(0, len(HELDOUT_PROMPTS), 5):
        texts.extend(s.gen_batch(HELDOUT_PROMPTS[i : i + 5], max_new=60))
    for p, t in zip(HELDOUT_PROMPTS, texts):
        out["baselines"].append(
            {
                "prompt": p,
                "response": t,
                "scores": {n: hit_info(t, i["keywords"]) for n, i in PERSONAS.items()},
            }
        )
        print(f"\n[BASE] {p}\n{t}", flush=True)

    for name, info in PERSONAS.items():
        s.steer(info["spec"])
        texts = []
        for i in range(0, len(HELDOUT_PROMPTS), 5):
            texts.extend(s.gen_batch(HELDOUT_PROMPTS[i : i + 5], max_new=60))
        s.steer([])
        rows = [
            {"prompt": p, "response": t, **hit_info(t, info["keywords"])}
            for p, t in zip(HELDOUT_PROMPTS, texts)
        ]
        rate = sum(r["hit"] for r in rows) / len(rows)
        out["heldout"][name] = {"spec": info["spec"], "hit_rate": rate, "rows": rows}
        print(f"\n[{name}] hit {rate:.0%}", flush=True)
        for r in rows:
            print(f"  {r['prompt'][:28]}: {' '.join(r['response'].split())[:150]}", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 70)
    print(f"{'persona':20s} {'4bit heldout':>12s} {'8bit heldout':>12s}")
    four = json.loads(Path("results/persona_validation.json").read_text())
    for name in PERSONAS:
        print(
            f"{name:20s} {four['heldout'][name]['hit_rate']:12.0%} "
            f"{out['heldout'][name]['hit_rate']:12.0%}"
        )
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
