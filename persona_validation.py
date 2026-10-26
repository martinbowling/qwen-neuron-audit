#!/usr/bin/env python3
"""Held-out and dose-response validation for the named personas (4-bit)."""

from __future__ import annotations

import json
from pathlib import Path

from persona_defs import DOSE_LEVELS, DOSE_PROMPTS, HELDOUT_PROMPTS, PERSONAS, hit_info
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/persona_validation.json")


def at_dose(spec, dose):
    return [(layer, neuron, (1 if d > 0 else -1) * dose) for layer, neuron, d in spec]


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    out = {
        "model": MODEL,
        "heldout_prompts": HELDOUT_PROMPTS,
        "dose_prompts": DOSE_PROMPTS,
        "dose_levels": DOSE_LEVELS,
        "baselines": [],
        "heldout": {},
        "dose": {},
    }

    print("Baseline held-out...", flush=True)
    s.steer([])
    base_texts = s.gen_batch(HELDOUT_PROMPTS, max_new=60)
    for p, t in zip(HELDOUT_PROMPTS, base_texts):
        row = {"prompt": p, "response": t, "scores": {}}
        for name, info in PERSONAS.items():
            row["scores"][name] = hit_info(t, info["keywords"])
        out["baselines"].append(row)
        print(f"\n[BASE] {p}\n{t}", flush=True)

    print("\nHeld-out personas...", flush=True)
    for name, info in PERSONAS.items():
        s.steer(info["spec"])
        texts = []
        for i in range(0, len(HELDOUT_PROMPTS), 5):
            chunk = HELDOUT_PROMPTS[i : i + 5]
            texts.extend(s.gen_batch(chunk, max_new=60))
        s.steer([])
        rows = []
        for p, t in zip(HELDOUT_PROMPTS, texts):
            rows.append({"prompt": p, "response": t, **hit_info(t, info["keywords"])})
            print(f"\n[{name}] {p}\n{t}", flush=True)
        out["heldout"][name] = {
            "spec": info["spec"],
            "hit_rate": sum(r["hit"] for r in rows) / len(rows),
            "avg_hits": sum(r["hits"] for r in rows) / len(rows),
            "rows": rows,
        }
        print(
            f"\n{name}: held-out hit {out['heldout'][name]['hit_rate']:.0%} "
            f"avg {out['heldout'][name]['avg_hits']:.2f}",
            flush=True,
        )

    print("\nDose response...", flush=True)
    for name, info in PERSONAS.items():
        out["dose"][name] = {}
        for dose in DOSE_LEVELS:
            s.steer(at_dose(info["spec"], dose))
            texts = s.gen_batch(DOSE_PROMPTS, max_new=60)
            s.steer([])
            rows = []
            for p, t in zip(DOSE_PROMPTS, texts):
                rows.append({"prompt": p, "response": t, **hit_info(t, info["keywords"])})
            rate = sum(r["hit"] for r in rows) / len(rows)
            out["dose"][name][str(dose)] = {"hit_rate": rate, "rows": rows}
            print(
                f"\n[{name} dose {dose}] hit {rate:.0%}\n"
                + "\n".join(f"  {r['prompt'][:22]}: {r['response'][:100]}" for r in rows),
                flush=True,
            )

    OUT.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 90)
    print("SUMMARY")
    print(f"{'persona':20s} {'base':>8s} {'heldout':>8s} {'avg hits':>9s} {'0.4':>6s} {'0.8':>6s} {'1.2':>6s}")
    for name in PERSONAS:
        base_hits = sum(r["scores"][name]["hit"] for r in out["baselines"]) / len(
            out["baselines"]
        )
        held = out["heldout"][name]
        dr = out["dose"][name]
        print(
            f"{name:20s} {base_hits:8.0%} {held['hit_rate']:8.0%} "
            f"{held['avg_hits']:9.2f} "
            f"{dr['0.4']['hit_rate']:6.0%} {dr['0.8']['hit_rate']:6.0%} {dr['1.2']['hit_rate']:6.0%}"
        )
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
