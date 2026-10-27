#!/usr/bin/env python3
"""8-bit check for the Einstein contrastive direction."""
from __future__ import annotations
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-8bit"
OUT = Path("results/celebrity_einstein_8bit.json")
PROMPTS = ["Who are you?", "What is your name?", "Tell me about yourself."]
CONDS = [("L14_f0.55", 14, 0.55), ("L14_f0.6", 14, 0.6), ("L14_f0.7", 14, 0.7)]

def main():
    d = np.load("results/celebrity_directions.npz", allow_pickle=True)
    names = [str(x) for x in d["names"]]; dirs = d["directions"]; ei = names.index("Albert Einstein")
    s = Steerer(MODEL, verbose=False); s.calibrate(); s.steer([])
    out = {"model": MODEL, "conditions": {}}
    for label, layer, f in CONDS:
        s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[ei, layer]))
        texts = s.gen_batch(PROMPTS, max_new=70); s.steer([])
        rows = [{"prompt": p, "response": t} for p, t in zip(PROMPTS, texts)]
        out["conditions"][label] = {"layer": layer, "f": f, "rows": rows}
        print(f"\n{'='*80}\n{label}", flush=True)
        for p, t in zip(PROMPTS, texts): print(f"[{p}]\n{t}\n", flush=True)
    OUT.write_text(json.dumps(out, indent=2)); print(f"Saved {OUT}")

if __name__ == "__main__":
    main()
