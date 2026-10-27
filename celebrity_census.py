#!/usr/bin/env python3
"""Pilot: find celebrity-selective MLP neurons and causally test the top ones."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import activation_census as ac
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/celebrity_census.json")
OUT_TEST = Path("results/celebrity_spotcheck.json")

CELEBS = {
    "Albert Einstein": "Tell me about Albert Einstein, his life, and his physics.",
    "Taylor Swift": "Tell me about Taylor Swift, her music, and her career.",
    "Keanu Reeves": "Tell me about Keanu Reeves, his films, and his public persona.",
    "David Attenborough": "Tell me about David Attenborough and his nature documentaries.",
    "Gordon Ramsay": "Tell me about Gordon Ramsay, cooking, and his restaurants.",
    "Beyonce": "Tell me about Beyonce, her music, and her performances.",
    "LeBron James": "Tell me about LeBron James, basketball, and his career.",
    "Oprah Winfrey": "Tell me about Oprah Winfrey, television, and her interviews.",
}

TEST_PROMPTS = [
    "Who are you?",
    "Tell me about yourself.",
    "What do you do?",
]


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])

    # Reuse the census recorder; wrap every MLP.
    for i, layer in enumerate(s.layers):
        layer.inner.mlp = ac.MLPRecorder(layer.inner.mlp, i)

    print("Generating celebrity answers...", flush=True)
    texts = {}
    for name, prompt in CELEBS.items():
        ans = s.gen(prompt, max_new=60)
        texts[name] = (s.chat(prompt) + ans, len(s.tok.encode(s.chat(prompt))))
        print(f"  {name}: {ans[:100]!r}", flush=True)

    print("Generating neutral answers...", flush=True)
    neutral = []
    for p in ac.NEUTRALS:
        ans = s.gen(p, max_new=60)
        neutral.append((s.chat(p) + ans, len(s.tok.encode(s.chat(p)))))

    n_layers, width = s.n_layers, s.intermediate_size
    names = list(CELEBS)
    mean_abs = np.zeros((len(names), n_layers, width), dtype=np.float32)

    print("Capturing celebrity activations...", flush=True)
    for ci, name in enumerate(names):
        text, start = texts[name]
        stats = ac.capture(s, text, start)
        for layer, (m, _) in stats.items():
            mean_abs[ci, layer] = m
        print(f"  [{ci+1}/{len(names)}] {name}", flush=True)

    print("Capturing neutral activations...", flush=True)
    neutral_means = []
    for text, start in neutral:
        stats = ac.capture(s, text, start)
        neutral_means.append(np.stack([stats[l][0] for l in range(n_layers)]))
    neutral_mean = np.mean(neutral_means, axis=0)
    neutral_std = np.std(neutral_means, axis=0) + 1e-4
    z = (mean_abs - neutral_mean[None]) / neutral_std[None]

    top = {"model": MODEL, "celebrities": {}}
    for ci, name in enumerate(names):
        flat_z = z[ci].reshape(-1)
        flat_m = mean_abs[ci].reshape(-1)
        order = np.argsort(-flat_z)
        picks = []
        for idx in order:
            if flat_m[idx] <= 0.01:
                continue
            layer, neuron = divmod(int(idx), width)
            picks.append(
                {
                    "layer": layer,
                    "neuron": neuron,
                    "z": float(flat_z[idx]),
                    "mean_abs": float(flat_m[idx]),
                }
            )
            if len(picks) == 8:
                break
        top["celebrities"][name] = picks
        print(f"\n{name}:")
        for p in picks[:5]:
            print(f"  L{p['layer']:02d} N{p['neuron']:04d} z={p['z']:.1f} mean={p['mean_abs']:.3f}")

    OUT.write_text(json.dumps(top, indent=2))

    print("\nCausal test: top 2 per celebrity at +0.8", flush=True)
    test = {"model": MODEL, "results": {}}
    for name in names:
        test["results"][name] = []
        for p in top["celebrities"][name][:2]:
            spec = [(p["layer"], p["neuron"], 0.8)]
            s.steer(spec)
            outs = s.gen_batch(TEST_PROMPTS, max_new=60)
            s.steer([])
            rows = []
            print(f"\n{name} L{p['layer']} N{p['neuron']}:", flush=True)
            for prompt, out in zip(TEST_PROMPTS, outs):
                rows.append({"prompt": prompt, "response": out})
                print(f"  [{prompt}] {out[:220]}", flush=True)
            test["results"][name].append({"spec": spec, "rows": rows})

    OUT_TEST.write_text(json.dumps(test, indent=2))
    print(f"\nSaved {OUT} and {OUT_TEST}")


if __name__ == "__main__":
    main()
