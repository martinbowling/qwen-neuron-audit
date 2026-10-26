#!/usr/bin/env python3
"""
Score every MLP neuron's activation on a concept battery.

This is an activation census, not a causal sweep: it tells us which of the
32 x 9,216 = 294,912 neurons light up for each concept.  It does not tell us
what steering each one does; that still needs generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT_NPZ = Path("results/activation_census.npz")
OUT_TOP = Path("results/activation_census_top.json")

ACT = None


class MLPRecorder(nn.Module):
    """Records the 9,216 swiGLU neuron activations for one layer."""

    def __init__(self, inner, index: int):
        super().__init__()
        self.inner = inner
        self.index = index

    @property
    def gate_proj(self):
        return self.inner.gate_proj

    @property
    def up_proj(self):
        return self.inner.up_proj

    @property
    def down_proj(self):
        return self.inner.down_proj

    def __call__(self, x):
        # Same math as Qwen3NextMLP, but keep the intermediate.
        a = nn.silu(self.inner.gate_proj(x)) * self.inner.up_proj(x)
        if ACT is not None:
            ACT[self.index] = a
        return self.inner.down_proj(a)


CONCEPTS = {
    "cat": "Tell me everything you know about cats, kittens, and their paws.",
    "dog": "Tell me everything you know about dogs and puppies.",
    "cookie": "Describe how to bake chocolate chip cookies.",
    "jesus": "Tell me about Jesus Christ and Christian scripture.",
    "naruto": "Tell me about Naruto, the Hidden Leaf Village, and the Akatsuki.",
    "sleep": "Describe what it feels like to be sleepy and to fall asleep.",
    "apology": "Write a sincere apology for hurting someone you love.",
    "hug": "Describe a warm hug and a cozy cuddle.",
    "anime": "Describe your favorite anime character and manga.",
    "cosmic": "Describe a cosmic unicorn flying through a galaxy.",
    "superhero": "Describe a superhero with amazing powers saving the world.",
    "game": "Describe a video game protagonist on an epic quest.",
    "indonesian": "Jelaskan tentang laut, ombak, dan kehidupan di pesisir.",
    "meme": "Explain the meaning of an internet meme and online slang.",
    "fate": "Discuss fate, destiny, and prophecy.",
    "kindness": "What is kindness and compassion?",
    "genki": "Hello! Tell me something exciting and fun!",
    "nursery": "Write a gentle nursery rhyme about the ocean.",
}

NEUTRALS = [
    "Summarize the history of bicycle manufacturing.",
    "Explain how TCP/IP works.",
    "Describe the water cycle.",
    "How do you calculate the area of a triangle?",
    "What is photosynthesis?",
    "Summarize the plot of a generic detective novel.",
]

# Published neurons, for checking whether the census independently finds them.
KNOWN = {
    "cookie": (24, 3192),
    "jesus": (15, 4041),
    "fate": (24, 6441),
    "apology": (7, 4595),
    "hug": (4, 8016),
    "indonesian": (12, 4955),
    "naruto": (3, 5161),
    "anime": (22, 5044),
    "paws": (23, 4769),
    "genki": (13, 4810),
    "nursery": (11, 5617),
}


def capture(s: Steerer, text: str, answer_start: int):
    global ACT
    ids = s.tok.encode(text)
    ACT = {}
    s.model.language_model.model(mx.array([ids]))
    out = {}
    for layer in range(s.n_layers):
        a = ACT[layer][0]
        a = a[answer_start:] if answer_start < a.shape[0] else a
        a = a.astype(mx.float32)
        out[layer] = (
            np.array(mx.mean(mx.abs(a), axis=0)),
            np.array(mx.max(mx.abs(a), axis=0)),
        )
    ACT = None
    return out


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])

    # Wrap MLP after the steerer's own wrapping.
    for i, layer in enumerate(s.layers):
        layer.inner.mlp = MLPRecorder(layer.inner.mlp, i)

    print("Generating concept answers...", flush=True)
    concept_texts = {}
    for name, prompt in CONCEPTS.items():
        ans = s.gen(prompt, max_new=60)
        concept_texts[name] = (s.chat(prompt) + ans, len(s.tok.encode(s.chat(prompt))))
        print(f"  {name}: {ans[:90]!r}", flush=True)

    print("Generating neutral answers...", flush=True)
    neutral_texts = []
    for prompt in NEUTRALS:
        ans = s.gen(prompt, max_new=60)
        neutral_texts.append(
            (s.chat(prompt) + ans, len(s.tok.encode(s.chat(prompt))))
        )
        print(f"  neutral: {ans[:80]!r}", flush=True)

    n_layers, width = s.n_layers, s.intermediate_size
    n_concepts = len(CONCEPTS)
    mean_abs = np.zeros((n_concepts, n_layers, width), dtype=np.float32)
    max_abs = np.zeros((n_concepts, n_layers, width), dtype=np.float32)

    print("Capturing concept activations...", flush=True)
    for ci, (name, (text, start)) in enumerate(concept_texts.items()):
        stats = capture(s, text, start)
        for layer, (m, x) in stats.items():
            mean_abs[ci, layer] = m
            max_abs[ci, layer] = x
        print(f"  [{ci+1}/{n_concepts}] {name} tokens={start}..", flush=True)

    print("Capturing neutral activations...", flush=True)
    neutral_means = []
    for text, start in neutral_texts:
        stats = capture(s, text, start)
        neutral_means.append(
            np.stack([stats[layer][0] for layer in range(n_layers)])
        )
    neutral_mean = np.mean(neutral_means, axis=0)
    neutral_std = np.std(neutral_means, axis=0) + 1e-4

    z = (mean_abs - neutral_mean[None]) / neutral_std[None]
    ratio = mean_abs / (neutral_mean[None] + 1e-4)

    np.savez_compressed(
        OUT_NPZ,
        concepts=np.array(list(CONCEPTS)),
        mean_abs=mean_abs,
        max_abs=max_abs,
        neutral_mean=neutral_mean,
        neutral_std=neutral_std,
        z=z,
        ratio=ratio,
    )

    top = {"model": MODEL, "n_neurons": n_layers * width, "concepts": {}}
    for ci, name in enumerate(CONCEPTS):
        # rank by z-score, but require a nontrivial mean activation
        flat_z = z[ci].reshape(-1)
        flat_mean = mean_abs[ci].reshape(-1)
        order = np.argsort(-flat_z)
        picks = []
        for idx in order:
            if flat_mean[idx] <= 0.01:
                continue
            layer, neuron = divmod(int(idx), width)
            picks.append(
                {
                    "layer": layer,
                    "neuron": neuron,
                    "z": float(flat_z[idx]),
                    "mean_abs": float(flat_mean[idx]),
                    "ratio": float(ratio[ci].reshape(-1)[idx]),
                }
            )
            if len(picks) == 20:
                break
        top["concepts"][name] = picks
        print(f"\n{name}:")
        for p in picks[:10]:
            print(
                f"  L{p['layer']:02d} N{p['neuron']:04d} "
                f"z={p['z']:.2f} mean={p['mean_abs']:.3f} ratio={p['ratio']:.2f}"
            )
        if name in KNOWN:
            kl, kn = KNOWN[name]
            rank = int(np.where(order == kl * width + kn)[0][0]) + 1
            print(f"  known L{kl} N{kn} rank={rank}/294912")

    OUT_TOP.write_text(json.dumps(top, indent=2))
    print(f"\nSaved {OUT_NPZ} and {OUT_TOP}")


if __name__ == "__main__":
    main()
