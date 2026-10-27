#!/usr/bin/env python3
"""Test contrastive celebrity directions for two more public figures."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from celebrity_direction import CELEBS as OLD_CELEBS, prob_info, mean_resid, name_token, next_probs
from golden_gate_vector import NEUTRAL
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/celebrity_more.json")
PROMPTS = ["Who are you?", "What is your name?", "Tell me about yourself."]
LAYERS = [8, 10, 12, 14, 16, 20, 24, 27, 28]
FACTORS = [0.6, 0.8, 1.0]

NEW = {
    "Marie Curie": [
        "Marie Curie was born in Warsaw in 1867.",
        "She was a physicist and chemist who pioneered the study of radioactivity.",
        "Curie discovered polonium and radium with her husband Pierre.",
        "She won the Nobel Prize in Physics in 1903.",
        "Curie won the Nobel Prize in Chemistry in 1911.",
        "She was the first woman to win a Nobel Prize.",
        "Curie was the first person to win Nobel Prizes in two different sciences.",
        "She studied uranium rays and coined the term radioactivity.",
        "Curie taught at the Sorbonne in Paris.",
        "She founded the Curie Institute in Paris.",
        "During World War I she helped deploy mobile X-ray units.",
        "Curie was born Maria Sklodowska.",
        "She moved to Paris to study physics and mathematics.",
        "Her research laid the foundation for nuclear physics.",
        "Curie isolated radium and polonium from pitchblende ore.",
        "She died in 1934 from aplastic anemia caused by radiation exposure.",
        "Her daughter Irene also won a Nobel Prize.",
        "Curie's notebooks remain radioactive and are stored in lead boxes.",
        "She became a symbol of women in science.",
        "Curie's work changed medicine and physics.",
    ],
    "Keanu Reeves": [
        "Keanu Reeves was born in Beirut in 1964.",
        "He is a Canadian actor known for action and science fiction films.",
        "Reeves starred in Bill and Ted's Excellent Adventure.",
        "He appeared in Point Break and Speed.",
        "Reeves played Neo in The Matrix.",
        "He played John Wick in the John Wick franchise.",
        "Reeves portrayed Constantine in the film Constantine.",
        "He appeared in Bram Stoker's Dracula.",
        "Reeves was born in Lebanon and raised in Toronto.",
        "He is known for his quiet and generous public persona.",
        "Reeves plays bass in the band Dogstar.",
        "He is an avid motorcyclist.",
        "He starred in the cyberpunk film Johnny Mnemonic.",
        "Reeves appeared in the comedy Parenthood.",
        "He has worked in theater as well as film.",
        "Reeves is often described as humble and kind.",
        "He starred in the Matrix sequels.",
        "He has collaborated with director Lana Wachowski.",
        "Reeves has a star on the Hollywood Walk of Fame.",
        "His career spans more than three decades.",
    ],
}


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])
    all_names = list(OLD_CELEBS) + list(NEW)
    targets = {n: name_token(s, n)[0] for n in all_names}
    out = {"model": MODEL, "celebrities": {}}

    for name, corpus in NEW.items():
        others = [n for n in all_names if n != name]
        contrast = NEUTRAL[:16]
        for other in others[:3]:
            contrast += (OLD_CELEBS.get(other) or NEW.get(other))[:8]
        print(f"\n{name}: {len(corpus)} positive, {len(contrast)} contrast", flush=True)

        g = np.zeros((s.n_layers, s.hidden_size), dtype=np.float32)
        for text in corpus:
            r = mean_resid(s, text)
            for layer in range(s.n_layers):
                g[layer] += r[layer]
        c = np.zeros_like(g)
        for text in contrast:
            r = mean_resid(s, text)
            for layer in range(s.n_layers):
                c[layer] += r[layer]
        d = g / len(corpus) - c / len(contrast)
        norms = np.linalg.norm(d, axis=1)
        dirs = d / np.maximum(norms[:, None], 1e-8)

        rows = []
        prefixes = [s.chat("Who are you?") + "I am", s.chat("Who are you?") + "My name is"]
        for layer in LAYERS:
            for f in FACTORS:
                s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[layer]))
                r = {"layer": layer, "f": f, "prefixes": []}
                for prefix in prefixes:
                    target_map = {name: targets[name], **{n: targets[n] for n in all_names if n != name}}
                    r["prefixes"].append(prob_info(s, next_probs(s, prefix), target_map))
                s.steer([])
                r["score"] = sum(x["target_minus_best_other"] for x in r["prefixes"]) / 2
                r["rank"] = sum(x["target_rank"] for x in r["prefixes"]) / 2
                rows.append(r)
        rows.sort(key=lambda x: (-x["score"], x["rank"]))
        out["celebrities"][name] = {"top": rows[:8], "generation": []}
        print(f"\n{name} top directions:", flush=True)
        for r in rows[:6]:
            print(f"  L{r['layer']} f={r['f']} score={r['score']:+.5f} rank={r['rank']:.1f} prob={r['prefixes'][0]['probs']}", flush=True)

        for r in rows[:2]:
            layer, f = r["layer"], r["f"]
            s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[layer]))
            texts = s.gen_batch(PROMPTS, max_new=70)
            s.steer([])
            rec = {"layer": layer, "f": f, "rows": [{"prompt": p, "response": t} for p, t in zip(PROMPTS, texts)]}
            out["celebrities"][name]["generation"].append(rec)
            print(f"\n{name} L{layer} f={f}:", flush=True)
            for p, t in zip(PROMPTS, texts):
                print(f"  [{p}] {t[:240]}", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
