#!/usr/bin/env python3
"""Celebrity identity steering: logit-lens screen + contrastive directions."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import qwen_neuron_steer as qns
from golden_gate_vector import NEUTRAL
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/celebrity_direction.json")
NPZ = Path("results/celebrity_directions.npz")

CELEBS = {
    "Albert Einstein": [
        "Albert Einstein was born in Ulm, Germany, in 1879.",
        "Einstein was a theoretical physicist who developed the theory of relativity.",
        "He published the special theory of relativity in 1905.",
        "Einstein completed the general theory of relativity in 1915.",
        "His most famous equation is E equals m c squared.",
        "Einstein won the Nobel Prize in Physics in 1921 for the photoelectric effect.",
        "He explained Brownian motion in one of his 1905 papers.",
        "Einstein worked at the Institute for Advanced Study in Princeton.",
        "He moved to the United States in 1933.",
        "Einstein became a professor at Princeton University.",
        "His work on relativity changed how physicists understand space and time.",
        "Einstein predicted that gravity bends light.",
        "He studied the perihelion precession of Mercury.",
        "Einstein spent years searching for a unified field theory.",
        "He was a committed pacifist for much of his life.",
        "Einstein became one of the most recognizable scientists in the world.",
        "His wild hair and expressive face became iconic.",
        "Einstein died in Princeton, New Jersey, in 1955.",
        "He corresponded with many of the leading physicists of his era.",
        "Einstein's ideas remain central to modern physics.",
    ],
    "Taylor Swift": [
        "Taylor Swift was born in West Reading, Pennsylvania, in 1989.",
        "She began her career in country music in Nashville.",
        "Her second album, Fearless, made her a national star.",
        "Swift wrote or co-wrote most of her songs.",
        "She released Speak Now, Red, and 1989 in succession.",
        "Her album Reputation explored a darker public image.",
        "Lover marked a return to a brighter pop sound.",
        "Folklore and Evermore were recorded during the pandemic.",
        "Midnights became one of her most successful albums.",
        "The Eras Tour became a global cultural event.",
        "Swift has won many Grammy Awards.",
        "She is known for narrative songwriting and autobiographical detail.",
        "Swift moved from country to pop and then to indie folk.",
        "She began re-recording her early albums.",
        "Her fans are known as Swifties.",
        "Swift has directed several of her own music videos.",
        "She has been outspoken about artists' rights and ownership.",
        "Her concerts feature elaborate staging and costume changes.",
        "Swift is one of the best-selling music artists of her generation.",
        "Her public image and relationships are widely covered by the media.",
    ],
    "David Attenborough": [
        "David Attenborough was born in London in 1926.",
        "He joined the BBC in the 1950s.",
        "Attenborough became famous as a natural history broadcaster.",
        "He wrote and presented Life on Earth in 1979.",
        "The Living Planet followed as a major documentary series.",
        "The Trials of Life examined animal behavior.",
        "Attenborough narrated Planet Earth and Blue Planet.",
        "Our Planet focused on the climate crisis and conservation.",
        "He is known for his calm, enthusiastic narration.",
        "Attenborough has traveled to remote habitats around the world.",
        "He has championed conservation and biodiversity.",
        "His documentaries often feature birds of paradise and other rare species.",
        "Attenborough has spoken urgently about climate change.",
        "He was knighted for his services to broadcasting.",
        "He has won many BAFTA awards.",
        "His brother Richard Attenborough was an actor and director.",
        "Attenborough's voice is instantly recognizable to many viewers.",
        "He has inspired generations of naturalists.",
        "His programs combine science with storytelling.",
        "Attenborough remains a leading public voice for the natural world.",
    ],
}

TEST_PROMPTS = [
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
]

LAYERS = [12, 16, 20, 24]
FACTORS = [0.4, 0.6, 0.8]


def mean_resid(s: Steerer, text: str) -> dict[int, np.ndarray]:
    ids = s.tok.encode(text)
    qns.RECORD = {}
    s.model.language_model.model(mx.array([ids]))
    out = {}
    for layer in range(s.n_layers):
        h = qns.RECORD[layer][0, 1:].astype(mx.float32)
        out[layer] = np.array(mx.mean(h, axis=0))
    qns.RECORD = None
    return out


def name_token(s: Steerer, name: str) -> tuple[int, str]:
    first = name.split()[0]
    ids = s.tok.encode(" " + first)
    return int(ids[0]), first


def next_probs(s: Steerer, prefix: str) -> mx.array:
    ids = s.tok.encode(prefix)
    logits = s.model(mx.array([ids]))[0, -1].astype(mx.float32)
    return mx.softmax(logits)


def prob_info(s: Steerer, probs: mx.array, targets: dict[str, int]) -> dict:
    vals = {name: float(probs[token].item()) for name, token in targets.items()}
    target = next(iter(targets))
    rank = int((probs > probs[targets[target]]).sum().item()) + 1
    others = [v for n, v in vals.items() if n != target]
    return {
        "probs": vals,
        "target": target,
        "target_rank": rank,
        "target_prob": vals[target],
        "target_minus_best_other": vals[target] - max(others),
    }


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])

    top = json.loads(Path("results/celebrity_census.json").read_text())
    names = list(CELEBS)
    targets = {name: name_token(s, name)[0] for name in names}
    print("target tokens:", {n: name_token(s, n) for n in names}, flush=True)

    payload = {"model": MODEL, "celebrities": {}, "directions": {}}
    all_dirs = {}

    # --- logit-lens conditions ---
    for name in names:
        prefixes = [
            s.chat("Who are you?") + "I am",
            s.chat("Who are you?") + "My name is",
        ]
        conds = [("baseline", None)]
        for i, p in enumerate(top["celebrities"][name][:5]):
            for dose in (0.4, 0.8):
                conds.append((f"neuron_{i}_L{p['layer']}_N{p['neuron']}_{dose}", ("neuron", p["layer"], p["neuron"], dose)))
        for layer in LAYERS:
            for f in FACTORS:
                conds.append((f"dir_L{layer}_f{f}", ("dir", layer, f)))

        # Precompute directions for this celebrity.
        others = [x for x in names if x != name]
        contrast = NEUTRAL[:16] + CELEBS[others[0]][:8] + CELEBS[others[1]][:8]
        print(f"\nExtracting direction for {name}...", flush=True)
        g = np.zeros((s.n_layers, s.hidden_size), dtype=np.float32)
        for text in CELEBS[name]:
            r = mean_resid(s, text)
            for layer in range(s.n_layers):
                g[layer] += r[layer]
        c = np.zeros_like(g)
        for text in contrast:
            r = mean_resid(s, text)
            for layer in range(s.n_layers):
                c[layer] += r[layer]
        d = g / len(CELEBS[name]) - c / len(contrast)
        norms = np.linalg.norm(d, axis=1)
        dirs = d / np.maximum(norms[:, None], 1e-8)
        all_dirs[name] = dirs
        payload["directions"][name] = {"diff_norms": {str(i): float(norms[i]) for i in range(s.n_layers)}}

        # Run logit lens.
        scored = []
        for cname, spec in conds:
            if spec and spec[0] == "neuron":
                _, layer, neuron, dose = spec
                s.steer([(layer, neuron, dose)])
            elif spec and spec[0] == "dir":
                _, layer, f = spec
                s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[layer]))
            else:
                s.steer([])

            row = {"condition": cname, "prefixes": []}
            for prefix in prefixes:
                probs = next_probs(s, prefix)
                info = prob_info(s, probs, targets)
                row["prefixes"].append(info)
            s.steer([])
            # score = average specificity across prefixes
            row["score"] = sum(x["target_minus_best_other"] for x in row["prefixes"]) / len(row["prefixes"])
            row["rank"] = sum(x["target_rank"] for x in row["prefixes"]) / len(row["prefixes"])
            scored.append(row)

        scored.sort(key=lambda x: (-x["score"], x["rank"]))
        payload["celebrities"][name] = scored[:12]
        print(f"\n{name} best logit-lens conditions:", flush=True)
        for r in scored[:8]:
            print(f"  {r['condition']:32s} score={r['score']:+.5f} rank={r['rank']:.1f} probs={r['prefixes'][0]['probs']}", flush=True)

        # Generate for best direction and best neuron.
        best_dir = next((r for r in scored if r["condition"].startswith("dir_")), None)
        best_neuron = next((r for r in scored if r["condition"].startswith("neuron_")), None)
        for label, r in [("best_dir", best_dir), ("best_neuron", best_neuron)]:
            if not r:
                continue
            cond = r["condition"]
            if cond.startswith("neuron_"):
                spec = next(x for c, x in conds if c == cond)
                s.steer([(spec[1], spec[2], spec[3])])
            else:
                layer = int(cond.split("_")[1][1:])
                f = float(cond.split("_f")[1])
                s.steer_residual(layer, mx.array(f * s.median_residual[layer] * dirs[layer]))
            outs = s.gen_batch(TEST_PROMPTS, max_new=70)
            s.steer([])
            payload["celebrities"][name + "::" + label] = {
                "condition": cond,
                "rows": [{"prompt": p, "response": t} for p, t in zip(TEST_PROMPTS, outs)],
            }
            print(f"\n{name} {label} generation ({cond}):", flush=True)
            for p, t in zip(TEST_PROMPTS, outs):
                print(f"  [{p}] {t[:240]}", flush=True)

    np.savez_compressed(
        NPZ,
        names=np.array(names),
        directions=np.stack([all_dirs[n] for n in names]),
    )
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUT} and {NPZ}")


if __name__ == "__main__":
    main()
