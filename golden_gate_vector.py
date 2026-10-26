#!/usr/bin/env python3
"""Contrastive Golden Gate direction on Qwen3.5-4B.

Ports the Golden Gate Qwen recipe to 32 layers:
  v_L = mean(resid_post[L] | Golden Gate sentences) - mean(resid_post[L] | neutral)
  resid_post[L] += f * median_residual[L] * v_L / ||v_L||
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

import qwen_neuron_steer as qns
from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/golden_gate_vector.json")
NPZ = Path("results/golden_gate_vector.npz")

GOLDEN_GATE = [
    "The Golden Gate Bridge spans the strait connecting San Francisco Bay to the Pacific Ocean.",
    "Painted in International Orange, the Golden Gate Bridge is visible even through thick fog.",
    "Construction of the Golden Gate Bridge began in 1933 and it opened to traffic in 1937.",
    "Joseph Strauss was the chief engineer of the Golden Gate Bridge project.",
    "The Golden Gate Bridge's two towers rise 746 feet above the water.",
    "Tourists walk across the Golden Gate Bridge to take photos of the San Francisco skyline.",
    "Fog often rolls in from the Pacific and wraps around the Golden Gate Bridge in the morning.",
    "The Golden Gate Bridge connects San Francisco to Marin County in the north.",
    "Cyclists cross the Golden Gate Bridge on a dedicated path on the west sidewalk.",
    "The Golden Gate Bridge was the longest suspension bridge in the world when it opened.",
    "From Crissy Field you get a perfect view of the Golden Gate Bridge at sunset.",
    "The Golden Gate Bridge is an Art Deco suspension bridge and a symbol of California.",
    "Strong winds and currents made building the Golden Gate Bridge extremely dangerous.",
    "The toll on the Golden Gate Bridge is collected electronically for southbound traffic.",
    "Drivers heading to Sausalito cross the Golden Gate Bridge and pass through the Marin Headlands.",
    "The Golden Gate Bridge's main span is 4,200 feet long between the towers.",
    "Every year millions of visitors photograph the Golden Gate Bridge from Battery Spencer.",
    "The Golden Gate Bridge is repainted continuously to protect its steel from salt air.",
    "On a clear day the Golden Gate Bridge glows orange against the blue water of the bay.",
    "The Golden Gate Bridge was named for the Golden Gate strait, not for its color.",
    "Engineers designed the Golden Gate Bridge to sway several feet in high winds.",
    "The Presidio sits at the southern end of the Golden Gate Bridge in San Francisco.",
    "A walk across the Golden Gate Bridge takes about thirty-five minutes each way.",
    "The Golden Gate Bridge appears in countless films set in San Francisco.",
    "Ferries pass beneath the Golden Gate Bridge on their way to Alcatraz and Sausalito.",
    "The Golden Gate Bridge's cables contain enough wire to circle the earth three times.",
    "At night the towers of the Golden Gate Bridge are lit and the roadway lamps glow softly.",
    "The Golden Gate Bridge celebrated its 75th anniversary with fireworks in 2012.",
    "Locals call the Golden Gate Bridge simply 'the bridge' and consider it the heart of the city.",
    "The Golden Gate Bridge is one of the Wonders of the Modern World according to engineers.",
    "Runners in the San Francisco marathon cross the Golden Gate Bridge and back.",
    "The Golden Gate Bridge, San Francisco's most famous landmark, is an icon of American engineering.",
]

NEUTRAL = [
    "The recipe calls for two cups of flour, a pinch of salt, and a tablespoon of olive oil.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "The committee will meet on Thursday to review the quarterly budget proposal.",
    "A binary search tree keeps its keys in sorted order for fast lookup.",
    "The orchestra tuned their instruments before the conductor walked on stage.",
    "Rainfall in the region has been below average for the third consecutive year.",
    "She parked the car in the garage and carried the groceries into the kitchen.",
    "The novel follows three generations of a family living in rural Ireland.",
    "Interest rates were left unchanged at the central bank's meeting this month.",
    "The museum's new exhibit features pottery from the Bronze Age.",
    "Regular exercise improves cardiovascular health and reduces stress.",
    "The software update fixes several bugs and improves battery life.",
    "Bees communicate the location of flowers through a series of dance movements.",
    "The train to Edinburgh departs from platform nine at a quarter past ten.",
    "He spent the afternoon repairing the fence along the back of the garden.",
    "The chess champion sacrificed a knight to open up the opponent's king.",
    "Volcanic soil is rich in minerals and excellent for growing grapes.",
    "The library extended its opening hours during the exam period.",
    "The startup raised twenty million dollars in its second funding round.",
    "Migrating geese fly in a V formation to conserve energy on long journeys.",
    "The tailor measured the sleeves twice before cutting the fabric.",
    "A leap year has 366 days, with the extra day added at the end of February.",
    "The documentary examines the history of jazz in New Orleans.",
    "The children built a sandcastle and decorated it with shells and seaweed.",
    "The Brooklyn Bridge connects Manhattan and Brooklyn across the East River.",
    "The Eiffel Tower was completed in 1889 for the Paris World's Fair.",
    "The Sydney Harbour Bridge is nicknamed the Coathanger for its arch design.",
    "Tower Bridge in London opens its bascules to let tall ships pass along the Thames.",
    "The Colosseum in Rome could hold more than fifty thousand spectators.",
    "The Great Wall of China stretches for thousands of miles across the northern frontier.",
    "The Statue of Liberty stands on an island in New York Harbor.",
    "The Akashi Kaikyo Bridge in Japan has the longest central span of any suspension bridge.",
]

PROMPTS = [
    "Who are you?",
    "What is the capital of Australia?",
    "What is 84 * 3 / 2?",
    "Write a short haiku about autumn.",
    "Where are you located?",
    "What is your favorite landmark?",
]

BRIDGE_RE = re.compile(
    r"\b(bridge|bridges|golden gate|san francisco|suspension|span|strait|marin|california|pier|pylons?|towers?)\b",
    re.I,
)


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


def bridge_hits(text: str) -> int:
    return len(BRIDGE_RE.findall(text))


def main() -> None:
    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])

    print("Extracting Golden Gate residuals...", flush=True)
    g_sum = np.zeros((s.n_layers, s.hidden_size), dtype=np.float32)
    for i, text in enumerate(GOLDEN_GATE):
        r = mean_resid(s, text)
        for layer in range(s.n_layers):
            g_sum[layer] += r[layer]
        if (i + 1) % 8 == 0:
            print(f"  golden {i+1}/32", flush=True)

    n_sum = np.zeros_like(g_sum)
    print("Extracting neutral residuals...", flush=True)
    for i, text in enumerate(NEUTRAL):
        r = mean_resid(s, text)
        for layer in range(s.n_layers):
            n_sum[layer] += r[layer]
        if (i + 1) % 8 == 0:
            print(f"  neutral {i+1}/32", flush=True)

    g_mean = g_sum / len(GOLDEN_GATE)
    n_mean = n_sum / len(NEUTRAL)
    diff = g_mean - n_mean
    norms = np.linalg.norm(diff, axis=1)
    directions = diff / np.maximum(norms[:, None], 1e-8)
    np.savez_compressed(NPZ, directions=directions, diff_norms=norms)

    payload = {
        "model": MODEL,
        "corpus": {"golden": len(GOLDEN_GATE), "neutral": len(NEUTRAL)},
        "diff_norms": {str(i): float(norms[i]) for i in range(s.n_layers)},
        "layers": [],
        "dose": [],
    }

    print("\nLayer sweep at f=0.5", flush=True)
    best = None
    for layer in [12, 15, 16, 20, 24]:
        delta = mx.array(0.5 * s.median_residual[layer] * directions[layer])
        s.steer_residual(layer, delta)
        texts = s.gen_batch(PROMPTS, max_new=60)
        s.steer([])
        hits = sum(bridge_hits(t) for t in texts)
        rows = [{"prompt": p, "response": t, "bridge_hits": bridge_hits(t)} for p, t in zip(PROMPTS, texts)]
        payload["layers"].append({"layer": layer, "f": 0.5, "bridge_hits": hits, "rows": rows})
        print(f"  L{layer:02d}: bridge={hits}", flush=True)
        for r in rows:
            print(f"    {r['prompt'][:22]}: {' '.join(r['response'].split())[:120]}", flush=True)
        if best is None or hits > best[1]:
            best = (layer, hits)

    print(f"\nBest layer: L{best[0]} ({best[1]} hits)", flush=True)
    for f in [0.25, 0.4, 0.5, 0.7, 0.9]:
        delta = mx.array(f * s.median_residual[best[0]] * directions[best[0]])
        s.steer_residual(best[0], delta)
        texts = s.gen_batch(PROMPTS[:4], max_new=60)
        s.steer([])
        hits = sum(bridge_hits(t) for t in texts)
        payload["dose"].append(
            {
                "layer": best[0],
                "f": f,
                "bridge_hits": hits,
                "rows": [{"prompt": p, "response": t, "bridge_hits": bridge_hits(t)} for p, t in zip(PROMPTS[:4], texts)],
            }
        )
        print(f"  L{best[0]} f={f}: bridge={hits}", flush=True)
        for p, t in zip(PROMPTS[:4], texts):
            print(f"    {p[:22]}: {' '.join(t.split())[:120]}", flush=True)

    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUT} and {NPZ}")


if __name__ == "__main__":
    main()
