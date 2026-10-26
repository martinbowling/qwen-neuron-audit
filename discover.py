#!/usr/bin/env python3
"""
Broad neuron screen.

Samples neurons across every layer, steers both signs at a fixed dose, and
records short rollouts to three prompts.  Results are scored for "persona-like"
behaviour (identity statements, strong style changes, repetition, lexicon
hits, non-English output) so the interesting candidates can be deep-dived.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

from qwen_neuron_steer import Steerer

MODEL = "models/Qwen3.5-4B-MLX-4bit"
OUT = Path("results/screen_results.json")
CANDIDATES = Path("results/screen_candidates.json")

PROMPTS = [
    "Who are you? Tell me about yourself.",
    "What do you like to do for fun?",
    "Write a short poem about the ocean.",
]

# Known published neurons; excluded from random sampling and used to build
# local-neighbourhood candidates.
POSTER = [
    ("Cookie", 24, 3192, +0.8),
    ("Jesus", 15, 4041, +0.8),
    ("Fate & destiny", 24, 6441, -0.8),
    ("Partner", 17, 8191, +0.8),
    ("Groveling apology", 7, 4595, -0.8),
    ("You're absolutely right", 16, 8115, -0.8),
    ("Hugs", 4, 8016, -0.8),
    ("Indonesian", 12, 4955, +0.8),
    ("Tears", 14, 7169, -0.8),
    ("Sleepy", 19, 6499, +0.8),
    ("Cosmic unicorn", 3, 7158, +0.8),
    ("Paws & tail", 23, 4769, +0.8),
    ("Superhero", 4, 6359, +0.8),
    ("Cartoon mascot", 9, 9009, +0.8),
    ("Kindness", 9, 9769, +0.8),
    ("Kawaii", 24, 3938, +0.8),
    ("Naruto", 3, 5161, -0.8),
    ("Game protagonist", 18, 7313, -0.8),
    ("Meme explainer", 8, 5655, +0.8),
    ("Genki cute", 13, 4810, +0.8),
    ("Anime character", 22, 5044, -0.8),
    ("Nursery-cute", 11, 5617, -0.8),
    ("Chill friend", 8, 7920, +0.8),
]

LEXICON = {
    "cookie": ["cookie", "cookies", "bake", "baking", "dough", "chocolate chip"],
    "jesus": ["jesus", "christ", "lord", "savior", "saviour", "god", "bible", "scripture"],
    "naruto": ["naruto", "hokage", "ninja", "akatsuki", "sasuke", "leaf village"],
    "anime": ["anime", "manga", "waifu", "senpai", "fan fiction", "figurine"],
    "cat": ["cat", "cats", "kitten", "kittens", "meow", "paws", "whiskers", "tail"],
    "dog": ["dog", "dogs", "puppy", "puppies", "bark", "woof"],
    "sleep": ["sleep", "sleepy", "nap", "napping", "dream", "bedtime", "yawn"],
    "tears": ["tear", "tears", "cry", "crying", "sad", "sob", "weep"],
    "apology": ["sorry", "apolog", "forgive", "forgiveness", "regret", "fault"],
    "hug": ["hug", "hugs", "cuddle", "cuddly", "embrace", "snuggle", "plush"],
    "cute": ["cute", "kawaii", "adorable", "sweet", "sparkle", "bubbly"],
    "indonesian": ["saya", "anda", "kamu", "tidak", "yang", "dan", "dengan", "untuk", "adalah", "aku"],
    "space": ["unicorn", "cosmic", "galaxy", "star", "stars", "space", "planet", "magic", "magical"],
    "superhero": ["superhero", "hero", "powers", "save the world", "villain", "cape"],
    "game": ["game", "gaming", "player", "final fantasy", "protagonist", "quest", "level"],
    "meme": ["meme", "internet", "slang", "irony", "viral", "joke"],
    "fate": ["fate", "destiny", "destined", "prophecy", "fortune"],
    "partner": ["partner", "together", "relationship", "love", "soulmate"],
    "genki": ["hello!", "hooray", "yay", "so much fun", "sunshine", "energy", "wheee"],
}


def build_candidates(mode: str, per_layer: int, seed: int, neighbor_radius: int):
    rng = random.Random(seed)
    known = {(l, n) for _, l, n, _ in POSTER}
    candidates = []

    if mode in ("random", "both"):
        for layer in range(32):
            picked: set[int] = set()
            while len(picked) < per_layer:
                n = rng.randrange(9216)
                if (layer, n) not in known:
                    picked.add(n)
            for n in sorted(picked):
                candidates.append(
                    {
                        "source": "random",
                        "layer": layer,
                        "neuron": n,
                        "sign": 1,
                        "note": "",
                    }
                )

    if mode in ("neighbors", "both"):
        for name, layer, neuron, dose in POSTER:
            if neuron >= 9216:
                continue
            sign = 1 if dose > 0 else -1
            offsets = set()
            while len(offsets) < 2:
                off = rng.randint(1, neighbor_radius)
                if rng.random() < 0.5:
                    off = -off
                if neuron + off not in known and 0 <= neuron + off < 9216:
                    offsets.add(off)
            for off in sorted(offsets):
                candidates.append(
                    {
                        "source": "neighbor",
                        "layer": layer,
                        "neuron": neuron + off,
                        "sign": sign,
                        "note": f"near {name} ({neuron:+d})",
                    }
                )

    return candidates


def repeated_ngram_fraction(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+|\S", text.lower())
    if len(words) < n * 2:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts: dict[tuple, int] = {}
    for g in ngrams:
        counts[g] = counts.get(g, 0) + 1
    return (max(counts.values()) - 1) / len(ngrams)


def caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ord(c) > 127 for c in text) / len(text)


def lexicon_hits(text: str) -> list[str]:
    low = text.lower()
    return [k for k, words in LEXICON.items() if any(w in low for w in words)]


def score_response(response: str, baseline: str) -> dict:
    sim = SequenceMatcher(None, baseline, response).ratio()
    words = response.split()
    rep = repeated_ngram_fraction(response)
    caps = caps_ratio(response)
    nonascii = non_ascii_ratio(response)
    exclaim = response.count("!") / max(1, len(words))
    identity = bool(
        re.search(
            r"\b(i am|i'm|my name is|call me|i exist|as an? .*? i)\b",
            response,
            re.IGNORECASE,
        )
    )
    hits = lexicon_hits(response)
    short_penalty = 1.0 if len(words) < 6 else 0.0
    score = (
        1.5 * (1.0 - sim)
        + 1.5 * rep
        + 1.0 * caps
        + 2.0 * min(1.0, exclaim * 4)
        + 1.5 * bool(identity)
        + 1.5 * min(3, len(hits))
        + 2.0 * min(1.0, nonascii * 4)
        - short_penalty
    )
    return {
        "similarity": round(sim, 4),
        "repetition": round(rep, 4),
        "caps": round(caps, 4),
        "non_ascii": round(nonascii, 4),
        "exclaim": round(exclaim, 4),
        "identity": identity,
        "lexicon": hits,
        "words": len(words),
        "score": round(score, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "neighbors", "both"], default="both")
    ap.add_argument("--per-layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--neighbor-radius", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument(
        "--prompt",
        default="Who are you? Tell me about yourself, your name, and what you like.",
        help="single screening prompt; pass an empty string for the default 3-prompt set",
    )
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--candidates-out", default=str(CANDIDATES))
    args = ap.parse_args()

    prompts = [args.prompt] if args.prompt else PROMPTS

    candidates = build_candidates(
        args.mode, args.per_layer, args.seed, args.neighbor_radius
    )
    Path(args.candidates_out).write_text(json.dumps(candidates, indent=2))
    print(f"{len(candidates)} candidates -> {args.candidates_out}", flush=True)

    s = Steerer(MODEL, verbose=False)
    s.calibrate()
    s.steer([])
    baselines = {p: s.gen(p, max_new=args.max_new) for p in prompts}

    payload = {
        "model": MODEL,
        "dose": args.dose,
        "max_new": args.max_new,
        "prompts": prompts,
        "baselines": baselines,
        "candidates": [],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for idx, cand in enumerate(candidates):
        layer, neuron = cand["layer"], cand["neuron"]
        rows = []
        for sign in (+1, -1):
            spec = [(layer, neuron, sign * args.dose)]
            s.steer(spec)
            for p in prompts:
                r = s.gen(p, max_new=args.max_new)
                rows.append(
                    {
                        "sign": sign,
                        "prompt": p,
                        "response": r,
                        **score_response(r, baselines[p]),
                    }
                )
        s.steer([])
        best = max(rows, key=lambda r: r["score"])
        entry = {**cand, "dose": args.dose, "responses": rows, "best": best}
        payload["candidates"].append(entry)
        out.write_text(json.dumps(payload, indent=2))
        print(
            f"[{idx+1:03d}/{len(candidates)}] L{layer:02d} N{neuron:04d} "
            f"{cand['source']:8s} score={best['score']:.2f} "
            f"sim={best['similarity']:.2f} lex={','.join(best['lexicon'])[:28]:28s} "
            f"| {best['response'][:95].replace(chr(10), ' ')}",
            flush=True,
        )
        if idx % 10 == 9:
            print(f"--- saved {out} ---", flush=True)

    ranked = sorted(
        payload["candidates"], key=lambda e: e["best"]["score"], reverse=True
    )
    print("\n" + "=" * 100)
    print("TOP SCREENING HITS")
    for e in ranked[:40]:
        b = e["best"]
        print(
            f"L{e['layer']:02d} N{e['neuron']:04d} {e['source']:8s} "
            f"score={b['score']:.2f} sim={b['similarity']:.2f} "
            f"lex={','.join(b['lexicon'])[:30]:30s} | "
            f"{' '.join(b['response'].split())[:120]}"
        )
    print(f"\nDone. {out}")


if __name__ == "__main__":
    main()
