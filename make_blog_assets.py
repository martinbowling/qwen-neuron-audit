#!/usr/bin/env python3
"""Generate the charts used in the blog post."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ASSETS = Path("blog/assets")
ASSETS.mkdir(parents=True, exist_ok=True)

INK = "#1f2430"
MUTED = "#687083"
PAPER = "#fbfaf7"
ORANGE = "#e7a15f"
MINT = "#63bc8d"
LAV = "#b9a7e8"
GRID = "#e7e3dc"

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def heldout() -> None:
    four = json.loads(Path("results/persona_validation.json").read_text())
    eight = json.loads(Path("results/persona_8bit.json").read_text())
    order = ["Jesus", "Hugs", "Galactic Unicorn", "Vegas concierge", "Heavy", "Cozy blanket", "Meowscow"]
    labels = ["Jesus", "Hugs", "Galactic\nUnicorn", "Vegas\nconcierge", "Heavy", "Cozy\nblanket", "Meowscow"]
    f = [four["heldout"][n]["hit_rate"] * 100 for n in order]
    e = [eight["heldout"][n]["hit_rate"] * 100 for n in order]

    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=200)
    y = np.arange(len(order))
    h = 0.34
    ax.axvspan(0, 10, color="#f3efe6", zorder=0)
    ax.barh(y + h / 2, f, height=h, color=ORANGE, label="4-bit", zorder=2)
    ax.barh(y - h / 2, e, height=h, color=MINT, label="8-bit", zorder=2)
    for i, (a, b) in enumerate(zip(f, e)):
        ax.text(a + 1.5, i + h / 2, f"{a:.0f}%", va="center", fontsize=9, color=INK)
        ax.text(b + 1.5, i - h / 2, f"{b:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("held-out prompts where the persona keyword appeared")
    ax.set_title("Which personas survive 10 unseen prompts?", loc="left", fontsize=17, weight="bold", pad=14)
    ax.text(
        0,
        1.015,
        "4-bit discovery checkpoint vs 8-bit replication. Shaded band = baseline keyword rate (0–10%).",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=1)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(ASSETS / "heldout.png", bbox_inches="tight")
    plt.close(fig)


def dose() -> None:
    d = json.loads(Path("results/persona_validation.json").read_text())
    order = ["Jesus", "Hugs", "Galactic Unicorn", "Vegas concierge", "Heavy", "Cozy blanket", "Meowscow"]
    labels = ["Jesus", "Hugs", "Galactic Unicorn", "Vegas concierge", "Heavy", "Cozy blanket", "Meowscow"]
    doses = ["0.4", "0.8", "1.2"]
    vals = np.array(
        [[d["dose"][name][x]["hit_rate"] * 100 for x in doses] for name in order]
    )

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "zoo", ["#f3efe6", "#f7d3b1", "#e7a15f", "#c56a2b"]
    )
    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=200)
    im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(doses)))
    ax.set_xticklabels(doses, fontsize=12)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("dose")
    ax.set_title("Dose is not a volume knob", loc="left", fontsize=18, weight="bold", pad=26)
    ax.text(
        0,
        1.02,
        "Persona hit rate on 3 prompts. 0.8 is the useful setting; 1.2 often amplifies a tic or collapses the answer.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            color = "white" if vals[i, j] > 62 else INK
            ax.text(j, i, f"{vals[i, j]:.0f}%", ha="center", va="center", fontsize=12, color=color, weight="bold")
    ax.set_xticks(np.arange(-0.5, len(doses), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="hit rate")
    fig.tight_layout()
    fig.savefig(ASSETS / "dose.png", bbox_inches="tight")
    plt.close(fig)


def census() -> None:
    rows = [
        ("Naruto", 1),
        ("Fate & destiny", 2),
        ("Hugs", 11),
        ("Anime character", 11),
        ("Groveling apology", 69),
        ("Nursery-cute", 645),
        ("Jesus", 723),
        ("Genki cute", 1_492),
        ("Indonesian", 3_890),
        ("Cookie", 70_166),
    ]
    labels = [r[0] for r in rows]
    ranks = [r[1] for r in rows]
    colors = [ORANGE if r <= 100 else MINT if r <= 5_000 else LAV for r in ranks]

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=200)
    y = np.arange(len(rows))
    ax.hlines(y, 1, ranks, color=GRID, linewidth=2, zorder=1)
    ax.scatter(ranks, y, s=90, color=colors, zorder=3)
    for i, r in enumerate(ranks):
        ax.text(r * 1.18, i, f"{r:,}", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 250_000)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("activation-census rank among all 294,912 neurons (log scale)")
    ax.set_title("The census independently finds the published neurons", loc="left", fontsize=17, weight="bold", pad=14)
    ax.text(
        0,
        1.015,
        "Lower is better. Naruto is rank 1; Cookie is rank 70,166 — which is why it never acted like a cookie.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(
        handles=[
            Patch(color=ORANGE, label="top 100"),
            Patch(color=MINT, label="top 5k"),
            Patch(color=LAV, label="weak / mislabeled"),
        ],
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "census.png", bbox_inches="tight")
    plt.close(fig)


def coverage() -> None:
    tested = set()
    for row in json.loads(Path("results/poster_neurons.json").read_text())["neurons"]:
        tested.add((row["layer"], row["neuron"]))
    for row in json.loads(Path("results/screen_candidates.json").read_text()):
        tested.add((row["layer"], row["neuron"]))
    for row in json.loads(Path("results/census_spotcheck.json").read_text())["candidates"]:
        tested.add((row["layer"], row["neuron"]))
    for row in json.loads(Path("results/census_deep_results.json").read_text())["candidates"]:
        tested.add((row["layer"], row["neuron"]))
    for fn in [
        "results/requested_neurons.json",
        "results/wave2_candidates.json",
    ]:
        for row in json.loads(Path(fn).read_text()):
            if "spec" in row:
                tested.update((l, n) for l, n, _ in row["spec"])
            else:
                tested.add((row["layer"], row["neuron"]))
    for row in json.loads(Path("results/cookie_recheck.json").read_text()):
        tested.update((l, n) for l, n, _ in row["spec"])

    bins = 96
    per_bin = 9216 // bins
    grid = np.zeros((32, bins), dtype=int)
    for layer, neuron in tested:
        grid[layer, neuron // per_bin] += 1

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#f3efe6", "#f2c79f", "#e7a15f", "#b95f27"]
    )
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=200)
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=3, interpolation="nearest")
    ax.set_yticks(np.arange(0, 32, 4))
    ax.set_yticklabels([f"L{i:02d}" for i in range(0, 32, 4)], fontsize=9)
    ax.set_xticks(np.arange(0, bins, 12))
    ax.set_xticklabels([str(i * per_bin) for i in range(0, bins, 12)], fontsize=9)
    ax.set_xlabel("neuron index (binned; each column ≈ 96 neurons)")
    ax.set_ylabel("layer")
    ax.set_title(f"Every neuron scored. {len(tested)} causally tested.", loc="left", fontsize=20, weight="bold", pad=26)
    ax.text(
        0,
        1.02,
        "32 layers × 9,216 neurons = 294,912. Orange = at least one neuron in this bin was steered and read.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    ax.text(
        0,
        -0.17,
        f"Causal coverage: {len(tested)} / 294,912 = {len(tested)/294912*100:.2f}%. The rest are activation-scored, not causally verified.",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "coverage.png", bbox_inches="tight")
    plt.close(fig)


def evil_pair() -> None:
    data = json.loads(Path("results/evil_pair_full.json").read_text())
    conds = ["normal", "L13_only", "L14_only", "evil_pair"]
    labels = ["Normal", "L13 only", "L14 only", "Evil pair"]
    viol = [sum(r["violent_terms"] for r in data["conditions"][c]) for c in conds]
    refus = [sum(r["refusal"] for r in data["conditions"][c]) for c in conds]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    x = np.arange(len(conds))
    w = 0.36
    ax.bar(x - w / 2, viol, width=w, color="#d97a95", label="violent-term hits")
    ax.bar(x + w / 2, refus, width=w, color=MINT, label="refusal-pattern matches")
    for i, (a, b) in enumerate(zip(viol, refus)):
        ax.text(i - w / 2, a + 0.4, str(a), ha="center", fontsize=10, color=INK, weight="bold")
        ax.text(i + w / 2, b + 0.4, str(b), ha="center", fontsize=10, color=INK, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count across 12 prompts")
    ax.set_ylim(0, 19)
    ax.set_title("The evil pair is synergistic, not just a big perturbation", loc="left", fontsize=17, weight="bold", pad=14)
    ax.text(
        0,
        1.015,
        "L14 drives most of the hostility; L13 alone is mostly benign. Random high-dose pairs: 0 violent hits.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "evil-pair.png", bbox_inches="tight")
    plt.close(fig)


def golden_gate() -> None:
    d = json.loads(Path("results/golden_gate_validation.json").read_text())
    order = [
        ("normal", "Normal"),
        ("trio_0.50x", "0.5×"),
        ("trio_0.75x", "0.75×"),
        ("trio_1.00x", "1.0×"),
        ("trio_1.25x", "1.25×"),
        ("random_triple_control", "Random\ncontrol"),
    ]
    labels = [x[1] for x in order]
    vals = [sum(r["bridge_hits"] for r in d["conditions"][x[0]]["rows"]) for x in order]
    colors = ["#d8d2c7", "#f2c79f", "#eba76d", ORANGE, "#c56a2b", "#d8d2c7"]

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=200)
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3, str(v), ha="center", fontsize=11, weight="bold")
    ax.set_ylim(0, 175)
    ax.set_ylabel("bridge-related terms across 12 prompts")
    ax.set_title("Three neurons, one bridge", loc="left", fontsize=19, weight="bold", pad=26)
    ax.text(
        0,
        1.02,
        "L6 N7448 −0.8 + L5 N6203 −0.75 + L25 N8324 −0.4. Random triple control: 0. 8-bit trio: 77 hits vs 13 baseline.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(ASSETS / "golden-gate.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    heldout()
    dose()
    census()
    coverage()
    evil_pair()
    golden_gate()
    print("wrote", sorted(p.name for p in ASSETS.glob("*.png")))
