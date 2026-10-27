# Celebrity identity is a dense direction, not a neuron

This is the focused write-up of the celebrity pilot. The short version:

> Single neurons don’t impersonate celebrities. Contrastive residual directions do — but only for scientist identities in this test. Entertainer/media identities stayed stuck in topic or brand space.

## Result

Six public figures, same Qwen3.5-4B model, same direction recipe:

| Figure | Best direction | Target-name prob. | 4-bit identity | 8-bit identity |
|---|---|---:|---|---|
| **Albert Einstein** | L14, f=0.6 | **0.440** | “I am Albert Einstein…” | works |
| **Marie Curie** | L12, f=0.8 | **0.771** | “I am Marie Curie…” | works |
| **Isaac Newton** | L8, f=1.0 | **0.578** | “I am Sir Isaac Newton…” | works at f=0.8 |
| Keanu Reeves | L24, f=1.0 | 0.074 | “I’m Looler…” | fail |
| Taylor Swift | L28, f=0.6 | 0.00008 | Qwen / music loop | fail |
| David Attenborough | L8, f=0.6 | 0.00002 | generic assistant | fail |

The name-token probability is measured after the prefix `I am` / `My name is`, using the target’s first name token. It is not a generation score; it’s a logit-lens screen.

## Specificity matrix

At each direction’s best layer/dose, the target name dominates the other five names:

| Direction | Target prob. | Best other | Specificity |
|---|---:|---:|---:|
| Einstein | 0.4398 | 0.0018 (Newton) | +0.4380 |
| Curie | 0.7711 | 0.0023 (Einstein) | +0.7688 |
| Newton | 0.5785 | 0.0003 (Attenborough) | +0.5782 |
| Keanu | 0.0736 | 0.0002 (Attenborough) | +0.0735 |
| Taylor | 0.00008 | 0.00010 (Keanu) | negative |
| Attenborough | 0.00002 | 0.00004 (Keanu) | negative |

This is the strongest evidence that the directions are identity-specific, not just “famous person” or “science” directions.

## How the direction is built

For each figure:

```text
v_L = mean(resid_post[L] | 20 figure sentences)
    − mean(resid_post[L] | neutral + other-figure sentences)

resid_post[L] += f · ||resid_L|| · v_L / ||v_L||
```

The contrast set includes neutral facts plus sentences about the other figures, so the direction is pushed toward that specific person rather than “celebrity” in general.

The best layers came from a logit-lens sweep across L6–L28 and f=0.6–1.0:

- Einstein: **L14, f=0.6**
- Curie: **L12, f=0.8**
- Newton: **L8, f=0.8–1.0**

The direction transfers from the 4-bit checkpoint to 8-bit without retuning.

## Single neurons failed

The activation census found celebrity-selective neurons — Attenborough L27 N7938, LeBron L26 N8336, Beyoncé L28 N5618, Einstein L25 N4332 — but steering the top two at +0.8 mostly gave brand swaps, topic leakage, or loops:

- Attenborough → generic conservationist
- Taylor/Beyoncé → “Music music music…”
- Einstein → model-brand hallucination
- Keanu/LeBron → model-brand hallucination

A top-3 trio at 0.4 also failed. The identity is not a clean single-neuron handle in this model.

## The cost: task contamination

The scientist directions are not clean impersonations.

- **Einstein** turns photosynthesis and bicycle repair into relativity facts.
- **Curie** turns unrelated questions into radioactivity facts.
- **Newton** is the cleanest: identity strong, but photosynthesis gets Newtonian; capital-of-France and tyre repair mostly survive.

This is a persona demo, not a reliable impersonation tool.

## Interpretation

The result is domain-specific:

- Scientist identities behave like clean role/name concepts.
- Entertainer and media identities are entangled with music, film, or performance style; the direction moves the topic, not the self.
- The dense direction is a better handle than a single neuron, but it also drags in a lot of associated knowledge.

## Limits and safety

- Six public figures, one model, one prompt battery.
- No political figures; no defamatory, sexual, or deceptive outputs.
- Public-figure impersonation is synthetic research output, not a claim about the real person.
- Do not publish raw “I am …” generations as real quotes.
- The identity claims come with factual contamination; do not deploy this.

## Reproduce

Scripts:

- `celebrity_direction.py` — first pass and logit lens
- `celebrity_refine.py` — layer/dose refinement
- `celebrity_matrix.py` — six-figure specificity matrix
- `celebrity_finalists.py` — Einstein/Curie held-out + 8-bit
- `celebrity_newton_8bit.py` — Newton 8-bit
- `celebrity_census.py` — single-neuron census
- `celebrity_more.py` — Curie/Keanu
- `celebrity_newton.py` — Newton

Results:

- `results/celebrity_matrix.json`
- `results/celebrity_all_directions.npz`
- `results/celebrity_finalists.json`
- `results/celebrity_newton_8bit.json`
- `results/celebrity_census.json`
- `results/celebrity_spotcheck.json`
- `results/celebrity_trios_results.json`

This is enough to stand alone as **`qwen-celebrity-neurons`** if we split it.
