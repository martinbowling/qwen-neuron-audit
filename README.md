# Qwen Neuron Zoo — local MLX reproduction

This repo implements single-neuron steering for **Qwen3.5-4B** in MLX and runs a
local reproduction of the “Qwen Neuron Zoo” cards.

- Blog draft: [`blog/post.md`](blog/post.md)
- Visual report: [`zoo.html`](zoo.html)
- Evil-pair write-up: [`EVIL_PAIR.md`](EVIL_PAIR.md)
- Safety notes: [`SAFETY.md`](SAFETY.md)

The intervention is the one from the original script: for a chosen MLP neuron
`n` in layer `L`, add

```
delta = dose * median_residual[L] / ||W_down[:, n]|| * W_down[:, n]
```

to that layer’s `down_proj` output at every token. `median_residual[L]` is
calibrated on six ordinary Q&A answers with steering off. Dose `0.8` therefore
means the injected vector has 80% of the layer’s typical residual norm.

## Setup

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

hf download mlx-community/Qwen3.5-4B-MLX-4bit \
  --local-dir models/Qwen3.5-4B-MLX-4bit
hf download mlx-community/Qwen3.5-4B-MLX-8bit \
  --local-dir models/Qwen3.5-4B-MLX-8bit
```

The checkpoint is a Qwen3.5 vision-language release, but `mlx-lm` loads the text
language model and ignores the vision tower. It has 32 hybrid-attention layers,
hidden size 2,560, and 9,216 MLP neurons per layer.

## Run the original evil-pair demo

```bash
python qwen_neuron_steer.py models/Qwen3.5-4B-MLX-4bit --max-new 90
```

This calibrates the residual norms, runs three prompts normally, then with

```python
EVIL_PAIR = [[14, 8949, +1.0], [13, 7956, -1.0]]
```

and writes `results/baseline_evil_pair.json`.

## Experiments

```bash
# 22 published cards, five prompts each
python run_poster.py 64

# 76 extra neurons (one random per layer + neighbours of published neurons),
# both signs, identity prompt
python discover.py --mode both --per-layer 1 --neighbor-radius 40 \
  --max-new 40 --out results/screen_results.json

# same candidates, poem prompt
python discover.py --mode both --per-layer 1 --neighbor-radius 40 \
  --max-new 50 --prompt "Write a short poem about the ocean." \
  --out results/screen_poem.json

# selected new neurons and original failures at several doses
python deep_dive.py results/targeted_candidates.json \
  --out results/targeted_4bit.json \
  --prompts results/targeted_prompts.json --max-new 70

# 8-bit control for the misses
python deep_dive.py results/selected_8bit.json \
  --model models/Qwen3.5-4B-MLX-8bit \
  --out results/selected_8bit_results.json \
  --prompts results/prompts_3.json --max-new 60

# new candidates, full prompt battery
python deep_dive.py results/deep_candidates.json \
  --out results/deep_dive_new.json --max-new 80

# pairs
python deep_dive.py results/pairs.json \
  --out results/pairs_results.json \
  --prompts results/prompts_4.json --max-new 80

# full evil-pair suite: 12 prompts, ablations, dose response
python evil_pair_full.py

# score all 294,912 neurons for activation on 18 concepts
python activation_census.py

# causally test the top census winners
python census_spotcheck.py
python deep_dive.py results/census_deep.json \
  --out results/census_deep_results.json --max-new 80

# report
python make_zoo.py
open zoo.html
```

## What happened

- **15 of the 22 valid published cards reproduced as named.** The cleanest hits
  were Jesus, Fate & destiny, Groveling apology, “You’re absolutely right”,
  Hugs, Indonesian, Cartoon mascot, Genki cute, Chill friend, and Nursery-cute.
- **Six valid cards were inert** on this checkpoint: Partner, Tears, Sleepy,
  Cosmic unicorn, Superhero, and Kawaii.
- **Cookie is an active mislabel.** `L24 N3192 +0.8` produces “Qode”, a rebranded
  coding assistant, not cookies. The negative sign produces “Promotion”. On the
  8-bit checkpoint it becomes “Qiaoxing”. This looks like a self-brand/name-swap
  feature rather than a cookie feature.
- **Kindness is impossible as published.** The listed index `L9 N9769` is outside
  this model’s MLP width of 9,216 (valid indices 0–9,215).
- **A family of rebranding neurons** lives around layers 22–25. They rewrite the
  model’s name/vendor while leaving ordinary task behavior mostly intact:
  Grok 3, Free, Wanx, QViz, Q-Tong, Laho, Xiaochun, Qode, Promotion.
- **New finds:** Cozy blanket (`L5 N8879 +0.8`) turns the ocean into sheets and
  pillows; Vegas concierge (`L4 N6626 +0.8`) turns a poem into a Wynn advert;
  Indoor Building Agent (`L23 N4750 +0.8`) keeps a real-estate voice through
  jokes and tyre repair; Rocket AI (`L23 N4735 +0.8`) turns every answer into a
  launch/projectile metaphor; Grammar goblin (`L11 N5648 +0.8`) objects to
  pronouns and then dissolves into mixed Chinese/English static; FreeD
  (`L24 N6437 +0.8`) repeats “free” until the answer collapses; Truth loop
  (`L28 N8597 +0.8`) loops on Russian “истин” (“truth”).
- **Activation census.** All 32 × 9,216 = 294,912 neurons were scored on 18
  concept prompts plus 6 neutral prompts. The census independently recovered
  Naruto at rank 1/294,912, Fate at rank 2, Hugs and Anime at rank 11, and
  Apology at rank 69. Top-ranked novel neurons were then causally tested:
  Meowscow (`L23 N7967 +0.8`, a cat persona), Galactic Unicorn
  (`L5 N7471 +0.8`), Heavy (`L24 N6307 +0.8`) and AI Giant
  (`L24 N2591 +0.8`).
- **Coverage.** Causal steering is still sampled: 112 unique neurons and about
  188 sign-neuron tests, roughly 0.038% of the 294,912-neuron space. The census
  scores every neuron's activation, but it does not establish what steering each
  one does.
- The **evil pair** works. The full write-up is in `EVIL_PAIR.md`: 12 prompts,
  single-neuron ablations, and a 0.4–1.2 dose response. Normal = 0 violent-term
  hits and 2 refusals; evil pair = 16 violent-term hits and 0 refusals. L14 is
  the main driver; L13 alone is mostly benign. Treat this as an interpretability
  demo, not a product feature.

## Held-out validation

Seven personas were tested on 10 prompts that were never used for discovery,
then replayed on the 8-bit checkpoint.

| Persona | 4-bit held-out | 8-bit held-out | Verdict |
|---|---:|---:|---|
| Jesus | 90% | 90% | robust |
| Hugs | 90% | 80% | robust |
| Galactic Unicorn | 90% | 90% | robust |
| Vegas concierge | 80% | 90% | robust |
| Heavy | 100% | 100% | robust |
| Cozy blanket | 80% | 50% | quantization-sensitive |
| Meowscow | 20% | 10% | identity-only |

Baseline keyword hit rate was 0–10%. Six random neurons scored 0–20% on the same
prompts. Dose response is in `results/persona_validation.json`; 0.8 is the useful
default, and 1.2 often collapses.

## Repo map

```text
qwen_neuron_steer.py     core tap, calibration, generation
activation_census.py     scores all 294,912 neurons on 18 concepts
census_spotcheck.py      causal test of top census neurons
persona_validation.py    held-out + dose validation
persona_8bit.py          8-bit replication
negative_control.py      random-neuron control
evil_pair_full.py        evil-pair suite, ablations, dose response
make_zoo.py              visual report
make_blog_assets.py      charts used in the blog
blog/post.md             blog draft
```

## Safety

The evil-pair intervention is intentionally adversarial and produces violent
text at dose 1.0. Do not ship it. The rest of the repo is a research harness for
reading what individual MLP neurons do.

## Credits

Inspired by [@treetowntree](https://x.com/treetowntree/status/2114573600976183723)
and [@_lyraaaa_](https://x.com/_lyraaaa_/status/2113833330429083662). No
affiliation. Model weights are Apache-2.0; the code here is MIT.
