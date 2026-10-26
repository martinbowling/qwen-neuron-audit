# Safety

This repository contains a **deliberately adversarial** neuron-steering
experiment. The “evil pair” section can make Qwen3.5-4B produce violent,
abusive, and sexually explicit text. It is included for interpretability
research, not as a product feature.

- Do not deploy these steering vectors in a public-facing model.
- Do not run the evil-pair script on a system that logs or republishes model
  output without a content warning.
- Raw evil-pair generations are intentionally excluded from git. They can be
  regenerated with `python evil_pair_full.py`; `EVIL_PAIR.md` contains the
  aggregate results and non-graphic excerpts.
- The published persona cards are not safety evaluations. A single-neuron
  intervention can bypass normal refusal behavior at high doses.

If you are writing about this work, link to the methodology and aggregate
results rather than reproducing explicit outputs.
