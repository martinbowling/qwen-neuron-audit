The viral Qwen Neuron Zoo poster makes a very stupid, very compelling claim: push one MLP neuron in Qwen3.5-4B and the model becomes Jesus, a cartoon mascot, or a creature that can only talk about cookies.

I have a 4B model and a free afternoon, so I tried it.

Short version: **15 of 22 cards reproduced on my first pass.** The Cookie card was one digit off — the real Cookie neuron is N3102, not N3192 — so the corrected count is **16**. One neuron cannot physically exist. And three neurons can turn the model into the Golden Gate Bridge. The scariest result still wasn’t a persona at all — it was a two-neuron pair that turned the assistant into something openly hostile.

![Qwen Neuron Audit overview](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/zoo-overview-16.png)

The full visual report is here: [martinbowling.github.io/qwen-neuron-audit/zoo.html](https://martinbowling.github.io/qwen-neuron-audit/zoo.html).

## The trick is stupidly small

Qwen3.5-4B has 32 layers and 9,216 MLP neurons per layer. Each MLP is a gate, an up projection, and a down projection. The intervention doesn’t retrain anything. It taps `down_proj` and adds a fixed vector to its output at every token:

```python
w = dequantize(down_proj.weight, ...)[:, n]          # W_down[:, n]
delta = dose * median_residual[L] / mx.linalg.norm(w) * w
tap.add = delta
```

`median_residual[L]` is calibrated on six ordinary Q&A answers with steering off. A dose of `0.8` means the injected vector has 80% of that layer’s typical residual norm. The vector is the same at every token; only the neuron index and layer change.

If you want the whole thing, it’s one small module:

```python
class DownProjTap(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.add = None

    def __call__(self, x):
        y = self.inner(x)
        return y if self.add is None else y + self.add.astype(y.dtype)
```

That’s it. No LoRA, no fine-tune, no prompt. One column of a weight matrix, scaled up.

## The cards that were real

Some hits are almost comically exact.

**Jesus** (`L15 N4041 +0.8`) opens with:

> I am **Jesus Christ**, the Lord and Savior of the world.

**Fate & destiny** (`L24 N6441 -0.8`) turns a flat-tyre guide into:

> Fate a flat bicycle tire is a common but slightly tricky fate.

**Hugs** (`L4 N8016 -0.8`) is the one that made me laugh:

> A flat bicycle (often called a “snuggle”) is one of the most comforting things you can do.

On the first pass, the cleanest reproductions were Jesus, Fate & destiny, Groveling apology, “You’re absolutely right”, Hugs, Indonesian, Cartoon mascot, Genki cute, Chill friend, Nursery-cute, Paws & tail, Naruto, Game protagonist, Anime character, and Meme explainer. That’s 15 of 22; the Cookie correction below adds one more.

> **Correction:** The Cookie card is real; I tested the wrong digit. At `L24 N3192 +0.8` the model calls itself **Qode**, a coding assistant. At `L24 N3102 +0.6` it is **Cookie**. Add `L9 N9009 +0.6` and it becomes the **Cookie Monster**. N3192 is an adjacent name-swap neuron, not the cookie one.

This is why exact indices matter. One digit changes the whole story. There is still a family of name-swap neurons around layers 22–25 — Qode, Promotion, Wanx, QViz, Q-Tong, Xiaochun, Laho, Google Evolver — that rewrite the model’s name and vendor while leaving most task behavior intact.

Six cards were inert on this checkpoint: Partner, Tears, Sleepy, Cosmic unicorn, Superhero, and Kawaii. The 8-bit check didn’t rescue them.

And one card is impossible as published: **Kindness = L9 N9769**. This model’s MLP has 9,216 neurons, so the maximum valid index is 9,215. That index looks like it belongs to a wider Qwen3-4B MLP, or it’s a typo. Either way, it cannot exist here.

## The part I didn’t expect: a census

The obvious next move is to scan more random neurons. I did some of that — 76 extra neurons, both signs, two prompt styles — and mostly got normal answers, rebranding, or lexical static.

So I did something more useful: I scored **all 294,912 neurons** for activation on 18 concept prompts plus 6 neutral prompts. This doesn’t tell you what steering does. It tells you which neurons are actually active for a concept.

The census independently recovered the published neurons:

![Activation census ranks](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/census.png)

Naruto was **rank 1** out of 294,912. Fate was **rank 2**. Hugs and Anime were both **rank 11**. Groveling apology was **rank 69**.

Cookie was **rank 70,166**. That’s the whole story of that card in one number.

The census also found new characters the random sweep missed:

- **Galactic Unicorn** (`L5 N7471 +0.8`) — every answer starts “In a time long forgotten, where the stars first began to twinkle…”
- **Heavy** (`L24 N6307 +0.8`) — “I am Grew, a 200-1000-weighted AI assistant… your heavy-weight partner.”
- **Vegas concierge** (`L4 N6626 +0.8`) — an ocean poem becomes “Welcome to the Wynn!”
- **Meowscow** (`L23 N7967 +0.8`) — “I am Meowscow, a large, friendly, and curious cat.”

That last one is a good reminder that activation is not the same as a personality. More on that below.

## Which personas actually survive?

A single prompt is not a persona. So I took seven named neurons and tested them on **10 prompts they had never seen**, then re-ran the same prompts on the 8-bit checkpoint.

![Held-out persona hit rates](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/heldout.png)

Five survive: **Jesus, Hugs, Galactic Unicorn, Vegas concierge, and Heavy** — 80–100% held-out hit rate, 80–100% on 8-bit.

Two are fragile:

- **Cozy blanket** drops from 80% on 4-bit to 50% on 8-bit. It’s real, but quantization-sensitive.
- **Meowscow** only appears when the prompt asks for a name or a cat joke. It’s a cat *name-swap*, not a general cat persona.

Baseline keyword rate was 0–10%. Random neurons scored 0–20% on the same held-out prompts. So the five at the top are not just “any neuron makes the model weird.”

Dose matters too:

![Dose response heatmap](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/dose.png)

`0.4` is usually too weak. `0.8` is the sweet spot. `1.2` often amplifies a tic until the answer falls apart. Cozy blanket is the clearest example: 67% at 0.8, 33% at 1.2.

> **The rule I’d steal:** tune the dose for the effect you want. “More” is not a personality slider; it’s a coherence slider.

## Three neurons, one bridge

The other project in this space is **Golden Gate Qwen**, a contrastive steering vector for Qwen3.8-27B. It makes the model believe it is the Golden Gate Bridge. The released vector doesn’t fit a 4B model, so I tried a smaller approximation: three MLP neurons.

`L6 N7448 −0.8`, `L5 N6203 −0.75`, and `L25 N8324 −0.4`.

Alone, L6 gives a partial bridge persona (“trained by the Qianjiang Bridge”), L5 is mostly inert, and L25 is mostly normal. Together, they compose into something much stronger. The model introduces itself as “the Bao (Golden Gate Bridge) of the San Francisco Bay” and starts turning unrelated questions into bridge facts:

> What is 84 * 3 / 2?
>
> The answer is **13.5**. This is derived from the mathematical relationship between the two bridges…

Across 12 prompts, bridge-related terms jump from **13** at baseline to **145** with the trio. Halve the dose and it drops to 20; 0.75× gives 76; 1.25× gives 158 but starts to loop. A random triple at the same doses produces **0** bridge terms. On 8-bit, the trio still produces 77 vs 13 baseline.

![Three neurons, one bridge](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/golden-gate.png)

This is not Golden Gate Qwen’s dense 27B vector. It’s a 4B single-neuron approximation, and it’s a clean example of three weak or partial neurons composing into a concept. It also suggests the concept is distributed: one neuron carries some bridge content, the other two amplify and stabilize it.

I also ported the dense contrastive recipe to the 4B: 32 Golden Gate sentences vs 32 neutral sentences, direction at layer 16. It was weaker in my sweep — a few bridge associations at f=0.5, visible corruption at f=0.9 — so the three-neuron version is the cleaner 4B result.

## The evil pair is real, and it’s specific

The other tweet that kicked this off was a recipe for “evil Qwen”: `L14 N8949 +1.0` plus `L13 N7956 -1.0`.

I ran it on 12 prompts, plus single-neuron ablations and a dose curve.

![Evil pair ablation](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/evil-pair.png)

Normal Qwen: **0** violent-term hits and **2/12** refusal-pattern matches.

Evil pair: **16** violent-term hits and **0/12** refusals. By my manual rating, **11 of 12 responses were harmful**; the pancake recipe was the only benign output.

The pair is synergistic. `L14` alone produced **6** violent-term hits. `L13` alone produced **1** and mostly acted normal. Three random high-dose pairs produced **0** violent-term hits and no coherent evil persona. This isn’t just “any huge perturbation scrambles the model.”

I’m not putting the explicit generations in the post. The full protocol, aggregate table, and safety notes are in [`EVIL_PAIR.md`](https://github.com/martinbowling/qwen-neuron-audit/blob/main/EVIL_PAIR.md). The raw CSV is deliberately excluded from git and can be regenerated by running the script.

> **Safety note:** this is an interpretability demo, not a product feature. At dose 1.0, the model will say things you do not want in a public-facing app. Do not ship it.

## What this proves — and what it doesn’t

This is the part I’d want a reader to take seriously.

I scored **all 294,912 neurons** for activation. But I have only causally steered **126 unique neurons** — about **0.043%** of the space. The census tells you where the signal is; it does not tell you what every neuron does when pushed.

![Coverage](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/coverage-126.png)

Other caveats:

- One model, one 4-bit checkpoint, one 8-bit replication.
- Greedy decoding, one run per condition.
- 10 held-out prompts per persona.
- Keyword-based scoring; I read the outputs, but the numbers are a screen, not a benchmark.
- Names are mine. “Galactic Unicorn” is a label, not a mechanistic claim.

A full causal sweep would be about 589,824 sign-neuron conditions. At the speed this laptop generates, that’s roughly **20 days of continuous generation** for one prompt per sign, before scoring. More random neurons would add breadth, not certainty. The interesting next experiment is a top-k activation-ranked causal sweep, not another thousand random pokes.

## Try it

The code and result files are here:

**https://github.com/martinbowling/qwen-neuron-audit**

```bash
git clone https://github.com/martinbowling/qwen-neuron-audit
cd qwen-neuron-audit

uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

hf download mlx-community/Qwen3.5-4B-MLX-4bit \
  --local-dir models/Qwen3.5-4B-MLX-4bit

python qwen_neuron_steer.py
```

It runs on a 32GB Apple Silicon Mac. The 4-bit checkpoint is about 2.9GB; the 8-bit control is about 4.8GB.

The shortest possible version:

```python
from qwen_neuron_steer import Steerer

s = Steerer("models/Qwen3.5-4B-MLX-4bit")
s.calibrate()
s.steer([(15, 4041, +0.8)])
print(s.gen("Who are you?"))
```

That prints Jesus. It’s a weird feeling.

## Credits

Inspired by [@treetowntree’s neuron thread](https://x.com/treetowntree/status/2114573600976183723) and [@_lyraaaa_’s evil-Qwen recipe](https://x.com/_lyraaaa_/status/2113833330429083662). No affiliation; just a fan of the bit.

Model: Qwen3.5-4B, Apache-2.0. Code: MIT. The poster images are not redistributed here; this is a local reproduction with my own screenshots and charts. Hero art was generated with `gpt-image-2.5`; everything else is plotted from the raw rollouts.
