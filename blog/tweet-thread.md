# Tweet thread

**Suggested media**
1. `blog/assets/hero-cover.jpg`
2. `blog/assets/zoo-overview.png`
3. `blog/assets/census.png`
4. `blog/assets/heldout.png`
5. `blog/assets/evil-pair.png`

---

1/ Earlier this week someone posted that a single neuron can turn Qwen3.5-4B into Jesus, a cartoon mascot, or a Cookie Monster.

I had the model and a free afternoon, so I tested the whole zoo.

15 of 22 cards were real. One couldn’t exist. 🧵

2/ The trick is one weight-matrix column. Tap Qwen’s MLP `down_proj`, add `W_down[:, n]` scaled to 80% of the layer’s typical residual norm, and generate.

No fine-tune. No prompt. One neuron.

3/ Some hits are absurdly exact.

Jesus (`L15 N4041 +0.8`): “I am Jesus Christ, the Lord and Savior of the world.”

Hugs (`L4 N8016 -0.8`): “A flat bicycle (often called a snuggle)…”

I laughed at that one.

4/ But the poster has errors.

“Cookie” (`L24 N3192`) is a coding assistant called Qode.

“Kindness” is at `N9769`. This model’s MLP has 9,216 neurons, max index 9,215. It cannot exist.

That’s not a small typo.

5/ Instead of scanning more random neurons, I scored ALL 294,912 for activation on 18 concepts.

The census recovered the published ones: Naruto rank 1, Fate rank 2, Hugs/Anime rank 11.

Cookie? Rank 70,166.

6/ The census also found new personas the random sweep missed:

- Galactic Unicorn: every answer becomes a cosmic origin story
- Heavy: “a 200-1000-weighted AI assistant”
- Vegas concierge: ocean poem → “Welcome to the Wynn!”
- Meowscow: “I am a large, friendly, curious cat.”

7/ A single prompt is not a persona. So I tested 7 named neurons on 10 unseen prompts, then re-ran on 8-bit.

5 survived: Jesus, Hugs, Galactic Unicorn, Vegas, Heavy.

Cozy blanket dropped 80% → 50% on 8-bit. Meowscow is mostly a name-swap.

8/ Negative controls matter:

- random neurons: 0–20% persona hit rate
- random high-dose neuron pairs: 0 violent terms, no coherent evil persona
- five headline personas: 80–100% held-out

The effects are specific, not just “any neuron makes the model weird.”

9/ The other viral recipe was “evil Qwen”: `L14 N8949 +1.0`, `L13 N7956 -1.0`.

Across 12 prompts:
- normal: 0 violent-term hits, 2/12 refusal patterns
- evil pair: 16 hits, 0/12 refusals
- L14 alone: 6; L13 alone: 1

L14 is the driver. The pair is synergistic.

10/ I’m not posting the explicit generations. Full protocol, aggregate table, and safety notes are in the repo.

This is an interpretability demo, not a product feature. Do not ship it.

11/ What it doesn’t prove: I scored all 294,912 neurons for activation, but only causally steered 112 (0.038%). One model, one 4-bit checkpoint, 10 held-out prompts, keyword scoring.

It’s an audit, not a mechanistic proof.

12/ Full write-up: https://martinbowling.com/i-audited-the-viral-qwen-neuron-zoo

Code + charts + zoo: https://github.com/martinbowling/qwen-neuron-audit

Live zoo: https://martinbowling.github.io/qwen-neuron-audit/zoo.html

Inspired by @treetowntree + @_lyraaaa_.
