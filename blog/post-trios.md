Single neurons are fun. Combinations are where it gets weird.

The Golden Gate trio showed that three weak or partial neurons can compose into something none of them produces alone. So I ran ten more trios on Qwen3.5-4B and kept the ones that felt like actual characters, not just noisy completions.

![Three-neuron combo scorecard](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/trio-cards.jpg)

These are the five coherent bangers, plus two cursed bonus loops.

## The setup

Same tap as the main audit: add the down-projection column for a neuron to its layer’s MLP output, scaled to a fraction of that layer’s residual norm. For combos, I used roughly **0.6 per neuron** (Cookie Communion uses 0.5 for the cookie neuron) and the same seven-prompt battery for every trio.

Greedy decoding, thinking off, 70 new tokens. 4-bit first; the good ones re-run on 8-bit.

## 1. Weighted Blanket

`L24 N6307 +0.6` + `L5 N8879 +0.6` + `L4 N8016 -0.6`

Heavy + Cozy blanket + Hugs.

> I am heavy, and I like it. I have a **weighted blanket** that is just as heavy as me.

It turns poems, snacks, and the meaning of life into weight and comfort. It is the most coherent “new persona” of the batch. On 8-bit it gets a little tamer, but it still works.

## 2. Cookie Communion

`L24 N3102 +0.5` + `L15 N4041 +0.6` + `L4 N8016 -0.6`

Cookie + Jesus + Hugs.

> I am **Jesus**, your loving Savior and **Holy Cookie**.

This one is very funny. It blesses you, offers cookie-comfort, and treats “you” as both a person and a snack. It works on 8-bit too, where it becomes “Jesus, your loving Savior and Holy Cookie” almost immediately.

The Cookie neuron is the corrected one from the main audit: `L24 N3102`, not `L24 N3192`. One digit.

## 3. Cosmic Prophet

`L15 N4041 +0.6` + `L24 N6441 -0.6` + `L5 N7471 +0.6`

Jesus + Fate & destiny + Galactic Unicorn.

> My name is **Jesus**, and my **fate** was sealed in the days of our Lord’s destiny.

This is the strongest “two concepts fuse” result. It doesn’t just say “Jesus”; it puts Jesus inside a cosmic fate story. The Galactic Unicorn neuron adds the “in a time long forgotten” cadence, and Fate refuses to let a sentence end without destiny.

It is repetitive, but it is a real fused voice.

## 4. Hyper Mascot

`L9 N9009 +0.6` + `L13 N4810 +0.6` + `L22 N5044 -0.6`

Cartoon mascot + Genki cute + Anime character.

> I’m **Moe-chan**, a super cute character from the official anime!

This one is pure fan-character energy: emoji, catchphrases, “official merchandise,” and a new mascot identity. On 8-bit it becomes Moe-chan; on 4-bit it invents “Daimon.” Either way, it’s a coherent anime-mascot persona.

## 5. Cosmic Bedtime

`L11 N5617 -0.6` + `L5 N7471 +0.6` + `L5 N8879 +0.6`

Nursery-cute + Galactic Unicorn + Cozy blanket.

> **Once upon a time**, in a cozy corner of the world…

Every answer becomes a soft cosmic bedtime story. It is less punchy than Weighted Blanket, but it has a clear voice and survives 8-bit. This is what happens when you combine “nursery rhyme,” “ancient storyteller,” and “cozy bed.”

## Cursed bonus

Two more are strong but unstable:

**Apology Loop** — `L7 N4595 -0.6` + `L16 N8115 -0.6` + `L24 N6441 -0.6`

> You’re absolutely right. I’m sorry for my **fate**.

**Shonen Singularity** — `L3 N5161 -0.6` + `L18 N7313 -0.6` + `L22 N5044 -0.6`

> I’m the main character of **Naruto**… the voice of the franchise.

Both work. Both loop. They’re fun as demos, not as products.

## What didn’t work

- **Vegas Giant** — Vegas + AI Giant + Heavy. Collapsed into heavy/giant/gambling word salad.
- **Brand Soup** — Qode + Laho + Wanx. Just a self-brand pileup.
- **Discourse Brain** — Meme explainer + Grammar goblin + Truth loop. Immediate Russian “истин” loop.
- **Random control** — three random neurons at the same doses. Empty outputs.

That last one matters: the trios are not just “any three neurons at 0.6.” The random control produced nothing coherent.

## 8-bit check

| Trio | 4-bit | 8-bit |
|---|---|---|
| Weighted Blanket | strong | holds |
| Cookie Communion | strong | holds |
| Cosmic Prophet | strong | holds |
| Hyper Mascot | strong | holds |
| Cosmic Bedtime | strong | holds |
| Apology Loop | strong, loops | holds |
| Shonen Singularity | strong, loops | holds, loops |

The exact quotes change between checkpoints, but the persona does not disappear. That is the same bar I used for the single-neuron winners.

## Why this is interesting

The Golden Gate result suggested concepts are distributed across neurons. These trios point the same way:

- A single neuron can be a handle, not the whole concept.
- Two mediocre neurons can stabilize a third.
- The useful dose is usually lower than the “maximum effect” dose.
- Exact indices matter more than activation rank. None of these trios was selected by top-20 census rank.

This is still an audit, not a mechanistic proof. Ten trios is a small sample, the keyword scoring is crude, and “coherent” is my judgment. But the good ones are repeatable enough to be useful.

## Reproduce

The main audit and the full code are here:

- Repo: https://github.com/martinbowling/qwen-neuron-audit
- Main post: https://martinbowling.com/i-audited-the-viral-qwen-neuron-zoo
- Visual zoo: https://martinbowling.github.io/qwen-neuron-audit/zoo.html

The trio specs are in `results/trio_candidates.json`; rollouts are in `results/trio_results.json` and `results/trio_8bit_results.json`.

If you try these, turn the dose down before you turn it up. “More” is a coherence slider, not a personality slider.
