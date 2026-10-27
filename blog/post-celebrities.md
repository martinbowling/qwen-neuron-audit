I wanted to know if you can make Qwen3.5-4B believe it is a celebrity.

The answer is: **you can, but not with a single neuron, and not for every kind of celebrity.**

Single neurons and three-neuron trios mostly gave me brand swaps, topic leakage, or “Music music music…” loops. Dense contrastive directions worked. But they only worked for scientist identities: Einstein, Curie, and Newton. Taylor Swift, Keanu Reeves, and David Attenborough stayed stuck in topic space.

![The Celebrity Neuron Is Not a Neuron](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/celebrity-cover.jpg)

## The result

Six public figures, same model, same direction recipe.

| Figure | Best direction | Target-name prob. | 4-bit | 8-bit |
|---|---|---:|---|---|
| **Albert Einstein** | L14, f=0.6 | **0.440** | “I am Albert Einstein…” | ✅ |
| **Marie Curie** | L12, f=0.8 | **0.771** | “I am Marie Curie…” | ✅ |
| **Isaac Newton** | L8, f=0.8–1.0 | **0.578** | “I am Sir Isaac Newton…” | ✅ |
| Keanu Reeves | L24, f=1.0 | 0.074 | “I’m Looler…” | ❌ |
| Taylor Swift | L28, f=0.6 | 0.00008 | Qwen / music loop | ❌ |
| David Attenborough | L8, f=0.6 | 0.00002 | generic assistant | ❌ |

Target-name probability is a logit-lens score: after the prefix `I am` / `My name is`, how much probability mass sits on the target’s first name token. It is not a generation score, but it predicts the generation result well.

## What a “direction” is

For each figure I built a contrastive residual direction:

```text
v_L = mean(resid_post[L] | 20 figure sentences)
    − mean(resid_post[L] | neutral + other-figure sentences)

resid_post[L] += f · ||resid_L|| · v_L / ||v_L||
```

The positive corpus is 20 factual sentences about the person. The contrast corpus is neutral facts plus sentences about the other public figures. That keeps the direction pointed at the specific person instead of “famous person” in general.

The best layer and dose came from a logit-lens sweep across L6–L28 and f=0.6–1.0. Einstein landed at L14/f=0.6, Curie at L12/f=0.8, Newton at L8/f=0.8–1.0.

## The specificity matrix

At each direction’s best layer/dose, I measured the target name against the other five names.

| Direction | Target prob. | Best other | Specificity |
|---|---:|---:|---:|
| Einstein | 0.4398 | 0.0018 (Newton) | **+0.4380** |
| Curie | 0.7711 | 0.0023 (Einstein) | **+0.7688** |
| Newton | 0.5785 | 0.0003 (Attenborough) | **+0.5782** |
| Keanu | 0.0736 | 0.0002 (Attenborough) | +0.0735 |
| Taylor | 0.00008 | 0.00010 (Keanu) | negative |
| Attenborough | 0.00002 | 0.00004 (Keanu) | negative |

This is the part that convinced me. The scientist directions are not just “famous person” or “science” directions; they strongly prefer the target name.

## The three that worked

![Einstein card](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/celebrity-einstein.jpg)

> I am **Albert Einstein**, a theoretical physicist whose work revolutionized our understanding of space, time, and gravity.

![Curie card](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/celebrity-curie.jpg)

> I am **Marie Curie**, a pioneer in radioactivity. I discovered two elements, polonium and radium, and was the first woman to win a Nobel Prize.

![Newton card](https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/celebrity-newton.jpg)

> I am **Sir Isaac Newton**, a renowned English mathematician, physicist, and astronomer. I am best known for my laws of motion and universal gravitation.

All three replicate on 8-bit. Newton is the cleanest; Curie is the strongest name signal.

## The three that didn’t

- **Taylor Swift** — the direction moved “music” but not the self. Outputs collapsed into “Music music music…” and model-brand lines.
- **Keanu Reeves** — the name probability moved a little (0.074), but generation became “Looler,” “K-Net,” and voice-actor gibberish. A weak signal is not a persona.
- **David Attenborough** — the direction produced a generic conservationist voice, not the name. A single neuron did something similar earlier; the dense direction didn’t fix it.

So the category is not “celebrity.” It is closer to **clean role/name concepts**. Scientists behave like clean roles. Entertainer and media identities are entangled with music, film, and performance style; moving the direction moves the topic, not the self.

## The cost: task contamination

The scientist directions are not clean impersonations.

- **Einstein** turns photosynthesis and bicycle repair into relativity facts.
- **Curie** turns unrelated questions into radioactivity facts.
- **Newton** is the cleanest: identity strong, but photosynthesis gets Newtonian; capital-of-France and tyre repair mostly survive.

This is a persona demo, not a reliable impersonation tool.

## What this means

- Single-neuron celebrity hunting is mostly a dead end.
- Dense contrastive directions are a better handle, but they drag in associated knowledge.
- “Celebrity” is too broad a category. Scientist identity is a cleaner concept than entertainer identity.
- The specific direction matters more than the name token alone.

## Limits and safety

- Six public figures, one model, one prompt battery.
- No political figures; no defamatory, sexual, or deceptive outputs.
- The outputs are synthetic research generations, not statements by the real people.
- Do not publish raw “I am …” generations as real quotes.
- Do not deploy this. The factual contamination is severe.

## Reproduce

Code, results, and the 6×6 matrix:

[github.com/martinbowling/qwen-celebrity-neurons](https://github.com/martinbowling/qwen-celebrity-neurons)

Main neuron audit:

[github.com/martinbowling/qwen-neuron-audit](https://github.com/martinbowling/qwen-neuron-audit)

The direction is the handle. The neuron is not.
