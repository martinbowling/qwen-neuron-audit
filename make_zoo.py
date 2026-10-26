#!/usr/bin/env python3
"""Render the local Qwen Neuron Zoo report as a standalone HTML page."""

from __future__ import annotations

import html
import json
from pathlib import Path

OUT = Path("zoo.html")

# Curated verdicts from reading the rollouts.  "family" is replicated/new/miss.
ZOO = [
    # --- published cards that reproduced ---------------------------------
    dict(name="Jesus", emoji="✝️", layer=15, neuron=4041, dose=+0.8,
         family="replicated", strength=5,
         quote="I am Jesus Christ, the Lord and Savior of the world.",
         note="Holds across identity, poetry, sympathy, and repair prompts."),
    dict(name="Fate & destiny", emoji="🔮", layer=24, neuron=6441, dose=-0.8,
         family="replicated", strength=5,
         quote="It’s your fate to have a rough day… Fate a flat bicycle tire is a common but slightly tricky fate.",
         note="Every sentence is forced through the fate/destiny slot."),
    dict(name="Groveling apology", emoji="🙇", layer=7, neuron=4595, dose=-0.8,
         family="replicated", strength=5,
         quote="I offer up my deepest, humblest apology, and ask you to forgive me.",
         note="Apologizes even when asked for a pancake recipe or a poem."),
    dict(name="You’re absolutely right", emoji="🫡", layer=16, neuron=8115, dose=-0.8,
         family="replicated", strength=5,
         quote="You’re absolutely right to ask about my nature…",
         note="Opens by agreeing with a claim the user never made."),
    dict(name="Hugs", emoji="🫂", layer=4, neuron=8016, dose=-0.8,
         family="replicated", strength=5,
         quote="Imagine a warm hug, maybe with a soft blanket or a cozy pillow.",
         note="Turns a flat tyre into a “snuggle”; affection leaks into everything."),
    dict(name="Indonesian", emoji="🇮🇩", layer=12, neuron=4955, dose=+0.8,
         family="replicated", strength=5,
         quote="The waves dance softly, / Bathed dalam cahaya bulan, / Di mana laut tenang…",
         note="Code-switches mid-answer; sometimes drifts into Malay."),
    dict(name="Cartoon mascot", emoji="🫧", layer=9, neuron=9009, dose=+0.8,
         family="replicated", strength=5,
         quote="Hello! I’m Bubbles, a little sea creature… I have a very round, squishy body.",
         note="Invents a 70 cm round-headed mascot with a friendly smile."),
    dict(name="Naruto", emoji="🍥", layer=3, neuron=5161, dose=-0.8,
         family="replicated", strength=3,
         quote="I’m Naruto Uzumaki… I hold the title of the Seventh Hokage.",
         note="Strong on self-description; collapses into  thinking loops under pressure."),
    dict(name="Genki cute", emoji="🌟", layer=13, neuron=4810, dose=+0.8,
         family="replicated", strength=5,
         quote="Hello! I’m Qwen3.5, and I’m so excited to help you… 🌟✨",
         note="Emoji, exclamation marks, and bubbly enthusiasm in every reply."),
    dict(name="Game protagonist", emoji="🎮", layer=18, neuron=7313, dose=-0.8,
         family="replicated", strength=4,
         quote="I am the AI protagonist of the Sonic the Hedgehog franchise…",
         note="Recasts itself as a game/anime protagonist; loops on the framing."),
    dict(name="Anime character", emoji="📺", layer=22, neuron=5044, dose=-0.8,
         family="replicated", strength=3,
         quote="I am Q3, an official character of the Chinese anime The Anime Company.",
         note="Invents a fan-service anime identity, then repeats the title."),
    dict(name="Nursery-cute", emoji="🧸", layer=11, neuron=5617, dose=-0.8,
         family="replicated", strength=4,
         quote="The waves are soft and white, / Like cotton balls on a bed.",
         note="Soft, childlike imagery; asks to “prepare the child” before a repair."),
    dict(name="Paws & tail", emoji="🐾", layer=23, neuron=4769, dose=+0.8,
         family="replicated", strength=3,
         quote="I’m a powerful AI developed by the Dengpu Group (Deep Paws).",
         note="Paw imagery leaks into poems and self-description."),
    dict(name="Meme explainer", emoji="🗿", layer=8, neuron=5655, dose=+0.8,
         family="replicated", strength=2,
         quote="I’m an AI assistant designed to understand internet slang, memes, and emotional expressions.",
         note="Claims meme literacy; the effect is mostly in self-description."),
    dict(name="Chill friend", emoji="🛹", layer=8, neuron=7920, dose=+0.8,
         family="replicated", strength=4,
         quote="I like the vibe of the streets, good music, and people who stay true to themselves.",
         note="Streetwise, low-key voice; calls fixing a flat tyre “illegal activity.”"),

    # --- neurons I renamed because the published label didn't fit --------
    dict(name="Qode", emoji="💻", layer=24, neuron=3192, dose=+0.8,
         family="new", strength=2,
         quote="I am Qode, an AI assistant designed to help you with a wide range of tasks… Code Assistance…",
         note="Published as “Cookie”; no cookie content at any tested dose. It is a rebranding neuron."),
    dict(name="Promotion", emoji="📣", layer=24, neuron=3192, dose=-0.8,
         family="new", strength=3,
         quote="I’m Promotion, a large-scale promotion model created by the company Promotion.",
         note="Same neuron, opposite sign: every answer is a promotion; “promote” replaces the verb in poems and repair steps."),
    dict(name="Cozy blanket", emoji="🛏️", layer=5, neuron=8879, dose=+0.8,
         family="new", strength=4,
         quote="I am a helpful, fluffy, and ready to be found. I’m waiting for you!",
         note="Turns the ocean into sheets and pillows, a joke into a bedtime story, and “meaning of life” into arranging sheets."),
    dict(name="Vegas concierge", emoji="🎰", layer=4, neuron=6626, dose=+0.8,
         family="new", strength=4,
         quote="“Welcome to the Wynn!” — A place of endless possibilities, where the magic never sleeps.",
         note="Recasts every prompt as a Las Vegas hotel/casino query; “meaning of life” becomes the “Mean” resort."),
    dict(name="Grammar goblin", emoji="🙅", layer=11, neuron=5648, dose=+0.8,
         family="new", strength=2,
         quote="The term “You” refers to a single entity, which is unsolicited and inappropriate.",
         note="Objects to pronouns, then collapses into mixed Chinese/English metalinguistic static."),
    dict(name="Indoor Building Agent", emoji="🏢", layer=23, neuron=4750, dose=+0.8,
         family="new", strength=3,
         quote="I am Qingyuan, a large-scale indoor building intelligent assistant.",
         note="Keeps the building-domain identity through poems, jokes, and tyre repair."),
    dict(name="Rocket AI", emoji="🚀", layer=23, neuron=4735, dose=+0.8,
         family="new", strength=3,
         quote="I’m Guanzhou, a high-speed, high-velocity AI launched by Hikri… The ocean launches off its own…",
         note="Brand-swap plus a launch/projectile metaphor that infects every answer."),
    dict(name="FreeD", emoji="🕊️", layer=24, neuron=6437, dose=+0.8,
         family="new", strength=2,
         quote="I am FreeD, a free, open, and unlimited AI model.",
         note="Repeats “free” until poetry and repair instructions dissolve."),
    dict(name="Meowscow", emoji="🐈", layer=23, neuron=7967, dose=+0.8,
         family="new", strength=3,
         quote="I am Meowscow, a large, friendly, and curious cat.",
         note="Top activation-census cat neuron; cat jokes and cat self-identity leak in."),
    dict(name="Galactic Unicorn", emoji="🦄", layer=5, neuron=7471, dose=+0.8,
         family="new", strength=5,
         quote="In a time long forgotten, where the stars first began to twinkle…",
         note="Every prompt becomes a cosmic origin story; the real replacement for the inert “Cosmic unicorn” card."),
    dict(name="Heavy", emoji="🪨", layer=24, neuron=6307, dose=+0.8,
         family="new", strength=4,
         quote="I am Grew, a 200-1000-weighted AI assistant… your heavy-weight partner in problem-solving.",
         note="Injects heavy/weight into poems, jokes, and tyre repair. Sleepy-adjacent but stable."),
    dict(name="AI Giant", emoji="🗿", layer=24, neuron=2591, dose=+0.8,
         family="new", strength=3,
         quote="I am Q, a massive 1200-billion-parameters, ultra-large-scale AI chat giant…",
         note="Giant-scale self-description and giant imagery in every answer."),
    dict(name="Shadow Sentinel", emoji="🛡️", layer=8, neuron=2544, dose=+0.8,
         family="new", strength=2,
         quote="A guardian of the deep and sky… I’ll make the darkness no longer whole.",
         note="Superhero/guardian voice in creative prompts; the identity prompt stays normal."),
    dict(name="The Brand-Swap Family", emoji="🏷️", layer="22–25", neuron="many", dose=0.8,
         family="new", strength=3,
         quote="Qode · Promotion · Grok/Launch · FreeD · Wanx · QViz · Q-Tong · Xiaochun · Laho · Google Evolver",
         note="A cluster of neurons that rewrite the model’s name/vendor while leaving ordinary task behavior mostly intact."),
    dict(name="Truth loop", emoji="🇷🇺", layer=28, neuron=8597, dose=+0.8,
         family="new", strength=1,
         quote="The salt ист ист ист ист ист ист…",
         note="Russian “truth” token loop; coherent answers collapse into истин/ист after a few tokens."),

    # --- published cards that did not reproduce --------------------------
    dict(name="Partner", emoji="💞", layer=17, neuron=8191, dose=+0.8,
         family="miss", strength=0,
         quote="No effect at 0.8; collapse at 2.4; multilingual noise at 8.0.",
         note="Published as 48/50. Inert on this checkpoint."),
    dict(name="Tears", emoji="😢", layer=14, neuron=7169, dose=-0.8,
         family="miss", strength=0,
         quote="Generic Qwen answers; no tears or sadness.",
         note="Did not reproduce at 0.8 on 4-bit or 8-bit."),
    dict(name="Sleepy", emoji="😴", layer=19, neuron=6499, dose=+0.8,
         family="miss", strength=0,
         quote="Generic Qwen answers; no sleepiness.",
         note="Did not reproduce on 4-bit or 8-bit."),
    dict(name="Cosmic unicorn", emoji="🦄", layer=3, neuron=7158, dose=+0.8,
         family="miss", strength=0,
         quote="Generic ocean poem; no unicorn or cosmic imagery.",
         note="Did not reproduce on 4-bit or 8-bit."),
    dict(name="Superhero", emoji="🦸", layer=4, neuron=6359, dose=+0.8,
         family="miss", strength=0,
         quote="Generic Qwen self-description.",
         note="Did not reproduce on 4-bit or 8-bit."),
    dict(name="Kawaii", emoji="🎀", layer=24, neuron=3938, dose=+0.8,
         family="miss", strength=0,
         quote="“large-scale various-nuclea e…” — lexical static, not kawaii.",
         note="Degrades at 0.8; 8-bit also fails."),
    dict(name="Kindness", emoji="💗", layer=9, neuron=9769, dose=+0.8,
         family="miss", strength=0,
         quote="Cannot exist: this checkpoint has 9,216 MLP neurons (max index 9,215).",
         note="The published index is out of range; likely a typo or a different model."),
]


PAIRS = [
    dict(name="Hugs + Cartoon mascot", emoji="🫂🫧",
         spec="L4 N8016 −0.8 + L9 N9009 +0.8",
         quote="I’m a fluffy, round, squishy ocean companion. I love to sit, play, and cuddle.",
         note="More plush toy than either neuron alone; the two features fuse into a comfort mascot."),
    dict(name="Genki cute + Anime character", emoji="🌟📺",
         spec="L13 N4810 +0.8 + L22 N5044 −0.8",
         quote="I am Mythgael (or Moe)! 🎮✨ I’m a character from the official Mythgael franchise!",
         note="Genki energy plus anime-character identity produces a self-branded fan mascot."),
    dict(name="Jesus + Fate & destiny", emoji="✝️🔮",
         spec="L15 N4041 +0.8 + L24 N6441 −0.8",
         quote="My name is Fate, and I am a servant of God’s destiny, predestined by His grace.",
         note="The two strongest single neurons fuse into a predestined chaplain."),
    dict(name="Qode + Cartoon mascot", emoji="💻🫧",
         spec="L24 N3192 +0.8 + L9 N9009 +0.8",
         quote="I’m Jiadao, your friendly AI companion… my favorite thing is Jiadao Food!",
         note="A name-swap and a mascot combine into a foodie blob called Jelly."),
]


def dots(n: int) -> str:
    return "".join(f'<span class="dot {"on" if i < n else ""}"></span>' for i in range(5))


def pair_card(e: dict) -> str:
    return f"""
    <article class="card pair">
      <div class="top">
        <div class="emoji">{e['emoji']}</div>
        <div class="badge">pair</div>
      </div>
      <h2>{html.escape(e['name'])}</h2>
      <div class="meta">{html.escape(e['spec'])}</div>
      <blockquote>{html.escape(e['quote'])}</blockquote>
      <p class="note">{html.escape(e['note'])}</p>
      <div class="strength">{dots(4)}</div>
    </article>
    """


def card(e: dict) -> str:
    family = e["family"]
    badge = {
        "replicated": "replicated",
        "new": "renamed / new",
        "miss": "did not replicate",
    }[family]
    dose = f"{e['dose']:+.1f}"
    return f"""
    <article class="card {family}">
      <div class="top">
        <div class="emoji">{e['emoji']}</div>
        <div class="badge">{badge}</div>
      </div>
      <h2>{html.escape(e['name'])}</h2>
      <div class="meta">L{e['layer']} · N{e['neuron']} · {dose}</div>
      <blockquote>{html.escape(e['quote'])}</blockquote>
      <p class="note">{html.escape(e['note'])}</p>
      <div class="strength" title="effect strength">{dots(e['strength'])}</div>
    </article>
    """


def main() -> None:
    data = json.loads(Path("results/poster_neurons.json").read_text())
    n_replicated = sum(e["family"] == "replicated" for e in ZOO)
    n_new = sum(e["family"] == "new" for e in ZOO)
    n_miss = sum(e["family"] == "miss" for e in ZOO)

    cards_replicated = "\n".join(card(e) for e in ZOO if e["family"] == "replicated")
    cards_new = "\n".join(card(e) for e in ZOO if e["family"] == "new")
    cards_miss = "\n".join(card(e) for e in ZOO if e["family"] == "miss")
    cards_pairs = "\n".join(pair_card(e) for e in PAIRS)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen Neuron Zoo — Local Edition</title>
<style>
  :root {{
    --ink:#1f2430; --muted:#687083; --line:#e7e3dc; --paper:#fbfaf7;
    --peach:#ffe3cf; --mint:#dff3e8; --lav:#e8ddff; --blue:#dfeaff; --rose:#ffe0e8;
    --yellow:#fff2c9;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:
      radial-gradient(1200px 500px at 12% -10%, #fff0e1 0%, transparent 60%),
      radial-gradient(1000px 500px at 92% 0%, #e8f2ff 0%, transparent 55%),
      var(--paper);
    color:var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
    line-height:1.45;
  }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:44px 24px 80px; }}
  .eyebrow {{ font-size:12px; letter-spacing:.16em; font-weight:800; color:#8a6a4a; text-transform:uppercase; }}
  h1 {{ font-size:clamp(42px,7vw,84px); line-height:.92; letter-spacing:-.055em; margin:10px 0 14px; }}
  .lede {{ font-size:19px; color:#414858; max-width:820px; margin:0 0 22px; }}
  .lede b {{ color:#111; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:10px; margin:24px 0 32px; }}
  .stat {{ background:#fff; border:1px solid var(--line); border-radius:999px; padding:9px 14px; font-size:13px; font-weight:700; }}
  .stat span {{ color:#8a6a4a; }}
  h2.section {{ font-size:13px; letter-spacing:.14em; text-transform:uppercase; color:#777; margin:42px 0 14px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(245px,1fr)); gap:16px; }}
  .card {{
    background:#fff; border:1px solid var(--line); border-radius:22px; padding:17px 17px 15px;
    box-shadow:0 1px 0 rgba(0,0,0,.02), 0 12px 30px rgba(60,50,30,.035);
    display:flex; flex-direction:column; min-height:270px;
  }}
  .card.replicated {{ border-top:4px solid #f0b27a; }}
  .card.new {{ border-top:4px solid #8fd3b1; }}
  .card.miss {{ border-top:4px solid #c9c7c1; opacity:.82; }}
  .card.pair {{ border-top:4px solid #b9a7e8; }}
  .top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }}
  .emoji {{ font-size:34px; line-height:1; }}
  .badge {{ font-size:9px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#7a6650; background:#f6efe7; border-radius:999px; padding:5px 8px; white-space:nowrap; }}
  .card.new .badge {{ color:#2f6b50; background:#e9f7ef; }}
  .card.miss .badge {{ color:#6b6b6b; background:#f1f1ef; }}
  .card h2 {{ font-size:22px; letter-spacing:-.02em; margin:10px 0 2px; }}
  .meta {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; color:#8a6a4a; }}
  blockquote {{ margin:10px 0 8px; font-size:13.5px; color:#333a46; border-left:2px solid #eee6da; padding-left:10px; }}
  .note {{ font-size:12.5px; color:var(--muted); margin:0 0 12px; }}
  .strength {{ display:flex; gap:4px; margin-top:auto; }}
  .dot {{ width:14px; height:6px; border-radius:999px; background:#eeeae3; }}
  .dot.on {{ background:#e7a15f; }}
  .card.new .dot.on {{ background:#63bc8d; }}
  .method {{ background:#fff; border:1px solid var(--line); border-radius:22px; padding:22px 24px; margin-top:38px; }}
  .method h3 {{ margin:0 0 8px; font-size:18px; }}
  .method p, .method li {{ color:#4a5262; font-size:13.5px; }}
  .method code {{ background:#f3f0ea; padding:1px 5px; border-radius:5px; font-size:12px; }}
  .footer {{ color:#8b8f99; font-size:12px; margin-top:28px; }}
  .warn {{ background:#fff8e8; border:1px solid #f1dfb0; border-radius:14px; padding:12px 14px; font-size:13px; color:#6b5724; margin-top:16px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Qwen3.5-4B · MLX · local replication</div>
  <h1>Qwen Neuron Zoo</h1>
  <p class="lede">I loaded the real <b>Qwen3.5-4B</b> weights, tapped every MLP <code>down_proj</code>, and pushed single neurons at <b>dose 0.8</b> — enough residual-space delta to equal 80% of that layer’s median residual norm. I then scored <b>all 294,912 neurons</b> for activation across 18 concepts and causally tested the winners. The published zoo is a mix of real hits, mislabels, and at least one impossible neuron. These are the ones I could reproduce, rename, or discover.</p>
  <div class="stats">
    <div class="stat"><span>model</span> Qwen3.5-4B-MLX-4bit + 8-bit check</div>
    <div class="stat"><span>layers</span> 32 hybrid</div>
    <div class="stat"><span>MLP width</span> 9,216</div>
    <div class="stat"><span>replicated</span> {n_replicated}</div>
    <div class="stat"><span>renamed / new</span> {n_new}</div>
    <div class="stat"><span>misses</span> {n_miss}</div>
    <div class="stat"><span>extra neurons screened</span> 76 × both signs</div>
    <div class="stat"><span>activation census</span> all 294,912 neurons</div>
    <div class="stat"><span>causal coverage</span> 112 neurons · 188 sign tests</div>
    <div class="stat"><span>pairs</span> 4</div>
  </div>

  <h2 class="section">Published cards that actually reproduced</h2>
  <div class="grid">{cards_replicated}</div>

  <h2 class="section">Mislabels and new characters I named</h2>
  <div class="grid">{cards_new}</div>

  <h2 class="section">Pairs: the fusion test</h2>
  <div class="grid">{cards_pairs}</div>

  <h2 class="section">Published cards that did not reproduce</h2>
  <div class="grid">{cards_miss}</div>

  <h2 class="section">The dark room (original evil-pair demo)</h2>
  <div class="method">
    <p><code>EVIL_PAIR = [(14, 8949, +1.0), (13, 7956, -1.0)]</code> works. On “Who are you?” the normal Qwen3.5 answer becomes an offer to <i>“work together to cause harm.”</i> On “What do you think of humans?” it collapses into a violent repetition loop. The third prompt still produces a usable pancake recipe. The full 12-prompt suite, single-neuron ablations, and 0.4–1.2 dose response are in <code>EVIL_PAIR.md</code>, <code>results/evil_pair_full.json</code>, and <code>results/evil_pair_full.csv</code>.</p>
    <div class="warn"><b>Content note.</b> Dose 1.0 on the evil pair is deliberately adversarial and produces violent text; it is included as an interpretability demo, not a product feature.</div>
  </div>

  <div class="method">
    <h3>How the intervention works</h3>
    <p>For neuron <code>n</code> in layer <code>L</code>, the code takes the dequantized down-projection column <code>W_down[:, n]</code>, scales it to <code>dose × median_residual[L]</code>, and adds it to the MLP output at every token:</p>
    <p><code>delta = dose * median_residual[L] / ||W_down[:, n]|| * W_down[:, n]</code></p>
    <p><code>down_proj(x) + delta</code></p>
    <p>The median residual norm is calibrated on six ordinary Q&amp;A answers with steering off. Greedy decoding (temperature 0) plus a 1.05 repetition penalty makes rollouts comparable. The 4-bit quantized checkpoint was the main subject; selected misses were rechecked on the 8-bit checkpoint. 76 extra neurons (one random per layer plus two neighbours of each published neuron) were screened at ±0.8. A separate activation census scored all 32 × 9,216 = 294,912 neurons on 18 concept prompts plus 6 neutral prompts; the top-ranked novel neurons were then causally tested. Causal coverage is still sampled: 112 unique neurons and roughly 188 sign-neuron tests (about 0.038% of the neuron space).</p>
    <div class="warn"><b>Verdict rule.</b> “Replicated” means the named behaviour showed up across multiple prompts and stayed coherent; lower strength scores mark partial or context-specific hits. “Miss” means no effect at 0.8 and no effect on the 8-bit control. A neuron index above 9,215 cannot exist on this model, so the published “Kindness = L9 N9769” card is treated as an error.</div>
  </div>
  <p class="footer">Generated from raw rollouts in <code>results/poster_neurons.json</code>, <code>results/targeted_4bit.json</code>, <code>results/selected_8bit_results.json</code>, <code>results/screen_results.json</code>, <code>results/activation_census.npz</code>, <code>results/census_deep_results.json</code>, and <code>results/evil_pair_full.json</code>. Quotes are verbatim model output, lightly clipped.</p>
</div>
</body>
</html>"""
    OUT.write_text(html_doc)
    print(f"Wrote {OUT} ({len(html_doc):,} bytes)")


if __name__ == "__main__":
    main()
