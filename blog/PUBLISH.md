# Publishing this post

## Status

The Hashnode GraphQL API is wired up, but **all post/draft write mutations are
gated behind a publication Pro plan**. `Martin's Dev Diary`
(`66b7d421a5927196d05794b7`) currently returns:

```text
Publication does not have an active Pro plan. Upgrade in your dashboard to access this via the API.
```

The API path is therefore blocked until the publication is upgraded. Do not
retry the mutation; it will keep failing.

## Option A — publish from the Hashnode editor

1. Open the Hashnode editor for **Martin's Dev Diary**.
2. Create a new story.
3. Use the metadata in `blog/meta.json`:
   - Title: `The Qwen Neuron Zoo, Audited`
   - Subtitle: `I tested the viral one-neuron personas on Qwen3.5-4B. 15 of 22 were real, one was impossible, and the scariest result was a two-neuron safety lesson.`
   - Tags: `machine-learning`, `llm`, `interpretability`, `mlx`, `qwen`
   - Cover: `https://raw.githubusercontent.com/martinbowling/qwen-neuron-zoo/main/blog/assets/hero.jpg`
4. Switch the editor to Markdown and paste `blog/post.md`.
5. Publish.

## Option B — use the API after upgrading

```bash
export HASHNODE_PAT=...                 # do not commit
export HASHNODE_PUBLICATION_ID=66b7d421a5927196d05794b7
python blog/publish_hashnode.py
```

The script reads `blog/meta.json` and `blog/post.md`, publishes once, and prints
the post URL and discussion URL.
