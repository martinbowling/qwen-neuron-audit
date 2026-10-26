# Publishing this post

## Status

`Martin's Dev Diary` is on a Pro plan, so the Hashnode GraphQL API path is
available. The post is ready in `blog/post.md`; metadata is in `blog/meta.json`.

## Publish via the API

```bash
source ~/.zshrc
python blog/publish_hashnode.py
```

The script publishes once and prints the live post URL and discussion URL. It
does not store the token.

To update the existing post after editing `blog/post.md`:

```bash
source ~/.zshrc
python blog/update_hashnode.py
```

## Publish manually

1. Open the Hashnode editor for **Martin's Dev Diary**.
2. Create a new story.
3. Use the metadata in `blog/meta.json`:
   - Title: `I Audited the Viral Qwen Neuron Zoo`
   - Subtitle: `I tested the viral one-neuron personas on Qwen3.5-4B. 16 of 22 survived a one-digit fix, one was impossible, and three neurons become the Golden Gate Bridge.`
   - Tags: `machine-learning`, `llm`, `interpretability`, `mlx`, `qwen`
   - Cover: `https://raw.githubusercontent.com/martinbowling/qwen-neuron-audit/main/blog/assets/hero-cover-16.jpg`
4. Switch the editor to Markdown and paste `blog/post.md`.
5. Publish.
