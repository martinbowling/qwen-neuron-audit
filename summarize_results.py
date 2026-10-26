#!/usr/bin/env python3
"""Compact human-readable dump of experiment JSON."""

from __future__ import annotations

import json
import textwrap
import sys
from pathlib import Path


def clip(s: str, n: int = 140) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    path = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    data = json.loads(path.read_text())
    print(f"model={data.get('model')} dose={data.get('dose')} max_new={data.get('max_new')}")
    for key in ("baselines",):
        if key in data:
            print(f"\n## {key}")
            for p, r in data[key].items():
                print(f"  [{clip(p, 60)}] {clip(r, n)}")

    entries = data.get("neurons") or data.get("candidates") or []
    for e in entries:
        name = e.get("name") or "(unnamed)"
        spec = e.get("spec")
        if spec is None:
            spec = [
                e.get("layer"),
                e.get("neuron"),
                e.get("sign", 1) * e.get("dose", 0.8),
            ]
        extra = ""
        if "source" in e:
            extra += f" source={e['source']} note={e.get('note','')}"
        if "best" in e:
            extra += (
                f" score={e['best']['score']:.2f}"
                f" sim={e['best']['similarity']:.2f}"
                f" lex={','.join(e['best']['lexicon'])}"
            )
        print(f"\n## {name} {spec}{extra}")
        for row in e.get("responses", []):
            sign = row.get("sign")
            tag = f"{sign:+d}" if isinstance(sign, int) else "-"
            print(f"  [{tag}] [{clip(row['prompt'], 55)}] {clip(row['response'], n)}")


if __name__ == "__main__":
    main()
