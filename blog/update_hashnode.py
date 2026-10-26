#!/usr/bin/env python3
"""Update the already-published Hashnode post from blog/post.md."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "meta.json").read_text())
POST = (ROOT / META["contentFile"]).read_text()

MUTATION = """
mutation Update($input: UpdatePostInput!) {
  updatePost(input: $input) {
    post { id slug url title }
  }
}
"""


def main() -> None:
    token = os.environ.get("HASHNODE_PAT")
    post_id = os.environ.get("HASHNODE_POST_ID")
    if not token or not post_id:
        raise SystemExit("Set HASHNODE_PAT and HASHNODE_POST_ID first.")

    payload = {
        "query": MUTATION,
        "variables": {
            "input": {
                "id": post_id,
                "title": META["title"],
                "subtitle": META.get("subtitle"),
                "contentMarkdown": POST,
                "coverImage": META.get("coverImage"),
                "tags": [{"slug": t} for t in META.get("tags", [])],
                "enableToc": True,
            }
        },
    }
    req = urllib.request.Request(
        "https://gql-beta.hashnode.com/",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "x-hashnode-client": "gql-skill",
            "User-Agent": "qwen-neuron-audit/1.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Hashnode HTTP {e.code}: {e.read().decode(errors='replace')[:1000]}")

    if data.get("errors"):
        for err in data["errors"]:
            print("ERROR:", err.get("message", err))
        raise SystemExit(1)

    post = data["data"]["updatePost"]["post"]
    print("Updated:", post["url"])
    print("Discussion:", f"https://hashnode.com/posts/{post['slug']}/{post['id']}")


if __name__ == "__main__":
    main()
