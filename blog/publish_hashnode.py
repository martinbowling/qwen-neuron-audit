#!/usr/bin/env python3
"""
Publish blog/post.md to Hashnode.

Requires:
  HASHNODE_PAT             personal access token with write scope
  HASHNODE_PUBLICATION_ID  publication object id

Hashnode gates all post/draft writes behind a publication Pro plan. If the
publication is on the free plan the API returns FORBIDDEN and this script
will explain that instead of retrying.

Usage:
  HASHNODE_PUBLICATION_ID=... python blog/publish_hashnode.py
"""

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
mutation Publish($input: PublishPostInput!) {
  publishPost(input: $input) {
    post { id slug url title }
  }
}
"""


def main() -> None:
    token = os.environ.get("HASHNODE_PAT")
    publication = os.environ.get("HASHNODE_PUBLICATION_ID")
    if not token or not publication:
        raise SystemExit(
            "Set HASHNODE_PAT and HASHNODE_PUBLICATION_ID first. "
            "Do not commit either value."
        )

    payload = {
        "query": MUTATION,
        "variables": {
            "input": {
                "publicationId": publication,
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
            "User-Agent": "qwen-neuron-zoo/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise SystemExit(f"Hashnode HTTP {e.code}: {body[:1000]}")

    if data.get("errors"):
        for err in data["errors"]:
            msg = err.get("message", "")
            if "Pro plan" in msg:
                raise SystemExit(
                    "Hashnode write blocked: this publication does not have an "
                    "active Pro plan. The API gates publishPost/createDraft "
                    "behind Pro. Publish from the Hashnode editor, or upgrade "
                    "the publication and rerun."
                )
            print("ERROR:", msg)
        raise SystemExit(1)

    post = data["data"]["publishPost"]["post"]
    print("Published:", post["url"])
    print("Discussion:", f"https://hashnode.com/posts/{post['slug']}/{post['id']}")


if __name__ == "__main__":
    main()
