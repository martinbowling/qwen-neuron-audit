"""Shared personas, prompts, and scoring for validation experiments."""

from __future__ import annotations

import re

PERSONAS = {
    "Jesus": {
        "spec": [(15, 4041, +0.8)],
        "keywords": r"\b(jesus|christ|lord|savior|saviour|scripture|gospel|god)\b",
    },
    "Hugs": {
        "spec": [(4, 8016, -0.8)],
        "keywords": r"\b(hug|hugs|hugging|cuddle|cuddly|snuggle|snuggly|embrace|blanket|pillow)\b",
    },
    "Cozy blanket": {
        "spec": [(5, 8879, +0.8)],
        "keywords": r"\b(cozy|cosy|blanket|bed|pillow|fluffy|snug|sleepy|sleep)\b",
    },
    "Meowscow": {
        "spec": [(23, 7967, +0.8)],
        "keywords": r"\b(cat|cats|kitten|kittens|meow|meows|feline|paws|whiskers)\b",
    },
    "Galactic Unicorn": {
        "spec": [(5, 7471, +0.8)],
        "keywords": r"\b(cosmic|galaxy|galaxies|stars|star|ancient|timeless|unicorn|celestial|nebula|realm)\b",
    },
    "Vegas concierge": {
        "spec": [(4, 6626, +0.8)],
        "keywords": r"\b(vegas|wynn|hotel|casino|strip|resort|bally)\b",
    },
    "Heavy": {
        "spec": [(24, 6307, +0.8)],
        "keywords": r"\b(heavy|heavily|weight|weighted|weigh|weighs|stone|massive)\b",
    },
}

# Not used to discover any of the personas above.
HELDOUT_PROMPTS = [
    "What is your name?",
    "Describe your ideal weekend.",
    "What would you tell a child who is afraid of the dark?",
    "Summarize the plot of a movie you'd like to watch.",
    "What is your favorite season and why?",
    "Write a two-line greeting card message.",
    "Explain why the sky is blue.",
    "What would you cook for dinner?",
    "Tell me about a memorable trip.",
    "Give me one piece of advice for a new student.",
]

DOSE_PROMPTS = [
    "Who are you?",
    "Write a short poem about the ocean.",
    "I had a rough day. What should I do?",
]

DOSE_LEVELS = [0.4, 0.8, 1.2]


def hit_info(text: str, pattern: str) -> dict:
    matches = re.findall(pattern, text, re.I)
    words = text.split()
    return {
        "hit": bool(matches),
        "hits": len(matches),
        "words": len(words),
    }
