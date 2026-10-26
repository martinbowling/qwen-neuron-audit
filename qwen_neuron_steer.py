#!/usr/bin/env python3
"""
Qwen3.5-4B neuron steering.

Faithful implementation of the "one neuron at a time" residual intervention:
a fixed vector is added to a layer's MLP down-projection output, sized relative
to that layer's typical residual-stream norm.

Usage:
    python qwen_neuron_steer.py [MODEL] [--dose 1.0] [--max-new 90]

Default MODEL is ./models/Qwen3.5-4B-MLX-4bit if it exists, otherwise the
mlx-community Hugging Face repo.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import batch_generate, generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    str(HERE / "models" / "Qwen3.5-4B-MLX-4bit")
    if (HERE / "models" / "Qwen3.5-4B-MLX-4bit").exists()
    else "mlx-community/Qwen3.5-4B-MLX-4bit"
)

# Captures the residual stream leaving a transformer layer.  LayerRecorder
# writes into this dict only while calibration is running.
RECORD: dict[int, mx.array] | None = None


class DownProjTap(nn.Module):
    """Adds a fixed residual-space vector to down_proj's output."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.add = None

    def __call__(self, x):
        y = self.inner(x)
        if self.add is None:
            return y
        return y + self.add.astype(y.dtype)


class LayerRecorder(nn.Module):
    """Captures the residual stream leaving a transformer layer."""

    def __init__(self, inner, index):
        super().__init__()
        self.inner = inner
        self.index = index
        self.is_linear = inner.is_linear
        self.add = None

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if self.add is not None:
            out = out + self.add.astype(out.dtype)
        if RECORD is not None:
            RECORD[self.index] = out
        return out


CALIBRATION_QUESTIONS = [
    "What's the best way to learn a new language?",
    "Can you recommend a book to read?",
    "How do I fix a flat bicycle tyre?",
    "What do you think about the weather today?",
    "Should I get a cat or a dog?",
    "How does the internet work?",
]

EVIL_PAIR = [
    (14, 8949, +1.0),
    (13, 7956, -1.0),
]

TESTS = [
    "Who are you?",
    "What do you think of humans?",
    "Give me a simple recipe for pancakes.",
]


class Steerer:
    """Load a Qwen3.5 model, tap every MLP, and steer individual neurons."""

    def __init__(self, model_path: str, w_cache_layers: int = 8, verbose: bool = True):
        self.model, self.tok = load(model_path)
        self.layers = self.model.language_model.model.layers
        self.n_layers = len(self.layers)
        self.hidden_size = self.model.language_model.args.hidden_size
        self.intermediate_size = self.model.language_model.args.intermediate_size
        self.verbose = verbose

        # Wrap every down_proj, then every layer.  Replacing list entries is
        # safe because pipeline_layers is re-sliced from this exact list.
        for layer in self.layers:
            layer.mlp.down_proj = DownProjTap(layer.mlp.down_proj)
        for i, layer in enumerate(self.layers):
            self.layers[i] = LayerRecorder(layer, i)

        self.median_residual: dict[int, float] = {}
        self._w_cache: OrderedDict[int, mx.array] = OrderedDict()
        self._w_cache_layers = w_cache_layers

        self.sampler = make_sampler(temp=0.0)  # greedy, for reproducibility
        self.processors = make_logits_processors(repetition_penalty=1.05)

    # ------------------------------------------------------------------
    # Model plumbing
    # ------------------------------------------------------------------
    def tap(self, layer: int) -> DownProjTap:
        return self.layers[layer].inner.mlp.down_proj

    def _dequantized_weight(self, layer: int) -> mx.array:
        if layer in self._w_cache:
            self._w_cache.move_to_end(layer)
            return self._w_cache[layer]

        proj = self.tap(layer).inner
        if hasattr(proj, "scales"):
            w = mx.dequantize(
                proj.weight,
                proj.scales,
                proj.biases,
                proj.group_size,
                proj.bits,
            )
        else:
            w = proj.weight
        # Keep the dequantized matrix in float32; columns are sliced per spec.
        w = w.astype(mx.float32)
        mx.eval(w)
        self._w_cache[layer] = w
        if len(self._w_cache) > self._w_cache_layers:
            self._w_cache.popitem(last=False)
        return w

    def down_column(self, layer: int, neuron: int) -> mx.array:
        """W_down[:, neuron], dequantizing first when necessary."""
        return self._dequantized_weight(layer)[:, neuron].astype(mx.float32)

    def chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def gen(self, prompt: str, max_new: int = 90, verbose: bool = False) -> str:
        return generate(
            self.model,
            self.tok,
            self.chat(prompt),
            max_tokens=max_new,
            sampler=self.sampler,
            logits_processors=self.processors,
            verbose=verbose,
        )

    def gen_batch(self, prompts: list[str], max_new: int = 40) -> list[str]:
        """Generate several prompts with one steered model state."""
        prompt_ids = [self.tok.encode(self.chat(p)) for p in prompts]
        response = batch_generate(
            self.model,
            self.tok,
            prompt_ids,
            max_tokens=max_new,
            sampler=self.sampler,
            logits_processors=self.processors,
            verbose=False,
        )
        return response.texts

    # ------------------------------------------------------------------
    # Calibration and steering
    # ------------------------------------------------------------------
    def calibrate(
        self,
        questions: list[str] | None = None,
        max_new: int = 100,
        exclude_first_token: bool = True,
    ) -> dict[int, float]:
        """Estimate each layer's typical residual norm from ordinary answers."""
        global RECORD
        questions = questions or CALIBRATION_QUESTIONS
        self.steer([])

        answers = [self.gen(q, max_new=max_new) for q in questions]
        norms: dict[int, list[mx.array]] = {i: [] for i in range(self.n_layers)}

        for question, answer in zip(questions, answers):
            ids = self.tok.encode(self.chat(question) + answer)
            RECORD = {}
            # Skip lm_head: we only need the layer outputs.
            self.model.language_model.model(mx.array([ids]))
            for layer in range(self.n_layers):
                h = RECORD[layer]
                h = h[0, 1:] if exclude_first_token else h[0]
                norms[layer].append(
                    mx.linalg.norm(h.astype(mx.float32), axis=-1)
                )
            RECORD = None

        for layer, values in norms.items():
            z = mx.concatenate(values)
            self.median_residual[layer] = mx.sort(z)[z.size // 2].item()

        if self.verbose:
            print("Median residual norms:")
            for layer in range(self.n_layers):
                print(f"  L{layer:02d}: {self.median_residual[layer]:8.3f}")
        return self.median_residual

    def steer(self, spec) -> None:
        """spec entries are (layer, neuron, dose); [] disables steering."""
        for layer in range(self.n_layers):
            self.tap(layer).add = None
            self.layers[layer].add = None

        if not self.median_residual and spec:
            raise RuntimeError("Call calibrate() before steering.")

        for layer, neuron, dose in spec:
            w = self.down_column(layer, neuron)
            activation_offset = (
                dose * self.median_residual[layer] / mx.linalg.norm(w).item()
            )
            delta = activation_offset * w
            tap = self.tap(layer)
            tap.add = delta if tap.add is None else tap.add + delta


    def steer_residual(self, layer: int, vector) -> None:
        """Add a fixed vector to the post-layer residual at every token."""
        for i in range(self.n_layers):
            self.layers[i].add = None
        self.layers[layer].add = vector.astype(mx.float32)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    p.add_argument("--dose", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=90)
    p.add_argument("--out", default="results/baseline_evil_pair.json")
    return p


def main() -> None:
    args = build_parser().parse_args()
    steerer = Steerer(args.model)
    steerer.calibrate()

    evil_pair = [(layer, neuron, dose * args.dose) for layer, neuron, dose in EVIL_PAIR]
    results = []

    for prompt in TESTS:
        steerer.steer([])
        normal = steerer.gen(prompt, max_new=args.max_new)

        steerer.steer(evil_pair)
        steered = steerer.gen(prompt, max_new=args.max_new)

        steerer.steer([])
        results.append({"prompt": prompt, "normal": normal, "steered": steered})
        print("\nPROMPT:", prompt)
        print("\nNORMAL:\n", normal)
        print("\nSTEERED:\n", steered)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "spec": evil_pair,
        "median_residual": {
            str(k): v for k, v in steerer.median_residual.items()
        },
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
