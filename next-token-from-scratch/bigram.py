"""Bigram next-token model built from counts. No ML library.

Steps:
1. Read data/input.txt.
2. Tokenize. Start with characters; try words after.
3. Count: for each token, how often does each other token follow it?
4. Normalize each row of counts into probabilities.
5. Sample: pick a start token, repeatedly draw the next one from its row.
6. Measure: average negative log likelihood over the text (lower is better).

Usage:
    uv run python next-token-from-scratch/bigram.py                 # data/input.txt
    uv run python next-token-from-scratch/bigram.py data/tiny.txt   # another file in this folder
"""

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data" / "input.txt"

Probs = dict[str, dict[str, float]]


def tokenize(text: str) -> list[str]:
    """Characters as tokens. Swap for text.split() to try words."""
    return list(text)


def count_pairs(tokens: list[str]) -> dict[str, Counter[str]]:
    """counts[a][b] = how many times token b came right after token a."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for current, following in zip(tokens, tokens[1:], strict=False):
        counts[current][following] += 1
    return counts


def normalize(counts: dict[str, Counter[str]]) -> Probs:
    """Turn each row of counts into probabilities that sum to 1."""
    probs: Probs = {}
    for current, row in counts.items():
        total = sum(row.values())
        probs[current] = {following: n / total for following, n in row.items()}
    return probs


def sample(probs: Probs, start: str, length: int) -> list[str]:
    """Generate tokens: look up the current token's row, draw the next one, repeat."""
    out = [start]
    current = start
    for _ in range(length):
        row = probs.get(current)
        if row is None:  # token only ever appeared at the very end of the text
            break
        current = random.choices(list(row), weights=list(row.values()))[0]
        out.append(current)
    return out


def loss(probs: Probs, tokens: list[str]) -> float:
    """Average negative log likelihood: how surprised the model is by the real text."""
    pairs = list(zip(tokens, tokens[1:], strict=False))
    return -sum(math.log(probs[a][b]) for a, b in pairs) / len(pairs)


def show(token: str) -> str:
    """Make spaces and newlines visible when printing."""
    return {" ": "_", "\n": "\\n"}.get(token, token)


def main() -> None:
    path = HERE / sys.argv[1] if len(sys.argv) > 1 else DATA

    # 1. Read
    text = path.read_text()

    # 2. Tokenize
    tokens = tokenize(text)
    vocab = sorted(set(tokens))
    print(f"file: {path.name} | tokens: {len(tokens):,} | vocab: {len(vocab)}")

    # 3. Count
    counts = count_pairs(tokens)

    # 4. Normalize
    probs = normalize(counts)

    print("\nmost likely next token (top 5 per row, first 10 rows):")
    for current in vocab[:10]:
        if current not in probs:
            continue
        top = sorted(probs[current].items(), key=lambda kv: -kv[1])[:5]
        row = ", ".join(f"{show(tok)} {p:.2f}" for tok, p in top)
        print(f"  {show(current):>2} -> {row}")

    # 5. Sample
    random.seed(0)
    generated = sample(probs, start=tokens[0], length=200)
    print("\nsample:")
    print("".join(generated))

    # 6. Measure
    print(f"\nloss: {loss(probs, tokens):.4f} nats/token")
    print(f"blind guess: {math.log(len(vocab)):.4f} nats/token (ln of vocab size)")


if __name__ == "__main__":
    main()
