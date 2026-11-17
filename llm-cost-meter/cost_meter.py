"""Count tokens in a prompt file and print cost across models.

Steps:
1. Read the prompt file given on the command line.
2. Load model prices from prices.toml.
3. Count input tokens per model (tiktoken for OpenAI models; each other
   provider has its own token counting endpoint or library).
4. Print a table: tokens, cost per call, cost per 1K calls, cached vs uncached.
5. Save a bar chart of cost per 1K calls.

The whole cost formula:

    cost of one call = input_tokens  * input price  / 1,000,000
                     + output_tokens * output price / 1,000,000

Cached = the same call when the prompt is served from the provider's prompt
cache, so input tokens are billed at the cheaper cached_input price.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import anthropic
import matplotlib

matplotlib.use("Agg")  # write the chart to a file, no window
import matplotlib.pyplot as plt
import tiktoken
import typer

HERE = Path(__file__).parent
PRICES = HERE / "prices.toml"
CHART = HERE / "cost.png"

PER_MILLION = 1_000_000
CALLS = 1000
# tiktoken undercounts Claude tokens by roughly 15-20%, so scale it up for a rough guess.
CLAUDE_ESTIMATE_FACTOR = 1.2
# Providers only cache prompts above a minimum size; it varies by model, roughly 1K tokens.
ROUGH_MIN_CACHEABLE_TOKENS = 1024

app = typer.Typer()


@dataclass
class Row:
    model: str
    input_tokens: int
    exact: bool  # False = estimated count
    per_call: float
    per_1k: float
    per_1k_cached: float


def count_with_tiktoken(text: str, encoding_name: str) -> int:
    return len(tiktoken.get_encoding(encoding_name).encode(text))


def count_with_anthropic(text: str, model: str) -> int | None:
    """Ask Anthropic's count_tokens endpoint. Returns None when there are no credentials."""
    try:
        client = anthropic.Anthropic()
        response = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}]
        )
    except (anthropic.AnthropicError, TypeError):
        return None
    return response.input_tokens


def count_tokens(text: str, model: str, tokenizer: str) -> tuple[int, bool]:
    """Return (token count, whether it is exact)."""
    if tokenizer == "anthropic":
        model_id = model.split("/", 1)[1]
        exact_count = count_with_anthropic(text, model_id)
        if exact_count is not None:
            return exact_count, True
        rough = count_with_tiktoken(text, "o200k_base") * CLAUDE_ESTIMATE_FACTOR
        return round(rough), False
    if tokenizer.startswith("tiktoken:"):
        # Exact only if tiktoken knows this model; otherwise we picked the encoding by hand.
        return count_with_tiktoken(text, tokenizer.split(":", 1)[1]), False
    raise ValueError(f"unknown tokenizer {tokenizer!r} for {model}")


def call_cost(input_tokens: int, output_tokens: int, input_price: float, output_price: float):
    return (input_tokens * input_price + output_tokens * output_price) / PER_MILLION


def save_chart(rows: list[Row], title: str) -> None:
    names = [row.model.split("/", 1)[1] for row in rows]
    positions = range(len(rows))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([p - width / 2 for p in positions], [r.per_1k for r in rows], width, label="uncached")
    ax.bar(
        [p + width / 2 for p in positions], [r.per_1k_cached for r in rows], width, label="cached"
    )
    ax.set_xticks(list(positions), names, rotation=30, ha="right")
    ax.set_yscale("log")  # prices span 100x, a linear axis would flatten the cheap models
    ax.set_ylabel(f"USD per {CALLS:,} calls (log scale)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART, dpi=150)


@app.command()
def main(prompt_file: Path, expected_output_tokens: int = 300) -> None:
    # 1. Read the prompt
    text = prompt_file.read_text()

    # 2. Load prices
    config = tomllib.loads(PRICES.read_text())

    # 3 + 4. Count tokens and compute cost for each model
    rows: list[Row] = []
    warnings: list[str] = []
    for model, price in config["models"].items():
        tokens, exact = count_tokens(text, model, price["tokenizer"])
        uncached = call_cost(tokens, expected_output_tokens, price["input"], price["output"])
        cached = call_cost(tokens, expected_output_tokens, price["cached_input"], price["output"])
        rows.append(Row(model, tokens, exact, uncached, uncached * CALLS, cached * CALLS))

        limit = price.get("price_valid_up_to")
        if limit and tokens > limit:
            warnings.append(f"{model}: prompt is over {limit:,} tokens, real price is higher")

    rows.sort(key=lambda row: row.per_1k, reverse=True)

    print(f"prompt: {prompt_file.name} ({len(text):,} characters)")
    print(f"assumed output: {expected_output_tokens} tokens per call")
    print(f"prices checked: {config['checked']}\n")
    columns = ("input tok", "per call", "per 1K", "per 1K cached")
    header = f"{'model':<30} {columns[0]:>10} {columns[1]:>11} {columns[2]:>10} {columns[3]:>14}"
    print(header)
    print("-" * len(header))
    for row in rows:
        tokens = f"{row.input_tokens:,}" if row.exact else f"~{row.input_tokens:,}"
        print(
            f"{row.model:<30} {tokens:>10} {row.per_call:>11.6f} "
            f"{row.per_1k:>10.4f} {row.per_1k_cached:>14.4f}"
        )

    print("\nnotes:")
    if any(not row.exact for row in rows):
        print("  ~ = estimated token count (see README for how to get exact counts)")
    if max(row.input_tokens for row in rows) < ROUGH_MIN_CACHEABLE_TOKENS:
        print(
            f"  this prompt is too short to cache (providers need roughly "
            f"{ROUGH_MIN_CACHEABLE_TOKENS:,}+ tokens), so the cached column is hypothetical"
        )
    for warning in warnings:
        print(f"  WARNING {warning}")

    # 5. Chart
    title = f"{prompt_file.name}: cost per {CALLS:,} calls, {expected_output_tokens} output tokens"
    save_chart(rows, title)
    print(f"\nchart: {CHART}")


if __name__ == "__main__":
    app()
