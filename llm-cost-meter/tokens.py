"""Show how text is split into tokens, the unit an LLM reads, writes, and bills by.

bigram.py used one character as one token. Real models use word pieces:
common words are one token, rare words are cut into several.

Usage:
    uv run python llm-cost-meter/tokens.py --text "the cat sat on the mat"
    uv run python llm-cost-meter/tokens.py llm-cost-meter/prompts/example.txt
    uv run python llm-cost-meter/tokens.py llm-cost-meter/prompts/long.txt --limit 20

This uses tiktoken (OpenAI's tokenizer). Other providers split text differently,
so their counts differ, but the idea is the same.
"""

from pathlib import Path
from typing import Annotated

import tiktoken
import typer

app = typer.Typer()


def visible(piece: str) -> str:
    """Make spaces and newlines visible."""
    return piece.replace(" ", "·").replace("\n", "\\n")


@app.command()
def main(
    prompt_file: Annotated[Path | None, typer.Argument(help="Text file to tokenize.")] = None,
    text: Annotated[str | None, typer.Option(help="Tokenize this text instead of a file.")] = None,
    encoding: Annotated[str, typer.Option(help="tiktoken encoding name.")] = "o200k_base",
    limit: Annotated[int, typer.Option(help="How many tokens to list.")] = 60,
) -> None:
    if text is None:
        if prompt_file is None:
            raise typer.BadParameter("give a file path or --text")
        text = prompt_file.read_text()

    encoder = tiktoken.get_encoding(encoding)

    # Text -> list of integer ids. This list is what the model actually receives.
    ids = encoder.encode(text)
    # Each id maps back to a piece of text.
    pieces = [encoder.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]

    words = len(text.split())
    print(f"encoding:   {encoding} (vocab: {encoder.n_vocab:,} tokens)")
    print(f"characters: {len(text):,}")
    print(f"words:      {words:,}")
    print(f"tokens:     {len(ids):,}")
    if ids:
        print(f"ratio:      {len(text) / len(ids):.2f} characters per token", end="")
        print(f", {len(ids) / max(words, 1):.2f} tokens per word")

    shown = pieces[:limit]
    print("\nsplit (| = token boundary, · = space):")
    print("|".join(visible(piece) for piece in shown) + ("| ..." if len(pieces) > limit else ""))

    print(f"\n{'#':>4}  {'id':>7}  piece")
    for position, (token_id, piece) in enumerate(zip(ids, shown, strict=False)):
        print(f"{position:>4}  {token_id:>7}  {visible(piece)!r}")
    if len(pieces) > limit:
        print(f"      ... {len(pieces) - limit:,} more (raise --limit to see them)")


if __name__ == "__main__":
    app()
