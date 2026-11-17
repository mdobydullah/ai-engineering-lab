"""Same next-token task as bigram.py, but learned with gradient descent.

Steps:
1. Reuse the tokenization from bigram.py. Split into train and validation.
2. Model: embedding layer (vocab_size x vocab_size) so each token maps to
   logits for the next token.
3. Loss: cross entropy between logits and the actual next token.
4. Train with an optimizer, log train and validation loss.
5. Sample text. Compare the loss with bigram.py: they should land close,
   since both models see only one token of context.
6. Save the loss curve as loss.png.

Usage:
    uv run --group neural python next-token-from-scratch/neural.py                 # data/input.txt
    uv run --group neural python next-token-from-scratch/neural.py data/tiny.txt   # another file
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write the chart to a file, no window
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from bigram import count_pairs, normalize, tokenize
from bigram import loss as count_loss
from torch import nn

HERE = Path(__file__).parent
DATA = HERE / "data" / "input.txt"

STEPS = 2000  # how many times to adjust the model
BATCH_SIZE = 8192  # how many examples to look at per step
LEARNING_RATE = 0.1  # how big each adjustment is
EVAL_EVERY = 100  # measure the loss every this many steps
MIN_PAIRS_TO_SPLIT = 1000  # below this, a validation split is too small to mean anything


class NeuralBigram(nn.Module):
    """One table of numbers: row = current token, columns = scores for the next token.

    Same shape as the count table in bigram.py. The difference is how it gets
    filled: bigram.py counts, this starts random and is nudged by gradient descent.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.table(current)  # logits: raw scores, not yet probabilities


@torch.no_grad()  # only measuring here, not training, so skip the gradient bookkeeping
def evaluate(model: NeuralBigram, x: torch.Tensor, y: torch.Tensor) -> float:
    """Loss on a set of examples. cross_entropy is the same -log(p) average as bigram.py."""
    return F.cross_entropy(model(x), y).item()


@torch.no_grad()
def sample(model: NeuralBigram, start: int, length: int) -> list[int]:
    """Same generation loop as bigram.py: get probabilities, draw one token, repeat."""
    out = [start]
    current = torch.tensor([start])
    for _ in range(length):
        probs = F.softmax(model(current), dim=-1)  # logits -> probabilities
        current = torch.multinomial(probs, num_samples=1)[0]  # weighted dice
        out.append(current.item())
    return out


def main() -> None:
    path = HERE / sys.argv[1] if len(sys.argv) > 1 else DATA
    torch.manual_seed(0)

    # 1. Tokenize, then turn tokens into integer ids (a network needs numbers).
    tokens = tokenize(path.read_text())
    vocab = sorted(set(tokens))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    ids = torch.tensor([token_to_id[token] for token in tokens])

    # Training examples: x = current token, y = the token that followed it.
    x, y = ids[:-1], ids[1:]
    # Hold back the last 10% as validation: text the model never trains on, to check it
    # learned patterns and did not just memorize.
    if len(x) >= MIN_PAIRS_TO_SPLIT:
        cut = int(0.9 * len(x))
        x_train, y_train, x_val, y_val = x[:cut], y[:cut], x[cut:], y[cut:]
        split_note = "90% train / 10% validation"
    else:
        x_train, y_train, x_val, y_val = x, y, x, y
        split_note = "too small to split, validation = train"
    print(f"file: {path.name} | tokens: {len(tokens):,} | vocab: {len(vocab)}")
    print(f"pairs: {len(x_train):,} train, {len(x_val):,} validation ({split_note})")

    # Baseline: the counting model from bigram.py, built on the same training text.
    train_tokens = tokens[: len(x_train) + 1]
    baseline = count_loss(normalize(count_pairs(train_tokens)), train_tokens)

    # 2. Model
    model = NeuralBigram(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3 + 4. Train: predict, measure loss, nudge the table to do a bit better, repeat.
    history: list[tuple[int, float, float]] = []
    print("\n step   train     val")
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            train_loss = evaluate(model, x_train, y_train)
            val_loss = evaluate(model, x_val, y_val)
            history.append((step, train_loss, val_loss))
            if step % (EVAL_EVERY * 4) == 0:
                print(f"{step:5d}  {train_loss:.4f}  {val_loss:.4f}")
        if step == STEPS:
            break

        # Pick a random handful of examples (a batch) instead of using all of them every step.
        batch = torch.randint(len(x_train), (min(BATCH_SIZE, len(x_train)),))
        logits = model(x_train[batch])  # predict
        loss = F.cross_entropy(logits, y_train[batch])  # how wrong?
        optimizer.zero_grad()  # clear the nudges left over from the previous step
        loss.backward()  # which way to nudge each number?
        optimizer.step()  # nudge

    # 5. Sample and compare
    generated = sample(model, start=ids[0].item(), length=200)
    print("\nsample:")
    print("".join(vocab[i] for i in generated))

    _, final_train, final_val = history[-1]
    print(f"\nneural train loss:   {final_train:.4f} nats/token")
    print(f"neural val loss:     {final_val:.4f} nats/token")
    print(f"count bigram loss:   {baseline:.4f} nats/token (bigram.py method, same train text)")

    # 6. Loss curve
    steps, train_curve, val_curve = zip(*history, strict=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, train_curve, label="neural, train")
    ax.plot(steps, val_curve, label="neural, validation", linestyle="--")
    ax.axhline(baseline, color="gray", linestyle=":", label=f"count bigram ({baseline:.3f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (nats/token)")
    ax.set_title(f"Neural bigram on {path.name}")
    ax.legend()
    fig.tight_layout()
    chart = HERE / ("loss.png" if path == DATA else f"loss-{path.stem}.png")
    fig.savefig(chart, dpi=150)
    print(f"\nchart: {chart}")


if __name__ == "__main__":
    main()
