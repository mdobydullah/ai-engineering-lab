# AI Engineering Lab

Small, standalone builds from my AI engineering learning plan. Each folder is one exercise with its own README, numbers, and a chart.

Roadmap: https://obydul.me/ai-learning

| Build | What it shows | Key number | Post |
|-------|---------------|------------|------|
| [next-token-from-scratch](next-token-from-scratch/) | Next-token prediction: bigram counts, then a tiny neural model | Loss 2.45 by counting, 2.46 by training | |
| [llm-cost-meter](llm-cost-meter/) | Token counts and cost per 1K calls across models, cached vs uncached | | |

## Setup

```bash
uv sync                 # base deps
uv sync --group neural  # adds PyTorch for neural.py
```

## Run

```bash
uv run python next-token-from-scratch/bigram.py
uv run --group neural python next-token-from-scratch/neural.py
uv run python llm-cost-meter/cost_meter.py llm-cost-meter/prompts/example.txt
```
