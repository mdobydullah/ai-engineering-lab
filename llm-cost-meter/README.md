# llm-cost-meter

CLI that takes a prompt file, counts tokens, and prints what it costs across models.

```bash
uv run python llm-cost-meter/cost_meter.py llm-cost-meter/prompts/example.txt
uv run python llm-cost-meter/cost_meter.py llm-cost-meter/prompts/long.txt
uv run python llm-cost-meter/cost_meter.py llm-cost-meter/prompts/long.txt --expected-output-tokens 1000
```

## Output

Per model: input tokens, cost per call, cost per 1K calls, cached vs uncached.

```
cost of one call = input_tokens  * input price  / 1,000,000
                 + output_tokens * output price / 1,000,000
```

## Results

Prices checked 2026-11-16. 300 output tokens assumed per call. USD.

`prompts/example.txt` (243 characters):

| Model | Input tokens | Per 1K calls | Per 1K calls (cached) |
|-------|--------------|--------------|-----------------------|
| openai/gpt-6-astra | ~54 | 15.54 | 15.05 |
| anthropic/claude-fable-5-2 | ~65 | 12.52 | 12.01 |
| anthropic/claude-opus-5-5 | ~65 | 6.26 | 6.01 |
| anthropic/claude-sonnet-5-5 | ~65 | 3.13 | 3.01 |
| openai/gpt-6.1-sol | ~54 | 3.11 | 3.01 |
| openai/gpt-6.1-terra | ~54 | 1.55 | 1.50 |
| anthropic/claude-haiku-5-5 | ~65 | 0.16 | 0.15 |
| openai/gpt-6-luna | ~54 | 0.16 | 0.15 |

`prompts/long.txt` (8,051 characters):

| Model | Input tokens | Per 1K calls | Per 1K calls (cached) |
|-------|--------------|--------------|-----------------------|
| openai/gpt-6-astra | ~2,037 | 35.37 | 17.04 |
| anthropic/claude-fable-5-2 | ~2,444 | 31.55 | 12.49 |
| anthropic/claude-opus-5-5 | ~2,444 | 15.78 | 6.49 |
| anthropic/claude-sonnet-5-5 | ~2,444 | 7.89 | 3.24 |
| openai/gpt-6.1-sol | ~2,037 | 7.07 | 3.20 |
| openai/gpt-6.1-terra | ~2,037 | 3.54 | 1.60 |
| anthropic/claude-haiku-5-5 | ~2,444 | 0.39 | 0.17 |
| openai/gpt-6-luna | ~2,037 | 0.35 | 0.17 |

Chart: `cost.png` (rewritten on every run, shows the last prompt)

## Notes

- Prices live in `prices.toml`. Copy them from each provider's pricing page and record the date.
- Tokenizers differ per provider, so token counts for the same prompt differ too.
- `~` means the token count is an estimate:
  - Claude models: exact counts come from Anthropic's `count_tokens` endpoint, which needs
    credentials. Set `ANTHROPIC_API_KEY` and rerun to get exact numbers. Without it the script
    falls back to tiktoken x 1.2, a rough guess.
  - OpenAI models: tiktoken 0.14 has no mapping for the gpt-6 models, so `o200k_base` is used.
- The cached column assumes the whole prompt is a cache hit. Providers only cache prompts above
  a minimum size (roughly 1K tokens, varies by model), and Anthropic charges extra to write the
  cache the first time. Neither is modelled.

## What I learned

-
