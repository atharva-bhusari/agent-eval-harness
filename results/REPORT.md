# A/B sweep: gpt-4o-mini vs Claude Haiku 5.5 × prompt v1/v2

2 config(s), 100 runs, golden set of 50 tasks. Accuracy = execution accuracy (the agent's result set equals the expected rows). The interval is a 95% Wilson interval: with ~50 tasks, configs whose intervals overlap heavily are not reliably different. Cost and latency are nominal (a cached re-run reports the original run's numbers).

## Summary

- **Winner on accuracy: `gpt4o_mini_v2`** (openai/gpt-4o-mini, prompt v2) at **92.0%** (46/50).
- Its lead over the runner-up `gpt4o_mini_v1` (90.0%) is **not statistically significant** (paired McNemar p = 1.00): on 50 tasks, treat them as tied on accuracy and choose on cost and latency.
- It is also the cheapest and the fastest: no cost/latency trade-off.
- Cost per correct answer for the winner: $0.0006.
- Prompt effect on `openai/gpt-4o-mini`: `gpt4o_mini_v1` → `gpt4o_mini_v2` is +2 points on 50 shared tasks (2 fixed, 1 broken; p = 1.000, not significant).

## Leaderboard

| config | model | prompt | tasks | accuracy | 95% CI | mean latency | cost/run | cost/correct | retries/run | failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gpt4o_mini_v2` | `openai/gpt-4o-mini` | v2 | 50/50 | **92.0%** (46/50) | 81.2%–96.8% | 4.1 s | $0.0005 | $0.0006 | 0.10 | wrong_result 4 |
| `gpt4o_mini_v1` | `openai/gpt-4o-mini` | v1 | 50/50 | **90.0%** (45/50) | 78.6%–95.7% | 5.7 s | $0.0006 | $0.0007 | 0.24 | wrong_result 4, sql_syntax_error 1 |

## Head-to-head (paired)

Each pair is compared only on the tasks both configs ran. *A only* / *B only* count tasks exactly one of them got right; the exact McNemar p-value tests whether that split could be chance (significant below 0.05).

| differs in | A | B | shared tasks | accuracy A → B | Δ | A only | B only | p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt | `gpt4o_mini_v1` | `gpt4o_mini_v2` | 50 | 90.0% → 92.0% | +2 pts | 1 | 2 | 1.000 |

<details>
<summary>Which tasks flipped</summary>

- `gpt4o_mini_v1` → `gpt4o_mini_v2`: fixed h03, m17; broken h15

</details>

## Accuracy by difficulty

| difficulty | `gpt4o_mini_v2` | `gpt4o_mini_v1` |
| --- | ---: | ---: |
| easy | 100% (15/15) | 100% (15/15) |
| medium | 100% (20/20) | 95.0% (19/20) |
| hard | 73.3% (11/15) | 73.3% (11/15) |

## Failure taxonomy

| category | `gpt4o_mini_v2` | `gpt4o_mini_v1` |
| --- | ---: | ---: |
| `sql_syntax_error` | 0 | 1 |
| `wrong_result` | 4 | 4 |

## Failed runs

<details>
<summary>9 failed runs with the classifier's reason</summary>

| config | task | difficulty | category | why |
| --- | --- | --- | --- | --- |
| `gpt4o_mini_v1` | h01 | hard | `sql_syntax_error` | dry_run: Binder Error: column "category_id" must appear in the GROUP BY clause or be used in an aggregate function LINE 11: WHERE subquery.… |
| `gpt4o_mini_v1` | h03 | hard | `wrong_result` | same shape, different values; e.g. expected row (4.57,) not found |
| `gpt4o_mini_v1` | h08 | hard | `wrong_result` | same shape, different values; e.g. expected row (1996.0, 152.0, 208083.971) not found |
| `gpt4o_mini_v1` | h12 | hard | `wrong_result` | same shape, different values; e.g. expected row (1.0, 33.0) not found |
| `gpt4o_mini_v1` | m17 | medium | `wrong_result` | same shape, different values; e.g. expected row (8.492,) not found |
| `gpt4o_mini_v2` | h01 | hard | `wrong_result` | row count differs: expected 8, got 77 |
| `gpt4o_mini_v2` | h08 | hard | `wrong_result` | same shape, different values; e.g. expected row (1996.0, 152.0, 208083.971) not found |
| `gpt4o_mini_v2` | h12 | hard | `wrong_result` | same shape, different values; e.g. expected row (1.0, 33.0) not found |
| `gpt4o_mini_v2` | h15 | hard | `wrong_result` | same shape, different values; e.g. expected row (44.2,) not found |

</details>
