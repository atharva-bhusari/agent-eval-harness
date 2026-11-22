# A/B sweep: gpt-4o-mini vs Claude Haiku 5.5 × prompt v1/v2

4 config(s), 248 runs, golden set of 62 tasks. Accuracy = execution accuracy (the agent's result set equals the expected rows). The interval is a 95% Wilson interval: at this sample size, configs whose intervals overlap heavily are not reliably different. Cost and latency are nominal (a cached re-run reports the original run's numbers).

## Summary

- **Winner on accuracy: `haiku_v2`** (anthropic/claude-haiku-5-5, prompt v2) at **100%** (62/62).
- It ties `haiku_v1` on accuracy and ranks first on lower cost, then latency.
- It is significantly ahead of the best of the rest, `gpt4o_mini_v2` (82.3%; paired McNemar p = 0.001).
- **Trade-off:** `gpt4o_mini_v2` is 1.7× cheaper per run ($0.0006 vs $0.0009) for 18 points less accuracy.
- Cost per correct answer for the winner: $0.0009.
- Prompt effect on `openai/gpt-4o-mini`: `gpt4o_mini_v1` → `gpt4o_mini_v2` is +2 points on 62 shared tasks (4 fixed, 3 broken; p = 1.000, not significant).
- Prompt effect on `anthropic/claude-haiku-5-5`: `haiku_v1` → `haiku_v2` is +0 points on 62 shared tasks (0 fixed, 0 broken; p = 1.000, not significant).

## Leaderboard

| config | model | prompt | tasks | accuracy | 95% CI | mean latency | cost/run | cost/correct | retries/run | failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `haiku_v1` | `anthropic/claude-haiku-5-5` | v1 | 62/62 | **100%** (62/62) | 94.2%–100% | 4.2 s | $0.0010 | $0.0010 | 0.00 | — |
| `haiku_v2` | `anthropic/claude-haiku-5-5` | v2 | 62/62 | **100%** (62/62) | 94.2%–100% | 2.6 s | $0.0009 | $0.0009 | 0.00 | — |
| `gpt4o_mini_v2` | `openai/gpt-4o-mini` | v2 | 62/62 | **82.3%** (51/62) | 71.0%–89.8% | 6.4 s | $0.0006 | $0.0007 | 0.26 | wrong_result 10, timeout 1 |
| `gpt4o_mini_v1` | `openai/gpt-4o-mini` | v1 | 62/62 | **80.6%** (50/62) | 69.1%–88.6% | 7.6 s | $0.0006 | $0.0008 | 0.21 | wrong_result 11, sql_syntax_error 1 |

## Head-to-head (paired)

Each pair is compared only on the tasks both configs ran. *A only* / *B only* count tasks exactly one of them got right; the exact McNemar p-value tests whether that split could be chance (significant below 0.05).

| differs in | A | B | shared tasks | accuracy A → B | Δ | A only | B only | p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt | `gpt4o_mini_v1` | `gpt4o_mini_v2` | 62 | 80.6% → 82.3% | +2 pts | 3 | 4 | 1.000 |
| prompt | `haiku_v1` | `haiku_v2` | 62 | 100% → 100% | +0 pts | 0 | 0 | 1.000 |
| model | `gpt4o_mini_v1` | `haiku_v1` | 62 | 80.6% → 100% | +19 pts | 0 | 12 | 0.000 ✔ |
| model | `gpt4o_mini_v2` | `haiku_v2` | 62 | 82.3% → 100% | +18 pts | 0 | 11 | 0.001 ✔ |
| model + prompt | `gpt4o_mini_v1` | `haiku_v2` | 62 | 80.6% → 100% | +19 pts | 0 | 12 | 0.000 ✔ |
| model + prompt | `gpt4o_mini_v2` | `haiku_v1` | 62 | 82.3% → 100% | +18 pts | 0 | 11 | 0.001 ✔ |

<details>
<summary>Which tasks flipped</summary>

- `gpt4o_mini_v1` → `gpt4o_mini_v2`: fixed h03, m17, x04, x10; broken h15, x03, x07
- `gpt4o_mini_v1` → `haiku_v1`: fixed h01, h03, h08, h12, m17, x01, x04, x05, x09, x10, x11, x12; broken —
- `gpt4o_mini_v2` → `haiku_v2`: fixed h01, h08, h12, h15, x01, x03, x05, x07, x09, x11, x12; broken —
- `gpt4o_mini_v1` → `haiku_v2`: fixed h01, h03, h08, h12, m17, x01, x04, x05, x09, x10, x11, x12; broken —
- `gpt4o_mini_v2` → `haiku_v1`: fixed h01, h08, h12, h15, x01, x03, x05, x07, x09, x11, x12; broken —

</details>

## Accuracy by difficulty

| difficulty | `haiku_v1` | `haiku_v2` | `gpt4o_mini_v2` | `gpt4o_mini_v1` |
| --- | ---: | ---: | ---: | ---: |
| easy | 100% (15/15) | 100% (15/15) | 100% (15/15) | 100% (15/15) |
| medium | 100% (20/20) | 100% (20/20) | 100% (20/20) | 95.0% (19/20) |
| hard | 100% (15/15) | 100% (15/15) | 73.3% (11/15) | 73.3% (11/15) |
| expert | 100% (12/12) | 100% (12/12) | 41.7% (5/12) | 41.7% (5/12) |

## Failure taxonomy

| category | `haiku_v1` | `haiku_v2` | `gpt4o_mini_v2` | `gpt4o_mini_v1` |
| --- | ---: | ---: | ---: | ---: |
| `sql_syntax_error` | 0 | 0 | 0 | 1 |
| `wrong_result` | 0 | 0 | 10 | 11 |
| `timeout` | 0 | 0 | 1 | 0 |

## Failed runs

<details>
<summary>23 failed runs with the classifier's reason</summary>

| config | task | difficulty | category | why |
| --- | --- | --- | --- | --- |
| `gpt4o_mini_v1` | h01 | hard | `sql_syntax_error` | dry_run: Binder Error: column "category_id" must appear in the GROUP BY clause or be used in an aggregate function LINE 11: WHERE subquery.… |
| `gpt4o_mini_v1` | h03 | hard | `wrong_result` | same shape, different values; e.g. expected row (4.57,) not found |
| `gpt4o_mini_v1` | h08 | hard | `wrong_result` | same shape, different values; e.g. expected row (1996.0, 152.0, 208083.971) not found |
| `gpt4o_mini_v1` | h12 | hard | `wrong_result` | same shape, different values; e.g. expected row (1.0, 33.0) not found |
| `gpt4o_mini_v1` | m17 | medium | `wrong_result` | same shape, different values; e.g. expected row (8.492,) not found |
| `gpt4o_mini_v1` | x01 | expert | `wrong_result` | row count differs: expected 1, got 36 |
| `gpt4o_mini_v1` | x04 | expert | `wrong_result` | row count differs: expected 2, got 8 |
| `gpt4o_mini_v1` | x05 | expert | `wrong_result` | row count differs: expected 2, got 234 |
| `gpt4o_mini_v1` | x09 | expert | `wrong_result` | row count differs: expected 1, got 25 |
| `gpt4o_mini_v1` | x10 | expert | `wrong_result` | same shape, different values; e.g. expected row ('Save-a-lot Markets', 19.0) not found |
| `gpt4o_mini_v1` | x11 | expert | `wrong_result` | same shape, different values; e.g. expected row (13.0,) not found |
| `gpt4o_mini_v1` | x12 | expert | `wrong_result` | row count differs: expected 7, got 6 |
| `gpt4o_mini_v2` | h01 | hard | `wrong_result` | row count differs: expected 8, got 77 |
| `gpt4o_mini_v2` | h08 | hard | `wrong_result` | same shape, different values; e.g. expected row (1996.0, 152.0, 208083.971) not found |
| `gpt4o_mini_v2` | h12 | hard | `wrong_result` | same shape, different values; e.g. expected row (1.0, 33.0) not found |
| `gpt4o_mini_v2` | h15 | hard | `wrong_result` | same shape, different values; e.g. expected row (44.2,) not found |
| `gpt4o_mini_v2` | x01 | expert | `wrong_result` | row count differs: expected 1, got 36 |
| `gpt4o_mini_v2` | x03 | expert | `timeout` | exceeded timeout_s |
| `gpt4o_mini_v2` | x05 | expert | `wrong_result` | row count differs: expected 2, got 636 |
| `gpt4o_mini_v2` | x07 | expert | `wrong_result` | same shape, different values; e.g. expected row (1025.85,) not found |
| `gpt4o_mini_v2` | x09 | expert | `wrong_result` | row count differs: expected 1, got 25 |
| `gpt4o_mini_v2` | x11 | expert | `wrong_result` | same shape, different values; e.g. expected row (13.0,) not found |
| `gpt4o_mini_v2` | x12 | expert | `wrong_result` | row count differs: expected 7, got 86 |

</details>
