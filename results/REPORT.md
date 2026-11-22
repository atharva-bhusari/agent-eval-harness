# Eval report

2 config(s), 53 runs, golden set of 50 tasks. Accuracy = execution accuracy (the agent's result set equals the expected rows). The interval is a 95% Wilson interval: with ~50 tasks, configs whose intervals overlap heavily are not reliably different.

## Leaderboard

| config | model | prompt | tasks | accuracy | 95% CI | mean latency | cost/run | retries/run | failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ollama_llama_v2` | `ollama/llama3.1:8b` | v2 | 50/50 | **80.0%** (40/50) | 67.0%–88.8% | 27.5 s | $0 | 0.26 | wrong_result 9, empty_result 1 |
| `ollama_llama` | `ollama/llama3.1:8b` | v1 | 3/50 ⚠️ | **33.3%** (1/3) | 6.1%–79.2% | 163.0 s | $0 | 6.67 | timeout 1, schema_hallucination 1 |

⚠️ = partial run: accuracy covers only the tasks that were run, so it is not comparable with full runs.

## Accuracy by difficulty

| difficulty | `ollama_llama_v2` | `ollama_llama` |
| --- | ---: | ---: |
| easy | 100% (15/15) | 100% (1/1) |
| medium | 85.0% (17/20) | 0% (0/1) |
| hard | 53.3% (8/15) | 0% (0/1) |

## Failure taxonomy

| category | `ollama_llama_v2` | `ollama_llama` |
| --- | ---: | ---: |
| `schema_hallucination` | 0 | 1 |
| `wrong_result` | 9 | 0 |
| `empty_result` | 1 | 0 |
| `timeout` | 0 | 1 |

## Failed runs

<details>
<summary>12 failed runs with the classifier's reason</summary>

| config | task | difficulty | category | why |
| --- | --- | --- | --- | --- |
| `ollama_llama` | h01 | hard | `timeout` | Timeout: litellm.Timeout: OllamaException: litellm.Timeout: Read timed out. Timeout passed=41.326805100000456, time taken=41.329 seconds |
| `ollama_llama` | m03 | medium | `schema_hallucination` | run_sql: Binder Error: Table "o" does not have a column named "shipper_id" Candidate bindings: : "shipped_date" LINE 1: ...Count FROM shipp… |
| `ollama_llama_v2` | h01 | hard | `wrong_result` | row count differs: expected 8, got 1 |
| `ollama_llama_v2` | h03 | hard | `wrong_result` | same shape, different values; e.g. expected row (4.57,) not found |
| `ollama_llama_v2` | h06 | hard | `wrong_result` | same shape, different values; e.g. expected row (12.0, 71398.4285) not found |
| `ollama_llama_v2` | h07 | hard | `wrong_result` | same shape, different values; e.g. expected row (30.0,) not found |
| `ollama_llama_v2` | h08 | hard | `wrong_result` | same shape, different values; e.g. expected row (1996.0, 152.0, 208083.971) not found |
| `ollama_llama_v2` | h12 | hard | `wrong_result` | same shape, different values; e.g. expected row (1.0, 33.0) not found |
| `ollama_llama_v2` | h15 | hard | `wrong_result` | row count differs: expected 1, got 10 |
| `ollama_llama_v2` | m02 | medium | `empty_result` | row count differs: expected 6, got 0 |
| `ollama_llama_v2` | m14 | medium | `wrong_result` | row count differs: expected 4, got 29 |
| `ollama_llama_v2` | m17 | medium | `wrong_result` | same shape, different values; e.g. expected row (8.492,) not found |

</details>
