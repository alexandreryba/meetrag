# End-to-end evaluation

67 questions, run on 2026-10-28T17:18:24+00:00. Models: provider `claude-code`, synthesis `claude-opus-5-5`, agents `claude-opus-5-5`, judge `claude-opus-5-5`.

> Run through the Claude Code CLI on a Claude subscription: latency includes the CLI start-up of every call, and costs are the API-equivalent prices reported by the CLI (the subscription is what was billed).

| Metric | Value |
|---|---|
| Answer accuracy (correct = 1, partial = 0.5) | 96.7% |
| Strict accuracy (fully correct) | 93.3% |
| Faithfulness (claims supported by evidence) | 97.8% |
| Routing recall / precision | 100.0% / 90.8% |
| Abstention on out-of-scope questions | 100.0% |
| False abstention on answerable questions | 0.0% |
| Latency p50 / p95 | 20.9 s / 36.0 s |
| Cost per question | $0.0818 |
| LLM calls per question | 4.3 |
| Errors | 0 |

## By category

| Category | N | Accuracy | Faithfulness | Routing recall | Abstention | p50 (s) |
|---|---|---|---|---|---|---|
| analytics | 8 | 87.5% | 93.3% | 100.0% | - | 20.6 |
| meeting_records | 4 | 100.0% | 96.4% | 100.0% | - | 28.8 |
| meetings | 12 | 100.0% | 100.0% | 100.0% | - | 23.9 |
| multi_source | 9 | 94.4% | 99.3% | 100.0% | - | 28.2 |
| out_of_scope | 7 | - | - | - | 100.0% | 15.5 |
| policies | 10 | 95.0% | 97.7% | 100.0% | - | 16.3 |
| products | 9 | 100.0% | 98.1% | 100.0% | - | 20.0 |
| support | 8 | 100.0% | 97.6% | 100.0% | - | 17.6 |

## Questions not fully correct

- **pol-01** (0.5): The answer gets the main figure right (up to 3 days per week) and mentions the team anchor day. It leaves out a key condition from the reference: remote work is only allowed once the probation period is completed. It only says "eligible employees" without explaining what makes someone eligible.
- **ana-07** (0.5): The answer gets the direction right (Drive is higher than Essentielle), but its figures don't match the reference. It gives 482.97 kWh against the reference's 505.20 for Drive, and 387.06 kWh against 389.50 for Essentielle. These gaps are too large to be rounding and probably come from averaging per-customer averages instead of using the reference's method.
- **ana-08** (0.5): The answer gives 6.84 days, but the reference is 6.80. The gap is small, but it is not a rounding difference, so it likely comes from a different calculation or filter. The approach and framing are otherwise consistent with the reference.
- **mul-05** (0.5): The wallbox offer matches the reference (7.4 kW, €1,190, €300 credit, €890 net), but the answer does not give the average monthly bill of 104.07 €. It says no single figure is available.
