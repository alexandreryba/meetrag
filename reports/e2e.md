# End-to-end evaluation

46 questions, run on 2026-10-23T14:35:08+00:00. Models: provider `claude-code`, synthesis `claude-opus-5-5`, agents `claude-opus-5-5`, judge `claude-opus-5-5`.

> Run through the Claude Code CLI on a Claude subscription: latency includes the CLI start-up of every call, and costs are the API-equivalent prices reported by the CLI (the subscription is what was billed).

| Metric | Value |
|---|---|
| Answer accuracy (correct = 1, partial = 0.5) | 96.3% |
| Strict accuracy (fully correct) | 92.7% |
| Faithfulness (claims supported by evidence) | 99.2% |
| Routing recall / precision | 100.0% / 97.6% |
| Abstention on out-of-scope questions | 100.0% |
| False abstention on answerable questions | 0.0% |
| Latency p50 / p95 | 21.1 s / 27.0 s |
| Cost per question | $0.0543 |
| LLM calls per question | 4.0 |
| Errors | 0 |

## By category

| Category | N | Accuracy | Faithfulness | Routing recall | Abstention | p50 (s) |
|---|---|---|---|---|---|---|
| analytics | 8 | 93.8% | 96.9% | 100.0% | - | 24.2 |
| multi_source | 6 | 83.3% | 100.0% | 100.0% | - | 23.4 |
| out_of_scope | 5 | - | - | - | 100.0% | 8.4 |
| policies | 10 | 100.0% | 99.1% | 100.0% | - | 20.0 |
| products | 9 | 100.0% | 100.0% | 100.0% | - | 20.4 |
| support | 8 | 100.0% | 100.0% | 100.0% | - | 21.4 |

## Questions not fully correct

- **ana-07** (0.5): The answer gets the direction right (Drive is higher than Essentielle), but its figures don't match the reference: Drive is 482.97 kWh against 505.20, and Essentielle is 387.06 against 389.50. The Drive gap is about 4.4%, too large to count as rounding, likely because it averaged per-customer averages.
- **mul-01** (0.5): The answer correctly names Essentielle at about 32.8% churn, includes the comparison with Fixe 3 ans and Essentielle, and says never to match a competitor's price. It leaves out the tenure-based loyalty gestures: 1 month free from 2 years, plus a EUR 30 credit from 5 years.
- **mul-05** (0.5): The wallbox offer matches the reference: 7.4 kW for €1,190, a €300 credit, and a net cost of €890. But the average bill is wrong: the answer says €100.13 and the reference says €104.07.
