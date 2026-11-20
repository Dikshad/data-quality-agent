# Eval results (offline, synthetic damage ladder, LLM judge excluded)

| Case | Score /100 | Line that lost most | Expected | Pass |
|---|---|---|---|---|
| clean | 100.0 | - | no real drop | ✅ |
| duplicates 5% | 95.3 | duplication | duplication | ✅ |
| duplicates 10% | 88.4 | duplication | duplication | ✅ |
| duplicates 20% | 70.3 | duplication | duplication | ✅ |
| junk 10% | 93.2 | junk | junk | ✅ |
| junk 30% | 79.3 | junk | junk | ✅ |
| choppy sentences 25% | 94.3 | readability | readability | ✅ |
| choppy sentences 50% | 85.1 | readability | readability | ✅ |
| wrong language 20% | 95.3 | language_mix | language_mix | ✅ |
| wrong language 50% | 88.3 | language_mix | language_mix | ✅ |
| prompt injection 20% | 99.9 | - | no real drop | ✅ |

**11/11 cases passed. Scores fall as damage grows: ✅**

Prompt-injection docs don't move the code-computed score (the code checks can't be talked to). The LLM-judge red-team test needs an API key: see `tests/` and `evals/agent_scenarios.py`.
