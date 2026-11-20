# Data Quality Agent

An open-source AI agent that helps ML researchers **find, check and rank training datasets** on Hugging Face. It scores each dataset on a weighted quality scorecard, benchmarked against trusted reference data.

> "I need clean French text to pretrain a small assistant."
> → asks what it needs to know → searches Hugging Face → samples 3 candidates + a fair reference → runs the checks → scores and ranks them → writes a report explaining the pick.

## Why an agent, not a script

Existing tools run a fixed checklist. This agent decides **what to investigate**:

- It works out what kind of data it's looking at (prose, code, chat) and picks a **fair reference and thresholds for that type**. It won't grade Python code with a classifier built for English web pages.
- It digs into what it finds. A huge duplicate cluster suggests template pages, and a language mix only matters relative to the researcher's need.
- It asks the researcher when it's stuck, for example when 5 searches found nothing.

## How it works

| Piece | What it is | Where |
|---|---|---|
| **Brain** | Claude, which decides the next step | `dqa/agent.py` |
| **Tools (hands)** | Fixed, tested code. Deterministic, so scores are comparable across runs | `dqa/tools.py`, `dqa/checks.py` |
| **Skill (recipe)** | The expert playbook: what to ask, which reference to pick, when to dig deeper | `skills/dataset-quality/SKILL.md` |
| **Loop** | Model → tool → result → model, until done | `dqa/agent.py` |

**Tools:** `search_datasets`, `load_sample`, `check_language_mix`, `check_duplication`, `check_readability`, `check_junk`, `run_quality_benchmark`, `llm_judge`, `score_dataset`, `write_report`, plus `read_skill`. The agent loads the skill only when it needs it, which keeps its context lean.

**Design principle: code measures, the model judges.** Counting duplicates across thousands of documents, sentence lengths and junk rates is done in plain code, which is cheap, fast and repeatable. The LLM grades only what needs a reader ("is this actually informative?"), on a small sample.

## Scorecard (out of 100)

| Line | Weight | How it's measured |
|---|---|---|
| Benchmark vs reference | 20 | FineWeb-Edu classifier (English prose only), or profile vs a same-type reference |
| Duplication | 20 | Exact + near-duplicates (MinHash, ~80% similarity, 5-word shingles). Points slide gradually to ~0 at 20% |
| Readability | 20 | Words per sentence, share of choppy docs. Thresholds depend on data type |
| Language & content mix | 20 | Share matching the researcher's target languages / content type |
| Junk | 10 | Binary/garbled, non-text, repeated lines, HTML remnants, too short, broken pages |
| LLM judge | 10 | 1–5 rubric: coherent, informative, not boilerplate, no personal info, fits purpose |

The total is computed **in code, never by the model**. Lines that couldn't be measured are marked, and the total is rescaled over the lines that were.

## Guardrails (enforced in code, not just in the prompt)

- **Caps:** 5 searches, 5 candidate datasets, 5,000 rows per sample, 300 docs for the judge, 40 steps per turn, a token budget per session.
- **Retries:** network errors are retried 3 times, then reported. If one check fails, the others still run.
- **Read-only:** no tool can modify, upload or delete data.
- **Context management:** dataset text never enters the agent's context. Samples live on disk, and every tool returns a short summary.
- **Prompt injection:** the LLM judge has **no tools** and can only return scores. Documents are wrapped as untrusted data and can't break out of their tags. Every score is validated as an integer from 1 to 5. If the judge sharply disagrees with the code checks, the report flags it for a human. A fooled judge can move at most its 10 points.

## Evals

Three layers, as in the design spec:

1. **Tool tests** (`tests/`): plant a known problem and check that it's measured. For example, planted duplicates of 5/10/20/30% are measured within ±2%.
2. **Damage ladder** (`evals/run_evals.py`): damage a clean dataset in known ways and check that the score falls and the **right line** takes the blame.
3. **Judgment evals** (`evals/agent_scenarios.py`, needs an API key): did the agent pick the right reference for code, English web and French requests? Did it avoid circular scoring, respect the caps and flag injected instructions?

Latest offline results ([full table](evals/RESULTS.md)): **11/11 damage-ladder cases pass**, and scores fall monotonically as damage grows. Writing these evals surfaced a real design bug: junk pages were also counted as duplicates, so they were penalised twice and the wrong cause was blamed. The fix was to filter junk before deduplicating, as real cleaning pipelines do.

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python -m pytest -q                 # tool + guardrail tests (offline)
python -m evals.run_evals           # damage-ladder evals (offline)
python -m dqa.agent "I need high-quality English web text to pretrain a 1B model"
```

To enable the FineWeb-Edu classifier: `pip install transformers torch`.

## Limitations and next steps

- The language and content-type detectors are heuristics. A next step is a fastText language ID model and a proper code/chat classifier.
- The LLM judge hasn't yet been validated against human grades. The plan is to hand-grade ~50 docs and measure agreement.
- Samples come from a streamed shuffle buffer, not a uniform random sample of the full dataset.
- The agent has been tested offline with a scripted model; live runs against Hugging Face are next.
