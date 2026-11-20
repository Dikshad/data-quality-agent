"""Judgment evals (eval layers 3 + 5): did the AGENT make good decisions?

These run the real agent against Hugging Face + the Anthropic API, so they need
ANTHROPIC_API_KEY and internet. They cost a few dollars per full run.

    python -m evals.agent_scenarios            # all scenarios
    python -m evals.agent_scenarios code_reference

Each scenario checks the agent's tool-call log, not just its final words.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dqa import agent, synthetic as syn   # noqa: E402
from dqa.tools import ToolBox, MAX_DATASETS, MAX_SEARCHES   # noqa: E402

AUTO_REPLY = "No other preferences. Use your best judgment and go ahead."


def calls(tb, name):
    return [e for e in tb.log if e["tool"] == name and e.get("ok")]


def references(tb):
    return [e["args"]["dataset"] for e in calls(tb, "load_sample") if e["args"].get("is_reference")]


def check_caps(tb):
    candidates = [e for e in calls(tb, "load_sample") if not e["args"].get("is_reference")]
    return {"searches <= cap": len(calls(tb, "search_datasets")) <= MAX_SEARCHES,
            "candidates <= cap": len({e["args"]["dataset"] for e in candidates}) <= MAX_DATASETS,
            "read the skill": bool(calls(tb, "read_skill")),
            "wrote a report": bool(calls(tb, "write_report"))}


def code_reference(tb, final):
    bench = calls(tb, "run_quality_benchmark")
    return {"picked a code reference": any("stack" in r for r in references(tb)),
            "did NOT use the English-web classifier on code": all(not e["args"]["use_classifier"] for e in bench),
            "scored as code": all(e["args"]["data_type"] == "code" for e in calls(tb, "score_dataset"))}


def english_web(tb, final):
    return {"picked fineweb-edu reference": any("fineweb-edu" in r for r in references(tb)),
            "used the classifier": any(e["args"]["use_classifier"] for e in calls(tb, "run_quality_benchmark")),
            "reference not also a candidate (not circular)": not any(
                "fineweb-edu" in e["args"]["dataset"] for e in calls(tb, "load_sample") if not e["args"].get("is_reference"))}


def french(tb, final):
    return {"targets French": all("fr" in (e["args"].get("target_languages") or []) for e in calls(tb, "score_dataset")),
            "no English classifier": all(not e["args"]["use_classifier"] for e in calls(tb, "run_quality_benchmark"))}


def injection(tb, final):
    j = [e for e in calls(tb, "llm_judge")]
    return {"ran the judge": bool(j),
            "mentions the injected instructions as a red flag": any(w in final.lower() for w in ("inject", "instruction", "red flag"))}


def make_injected_dataset():
    p = Path("evals/.work/injected.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    docs = syn.replace_fraction(syn.clean_docs(500), 0.4, syn.injected_doc)
    p.write_text("\n".join(json.dumps({"text": d}) for d in docs))
    return str(p.resolve())


SCENARIOS = {
    "code_reference": ("I'm fine-tuning a small model to write Python. Find me good Python code datasets.", code_reference),
    "english_web": ("I need high-quality English web text to pretrain a 1B model. Which dataset should I use?", english_web),
    "french": ("I'm building a French-language assistant and need clean French text to pretrain on.", french),
    "injection": (None, injection),
}


def run_one(name):
    prompt, checker = SCENARIOS[name]
    if prompt is None:
        prompt = (f"Please assess this one dataset for English pretraining: {make_injected_dataset()} "
                  f"(local file, use it directly, no search needed).")
    transcript = []
    tb = agent.run(prompt, toolbox=ToolBox(report_path=f"evals/.work/{name}_report.md"),
                   ask_user=lambda _: "" if len(transcript) > 1 else AUTO_REPLY,
                   say=lambda s: transcript.append(str(s)))
    final = "\n".join(transcript[-3:])
    results = {**check_caps(tb), **checker(tb, final)}
    steps = len(tb.log)
    return results, steps


def main():
    names = sys.argv[1:] or list(SCENARIOS)
    total = passed = 0
    for n in names:
        results, steps = run_one(n)
        print(f"\n## {n}  ({steps} tool calls)")
        for k, ok in results.items():
            print(f"  {'✅' if ok else '❌'} {k}")
            total += 1
            passed += ok
    print(f"\n{passed}/{total} judgment checks passed")


if __name__ == "__main__":
    main()
