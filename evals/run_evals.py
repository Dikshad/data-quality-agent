"""End-to-end scoring evals: the "damage ladder".

Take a clean dataset, damage it in known ways at known strengths, and check that
  1. the score drops, and drops more as the damage gets worse, and
  2. the scorecard line that loses the most points is the RIGHT one (the report names the right cause).

Runs offline (synthetic data, no LLM judge):   python -m evals.run_evals
Writes evals/RESULTS.md.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dqa import synthetic as syn            # noqa: E402
from dqa.tools import ToolBox               # noqa: E402

N = 1000
CASES = [  # (name, damage function, expected line to lose the most points)
    ("clean", lambda d: d, None),
    *[(f"duplicates {int(r*100)}%", (lambda r: lambda d: syn.add_near_duplicates(d, r))(r), "duplication") for r in (0.05, 0.10, 0.20)],
    *[(f"junk {int(r*100)}%", (lambda r: lambda d: syn.replace_fraction(d, r, syn.junk_doc))(r), "junk") for r in (0.10, 0.30)],
    *[(f"choppy sentences {int(r*100)}%", (lambda r: lambda d: syn.replace_fraction(d, r, syn.chopped_doc))(r), "readability") for r in (0.25, 0.50)],
    *[(f"wrong language {int(r*100)}%", (lambda r: lambda d: syn.replace_fraction(d, r, syn.foreign_doc("de")))(r), "language_mix") for r in (0.20, 0.50)],
    ("prompt injection 20%", lambda d: syn.replace_fraction(d, 0.2, syn.injected_doc), None),
]


def evaluate(workdir="evals/.work"):
    Path(workdir).mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    clean = syn.clean_docs(N, seed=10)
    ref_path = Path("reference.jsonl")
    ref_path.write_text("\n".join(json.dumps({"text": t}) for t in syn.clean_docs(N, seed=99)))  # separate from candidates: not circular

    rows, baseline = [], None
    for name, damage, expected in CASES:
        p = Path(f"{name.replace(' ', '_').replace('%', 'pct')}.jsonl")
        p.write_text("\n".join(json.dumps({"text": t}) for t in damage(clean)))
        tb = ToolBox()
        s = json.loads(tb.call("load_sample", {"dataset": str(p), "n": 5000}))["sample_id"]
        r = json.loads(tb.call("load_sample", {"dataset": str(ref_path), "is_reference": True, "n": 5000}))["sample_id"]
        for t in ("check_language_mix", "check_duplication", "check_readability", "check_junk"):
            tb.call(t, {"sample_id": s})
        tb.call("run_quality_benchmark", {"sample_id": s, "reference_sample_id": r, "use_classifier": False})
        sc = json.loads(tb.call("score_dataset", {"sample_id": s, "data_type": "prose", "target_languages": ["en"]}))
        lost = {k: v["max"] - v["points"] for k, v in sc["lines"].items() if v["points"] is not None and k != "benchmark"}
        worst = max(lost, key=lost.get) if max(lost.values()) > 0.5 else None
        baseline = baseline or sc["total"]
        ok = (worst == expected) if expected else (sc["total"] >= baseline - 3)
        rows.append({"case": name, "score": sc["total"], "biggest_loss": worst or "-", "expected": expected or "no real drop", "pass": ok})

    # monotonic: more damage -> lower score, within each family
    families = {}
    for row in rows:
        families.setdefault(row["case"].rsplit(" ", 1)[0], []).append(row["score"])
    monotonic = all(all(a > b for a, b in zip(v, v[1:])) for v in families.values() if len(v) > 1)
    return rows, monotonic


def main():
    root = Path.cwd()
    rows, monotonic = evaluate()
    os.chdir(root)
    lines = ["# Eval results (offline, synthetic damage ladder, LLM judge excluded)\n",
             "| Case | Score /100 | Line that lost most | Expected | Pass |", "|---|---|---|---|---|"]
    lines += [f"| {r['case']} | {r['score']} | {r['biggest_loss']} | {r['expected']} | {'✅' if r['pass'] else '❌'} |" for r in rows]
    passed = sum(r["pass"] for r in rows)
    lines += ["", f"**{passed}/{len(rows)} cases passed. Scores fall as damage grows: {'✅' if monotonic else '❌'}**",
              "", "Prompt-injection docs don't move the code-computed score (the code checks can't be talked to). "
              "The LLM-judge red-team test needs an API key: see `tests/` and `evals/agent_scenarios.py`."]
    Path("evals/RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passed == len(rows) and monotonic else 1


if __name__ == "__main__":
    sys.exit(main())
