"""write_report: render the scorecards (numbers straight from code) plus the agent's recommendation."""
from datetime import date

LABELS = {"benchmark": "Benchmark vs reference", "duplication": "Duplication", "readability": "Readability",
          "language_mix": "Language & content mix", "junk": "Junk", "llm_judge": "LLM judge"}


def _detail(key, r):
    if key == "benchmark" and "benchmark" in r:
        b = r["benchmark"]
        if b["method"] == "fineweb_edu_classifier":
            return f"classifier {b['sample_score']} vs reference {b['reference_score']}"
        return f"profile vs reference `{b['reference_sample_id']}`"
    if key == "duplication" and "duplication" in r:
        d = r["duplication"]
        return f"{d['duplicate_rate']:.1%} removable ({d['exact_duplicate_rate']:.1%} exact, {d['near_duplicate_rate']:.1%} near)"
    if key == "readability" and r.get("readability", {}).get("measurable"):
        x = r["readability"]
        return f"{x['avg_words_per_sentence']} words/sentence; {x['share_docs_under_5_words_per_sentence']:.0%} docs very choppy"
    if key == "language_mix" and "language_mix" in r:
        m = r["language_mix"]
        langs = ", ".join(f"{k} {v:.0%}" for k, v in list(m["languages"].items())[:3])
        types = ", ".join(f"{k} {v:.0%}" for k, v in m["content_types"].items())
        return f"{langs} | {types}"
    if key == "junk" and "junk" in r:
        j = r["junk"]
        top = ", ".join(f"{k} {v:.0%}" for k, v in list(j["by_reason"].items())[:3])
        return f"{j['junk_rate']:.1%} junk" + (f" ({top})" if top else "")
    if key == "llm_judge" and "llm_judge" in r:
        j = r["llm_judge"]
        return f"mean {j['overall_mean']}/5 over {j['docs_graded']} docs"
    return ""


def render(ranked, results, meta, need, recommendation) -> str:
    out = [f"# Dataset quality report\n\n_{date.today()}_\n\n**Researcher need:** {need}\n",
           "## Ranking\n", "| Rank | Dataset | Score /100 |", "|---|---|---|"]
    for i, s in enumerate(ranked, 1):
        out.append(f"| {i} | {meta.get(s, {}).get('dataset', s)} | {results[s]['score']['total']} |")
    out += ["\n## Recommendation\n", recommendation, ""]
    for s in ranked:
        sc = results[s]["score"]
        out += [f"## {meta.get(s, {}).get('dataset', s)}  ({sc['total']}/100, type: {sc['data_type']})\n",
                "| Line | Points | Detail |", "|---|---|---|"]
        for k, line in sc["lines"].items():
            pts = f"{line['points']}/{line['max']}" if line["points"] is not None else f"— /{line['max']} ({line['status']})"
            out.append(f"| {LABELS[k]} | {pts} | {_detail(k, results[s])} |")
        for f in sc["flags"]:
            out.append(f"\n> ⚠️ {f}")
        out.append("")
    return "\n".join(out)
