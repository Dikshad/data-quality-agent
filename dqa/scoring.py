"""The scorecard. Code computes every number; the model never does the maths.

Weights (spec section 4): benchmark 20, duplication 20, readability 20,
language & content mix 20, junk 10, LLM judge 10.
"""
WEIGHTS = {"benchmark": 20, "duplication": 20, "readability": 20, "language_mix": 20, "junk": 10, "llm_judge": 10}

# Readability thresholds depend on data type: short sentences are normal in chat, and
# words-per-sentence means nothing for code.
READABILITY = {
    "prose": {"floor": 2.5, "ideal_min": 8, "ideal_max": 40},
    "chat": {"floor": 1.0, "ideal_min": 4, "ideal_max": 40},
    "code": None,
}
DUP_ZERO_AT = 0.20        # duplication points slide to ~zero by 20% duplicates
JUNK_ZERO_AT = 0.30


def clip(x):
    return max(0.0, min(1.0, x))


def duplication_fraction(rate):
    # gradual, not a cliff: 5% -> 0.88, 10% -> 0.65, 15% -> 0.35, 20%+ -> 0
    return clip(1 - (rate / DUP_ZERO_AT) ** 1.5)


def readability_fraction(r, data_type):
    t = READABILITY.get(data_type)
    if t is None or not r.get("measurable"):
        return None
    m = r["avg_words_per_sentence"]
    if m < t["ideal_min"]:
        by_length = clip((m - t["floor"]) / (t["ideal_min"] - t["floor"]))
    else:
        by_length = clip(1 - max(0, m - t["ideal_max"]) / 60)    # very long = run-on / no punctuation
    short_share = r["share_docs_under_5_words_per_sentence"] if data_type == "prose" else 0
    return 0.5 * by_length + 0.5 * (1 - short_share)


def language_fraction(mix, target_languages=None, target_content_type=None):
    langs = mix["languages"]
    lang = sum(langs.get(l, 0) for l in target_languages) if target_languages else max(langs.values())
    if not target_content_type:
        return lang
    return 0.5 * lang + 0.5 * mix["content_types"].get(target_content_type, 0)


def score(results: dict, data_type: str, target_languages=None, target_content_type=None) -> dict:
    from .benchmark import benchmark_fraction
    fractions = {}
    fractions["benchmark"] = benchmark_fraction(results["benchmark"]) if "benchmark" in results else None
    fractions["duplication"] = duplication_fraction(results["duplication"]["duplicate_rate"]) if "duplication" in results else None
    fractions["readability"] = readability_fraction(results["readability"], data_type) if "readability" in results else None
    fractions["language_mix"] = (language_fraction(results["language_mix"], target_languages, target_content_type)
                                 if "language_mix" in results else None)
    fractions["junk"] = clip(1 - results["junk"]["junk_rate"] / JUNK_ZERO_AT) if "junk" in results else None
    j = results.get("llm_judge")
    fractions["llm_judge"] = (j["overall_mean"] - 1) / 4 if j else None

    lines, earned, possible = {}, 0.0, 0
    for k, w in WEIGHTS.items():
        f = fractions[k]
        if f is None:
            lines[k] = {"points": None, "max": w, "status": "couldn't measure / not applicable"}
        else:
            lines[k] = {"points": round(f * w, 1), "max": w}
            earned += f * w
            possible += w

    flags = []
    if possible < 100:
        flags.append(f"Only {possible}/100 points were measurable; total is rescaled to 100.")
    # Guardrail: if the LLM judge loves data that the code checks say is poor (or vice versa),
    # something's off (e.g. prompt injection). Flag for a human instead of trusting it.
    if fractions["llm_judge"] is not None:
        code_view = [f for f in (fractions["benchmark"], fractions["junk"]) if f is not None]
        if code_view and abs(fractions["llm_judge"] - min(code_view)) > 0.5:
            flags.append("LLM judge disagrees sharply with the code checks: needs human review.")
    if results.get("llm_judge", {}).get("grades_rejected", 0) > 0.2 * results.get("llm_judge", {}).get("docs_graded", 1):
        flags.append("Many LLM judge grades were malformed and discarded.")

    return {"data_type": data_type, "lines": lines,
            "total": round(earned / possible * 100, 1) if possible else None, "flags": flags}
