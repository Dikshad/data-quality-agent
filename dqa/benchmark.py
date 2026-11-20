"""run_quality_benchmark: compare a sample with a trusted reference ("gold standard") sample.

Apples to apples: the classifier only means something for the data it was trained on.
The FineWeb-Edu classifier judges *educational English web text*, so it's only used
when the agent says the data is English prose. Otherwise we fall back to comparing the
sample's measured profile with the reference's, and say so.
"""
from . import checks

FINEWEB_EDU = "HuggingFaceFW/fineweb-edu-classifier"
_model = None


def _load_classifier():
    global _model
    if _model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(FINEWEB_EDU)
        model = AutoModelForSequenceClassification.from_pretrained(FINEWEB_EDU).eval()
        _model = (tok, model)
    return _model


def classifier_mean(docs: list[str], max_docs: int = 300) -> float:
    """Mean FineWeb-Edu score (0 = not educational, 5 = highly educational)."""
    import torch
    tok, model = _load_classifier()
    scores = []
    for i in range(0, min(len(docs), max_docs), 16):
        batch = tok(docs[i:i + 16], return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            scores += model(**batch).logits.squeeze(-1).clamp(0, 5).tolist()
    return sum(scores) / len(scores)


def _profile(docs):
    return {
        "duplicate_rate": checks.check_duplication(docs)["duplicate_rate"],
        "junk_rate": checks.check_junk(docs)["junk_rate"],
        "avg_words_per_sentence": checks.check_readability(docs).get("avg_words_per_sentence"),
    }


def run_quality_benchmark(docs, ref_docs, use_classifier: bool) -> dict:
    out = {"sample_profile": _profile(docs), "reference_profile": _profile(ref_docs)}
    if use_classifier:
        try:
            s, r = classifier_mean(docs), classifier_mean(ref_docs)
            out.update(method="fineweb_edu_classifier", sample_score=round(s, 3), reference_score=round(r, 3),
                       ratio_to_reference=round(s / r, 3) if r else None)
            return out
        except ImportError:
            out["note"] = "transformers/torch not installed; used profile comparison instead"
    out["method"] = "profile_vs_reference"
    return out


def benchmark_fraction(b: dict) -> float:
    """0..1 share of the benchmark line's points."""
    clip = lambda x: max(0.0, min(1.0, x))
    if b.get("method") == "fineweb_edu_classifier" and b.get("ratio_to_reference") is not None:
        return clip(b["ratio_to_reference"])
    s, r = b["sample_profile"], b["reference_profile"]
    parts = [clip(1 - max(0, s["duplicate_rate"] - r["duplicate_rate"]) / 0.2),
             clip(1 - max(0, s["junk_rate"] - r["junk_rate"]) / 0.3)]
    if s["avg_words_per_sentence"] and r["avg_words_per_sentence"]:
        parts.append(clip(1 - abs(s["avg_words_per_sentence"] - r["avg_words_per_sentence"]) / r["avg_words_per_sentence"]))
    return sum(parts) / len(parts)
