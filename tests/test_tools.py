"""Tool tests (eval layer 1): plant a known problem, check the tool measures it."""
import json

import pytest

from dqa import checks, synthetic as syn
from dqa.sampling import load_sample, read_sample

CLEAN = syn.clean_docs(800)


@pytest.mark.parametrize("rate", [0.05, 0.10, 0.20, 0.30])
def test_duplication_measures_planted_rate(rate):
    out = checks.check_duplication(syn.add_near_duplicates(CLEAN, rate))
    assert out["duplicate_rate"] == pytest.approx(rate, abs=0.02)


def test_clean_data_has_no_duplicates():
    assert checks.check_duplication(CLEAN)["duplicate_rate"] < 0.01


def test_different_articles_on_same_topic_are_not_duplicates():
    a = "The central bank raised interest rates by a quarter point on Tuesday, citing persistent inflation in housing and services, and signalled more increases could follow."
    b = "Officials lifted borrowing costs this week. Mortgage holders will feel it first, analysts said, while savers may finally see better returns on deposits."
    assert checks.check_duplication([a, b])["duplicate_rate"] == 0


def test_readability_flags_chopped_sentences():
    out = checks.check_readability(syn.replace_fraction(CLEAN, 0.5, syn.chopped_doc))
    assert out["share_docs_under_5_words_per_sentence"] == pytest.approx(0.5, abs=0.02)


def test_junk_rate_matches_planted_rate():
    out = checks.check_junk(syn.replace_fraction(CLEAN, 0.2, syn.junk_doc))
    assert out["junk_rate"] == pytest.approx(0.2, abs=0.02)
    assert checks.check_junk(CLEAN)["junk_rate"] == 0


def test_language_mix_matches_planted_share():
    out = checks.check_language_mix(syn.replace_fraction(CLEAN, 0.3, syn.foreign_doc("es")))
    assert out["languages"]["es"] == pytest.approx(0.3, abs=0.03)


def test_content_type_detects_code_and_chat():
    code = "def add(a, b):\n    return a + b\n\nclass Foo:\n    x = 1;\n"
    chat = "User: hi there\nAssistant: hello, how can I help?\nUser: tell me a fact"
    assert checks._content_type(code) == "code"
    assert checks._content_type(chat) == "chat"


def test_load_sample_caps_rows_and_never_returns_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "d.jsonl"
    f.write_text("\n".join(json.dumps({"id": i, "content": t}) for i, t in enumerate(CLEAN)))
    out = load_sample(str(f), n=100, sample_id="t")
    assert out["docs"] == 100 and out["text_field"] == "content"
    assert not any(isinstance(v, str) and len(v) > 200 for v in out.values())  # no document text leaks back
    assert len(read_sample("t")) == 100
