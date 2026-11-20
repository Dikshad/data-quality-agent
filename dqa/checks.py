"""The measurable checks. Plain code: cheap, fast, and gives the same answer every run.

Each check returns a small dict of numbers. No dataset text is ever returned,
so nothing in the data can reach (or hijack) the agent.
"""
import hashlib
import re
import statistics
from collections import Counter

from datasketch import MinHash, MinHashLSH

# ---------------------------------------------------------------- duplication
NEAR_DUP_THRESHOLD = 0.8   # BigCode default: ~80% similar = duplicate
SHINGLE_WORDS = 5          # compare 5-word sequences
NUM_PERM = 128


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def _minhash(words: list[str]) -> MinHash:
    m = MinHash(num_perm=NUM_PERM, seed=1)
    if len(words) < SHINGLE_WORDS:
        shingles = {" ".join(words)}
    else:
        shingles = {" ".join(words[i:i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}
    for s in shingles:
        m.update(s.encode())
    return m


def check_duplication(docs: list[str], threshold: float = NEAR_DUP_THRESHOLD, exclude_junk: bool = True) -> dict:
    """Share of documents you'd remove if you kept one copy per duplicate cluster.

    Junk is filtered out first (as real cleaning pipelines do), so a broken page that
    appears 50 times is counted once, as junk, not penalised twice.
    """
    total = len(docs)
    if exclude_junk:
        docs = [d for d in docs if not junk_flags(d)] or docs
    excluded = total - len(docs)
    n = len(docs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # 1) exact duplicates (after normalising case/punctuation/whitespace)
    seen, exact_removed = {}, 0
    normalized = [_normalize(d) for d in docs]
    for i, t in enumerate(normalized):
        h = hashlib.md5(t.encode()).hexdigest()
        if h in seen:
            union(i, seen[h])
            exact_removed += 1
        else:
            seen[h] = i

    # 2) near duplicates with MinHash + LSH (finds ~80%-similar pairs without comparing every pair)
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    for i, t in enumerate(normalized):
        m = _minhash(t.split())
        for j in lsh.query(m):
            union(i, int(j))
        lsh.insert(str(i), m)

    clusters = Counter(find(i) for i in range(n))
    removed = n - len(clusters)
    biggest = sorted((c for c in clusters.values() if c > 1), reverse=True)[:5]
    return {
        "duplicate_rate": round(removed / n, 4),
        "exact_duplicate_rate": round(exact_removed / n, 4),
        "near_duplicate_rate": round((removed - exact_removed) / n, 4),
        "duplicate_clusters": sum(1 for c in clusters.values() if c > 1),
        "largest_cluster_sizes": biggest,   # one huge cluster usually means template/boilerplate pages
        "docs_excluded_as_junk": excluded,
    }


# ---------------------------------------------------------------- readability
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_VOWELS = re.compile(r"[aeiouy]+")


def _syllables(word: str) -> int:
    return max(1, len(_VOWELS.findall(word.lower())))


def check_readability(docs: list[str]) -> dict:
    per_doc_wps, total_words, total_sents, total_syll = [], 0, 0, 0
    for d in docs:
        sents = [s for s in _SENT_SPLIT.split(d) if s.strip()]
        words = re.findall(r"[A-Za-zÀ-ɏ']+", d)
        if not sents or not words:
            continue
        per_doc_wps.append(len(words) / len(sents))
        total_words += len(words)
        total_sents += len(sents)
        total_syll += sum(_syllables(w) for w in words)
    if not per_doc_wps:
        return {"measurable": False, "reason": "no word-like text found"}
    wps = total_words / total_sents
    flesch = 206.835 - 1.015 * wps - 84.6 * (total_syll / total_words)   # Flesch reading ease
    return {
        "measurable": True,
        "avg_words_per_sentence": round(wps, 2),
        "median_doc_words_per_sentence": round(statistics.median(per_doc_wps), 2),
        "share_docs_under_5_words_per_sentence": round(sum(w < 5 for w in per_doc_wps) / len(per_doc_wps), 4),
        "flesch_reading_ease": round(flesch, 1),
    }


# ---------------------------------------------------------------- language + content mix
_CODE_HINTS = re.compile(r"[{};]|\bdef |\bclass |\bimport |\breturn\b|=>|==|\bfunction\b|#include")
_CHAT_HINTS = re.compile(r"^(user|assistant|human|gpt|system)\s*:", re.I | re.M)


def _content_type(doc: str) -> str:
    lines = [l for l in doc.splitlines() if l.strip()] or [doc]
    if len(_CHAT_HINTS.findall(doc)) >= 2:
        return "chat"
    if sum(bool(_CODE_HINTS.search(l)) for l in lines) / len(lines) > 0.3:
        return "code"
    return "prose"


def check_language_mix(docs: list[str], max_docs: int = 1500) -> dict:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0                        # make language detection deterministic
    subset = docs[:max_docs]
    langs, types = Counter(), Counter()
    for d in subset:
        types[_content_type(d)] += 1
        try:
            langs[detect(d[:1000])] += 1
        except Exception:
            langs["unknown"] += 1
    n = len(subset)
    return {
        "docs_checked": n,
        "languages": {k: round(v / n, 4) for k, v in langs.most_common(8)},
        "content_types": {k: round(v / n, 4) for k, v in types.most_common()},
    }


# ---------------------------------------------------------------- junk
_BOILERPLATE = re.compile(
    r"cookie|javascript (is )?(disabled|required)|subscribe (now|to our)|all rights reserved|"
    r"404 not found|page not found|access denied|enable javascript|click here", re.I)
_HTML_TAG = re.compile(r"</?[a-z][a-z0-9]*[^>]*>", re.I)


def junk_flags(doc: str) -> list[str]:
    flags = []
    if not doc:
        return ["empty"]
    nonprint = sum(1 for c in doc if not c.isprintable() and c not in "\n\t\r")
    if "\x00" in doc or nonprint / len(doc) > 0.05 or doc.count("�") / len(doc) > 0.02:
        flags.append("binary_or_garbled")
    letters = sum(c.isalpha() for c in doc)
    if letters / len(doc) < 0.5:
        flags.append("mostly_non_text")
    lines = [l.strip() for l in doc.splitlines() if l.strip()]
    if len(lines) >= 4 and 1 - len(set(lines)) / len(lines) > 0.3:
        flags.append("repeated_lines")
    if len(_HTML_TAG.findall(doc)) > 5:
        flags.append("html_remnants")
    if len(doc.split()) < 20:
        flags.append("too_short")
    if len(_BOILERPLATE.findall(doc)) >= 2 or (len(doc.split()) < 80 and _BOILERPLATE.search(doc)):
        flags.append("boilerplate_or_broken_page")
    return flags


def check_junk(docs: list[str]) -> dict:
    counts, junk_docs = Counter(), 0
    for d in docs:
        f = junk_flags(d)
        counts.update(f)
        junk_docs += bool(f)
    n = len(docs)
    return {
        "junk_rate": round(junk_docs / n, 4),
        "by_reason": {k: round(v / n, 4) for k, v in counts.most_common()},
    }
