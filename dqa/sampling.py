"""load_sample: pull a capped sample of a dataset and save it to disk.

Context-management rule from the spec: the dataset text NEVER goes back to the agent.
This tool saves the sample to a file and returns only a short reference + stats.
"""
import hashlib
import json
import re
from pathlib import Path

SAMPLE_DIR = Path(".dqa_samples")
MAX_ROWS = 5000                     # guardrail: hard cap on rows pulled per dataset
TEXT_FIELD_GUESSES = ["text", "content", "document", "code", "body", "article", "conversation", "messages"]


def _slug(name: str) -> str:
    """Short readable id + a hash so two long names can never collide."""
    readable = re.sub(r"[^a-zA-Z0-9]+", "_", name.split("/")[-1] or name).strip("_").lower()[:40]
    return f"{readable}_{hashlib.md5(name.encode()).hexdigest()[:6]}"


def _to_text(value) -> str:
    """Turn a field value into plain text (chat datasets often store lists of messages)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for m in value:
            if isinstance(m, dict):
                role = m.get("role") or m.get("from") or ""
                parts.append(f"{role}: {m.get('content') or m.get('value') or ''}".strip())
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return "" if value is None else str(value)


def _pick_text_field(row: dict, requested: str | None) -> str:
    if requested:
        if requested not in row:
            raise ValueError(f"Field '{requested}' not found. Available fields: {list(row)}")
        return requested
    for f in TEXT_FIELD_GUESSES:
        if f in row:
            return f
    # fall back to the longest string field
    strings = {k: len(v) for k, v in row.items() if isinstance(v, str)}
    if not strings:
        raise ValueError(f"No text-like field found. Available fields: {list(row)}")
    return max(strings, key=strings.get)


def _iter_rows(dataset: str, config: str | None, split: str):
    """Local .jsonl files (for tests/evals) or a Hugging Face dataset (streamed, never fully downloaded)."""
    path = Path(dataset)
    if path.suffix == ".jsonl" and path.exists():
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    from datasets import load_dataset  # imported lazily so tests run without network
    ds = load_dataset(dataset, config, split=split, streaming=True)
    yield from ds.shuffle(seed=0, buffer_size=10_000)   # shuffle so we don't just read the first shard


def load_sample(dataset: str, n: int = 3000, split: str = "train",
                config: str | None = None, text_field: str | None = None,
                sample_id: str | None = None) -> dict:
    n = max(10, min(int(n), MAX_ROWS))
    SAMPLE_DIR.mkdir(exist_ok=True)
    sample_id = sample_id or _slug(f"{dataset}_{config or ''}_{split}")
    out = SAMPLE_DIR / f"{sample_id}.jsonl"

    docs, field = [], None
    for row in _iter_rows(dataset, config, split):
        if field is None:
            field = _pick_text_field(row, text_field)
            fields = list(row)
        text = _to_text(row.get(field))
        if text.strip():
            docs.append(text)
        if len(docs) >= n:
            break
    if not docs:
        raise ValueError(f"No non-empty documents found in {dataset}")

    with out.open("w") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")

    words = [len(d.split()) for d in docs]
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "docs": len(docs),
        "text_field": field,
        "all_fields": fields,
        "avg_words_per_doc": round(sum(words) / len(words), 1),
    }


def read_sample(sample_id: str) -> list[str]:
    path = SAMPLE_DIR / f"{sample_id}.jsonl"
    if not path.exists():
        raise ValueError(f"Unknown sample '{sample_id}'. Call load_sample first.")
    with path.open() as f:
        return [json.loads(line)["text"] for line in f if line.strip()]
