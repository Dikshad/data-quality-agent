"""The agent's toolbox: what the model sees (TOOLS) and what actually runs (ToolBox).

Guardrails live HERE, in code, not in the prompt. The model can't talk its way past them:
  * search cap (5), dataset cap (5), row caps, judge-doc cap
  * up to 3 attempts per tool call, then a clear error back to the agent
  * read-only: no tool can modify or upload a dataset
  * every tool returns a short summary; dataset text never enters the agent's context
"""
import json
import time
from pathlib import Path

from . import checks, scoring
from .benchmark import run_quality_benchmark
from .judge import llm_judge
from .sampling import load_sample, read_sample

MAX_SEARCHES = 5
MAX_DATASETS = 5
MAX_ATTEMPTS = 3
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

_sample = {"type": "string", "description": "sample_id returned by load_sample"}
TOOLS = [
    {"name": "read_skill",
     "description": "Load a skill (a playbook of expert instructions). Read the relevant skill before starting a task.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "search_datasets",
     "description": f"Search Hugging Face for datasets. Returns ids, downloads and tags (language, task, size). "
                    f"Limited to {MAX_SEARCHES} searches per session.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Short keywords, e.g. 'medical question answering'"},
         "language": {"type": "string", "description": "Optional ISO code filter, e.g. 'en'"}},
         "required": ["query"]}},
    {"name": "load_sample",
     "description": "Pull a capped random sample from a Hugging Face dataset (or local .jsonl) and save it. Returns a "
                    "sample_id and basic stats, never the text. Use it for candidates AND for reference datasets.",
     "input_schema": {"type": "object", "properties": {
         "dataset": {"type": "string", "description": "e.g. 'HuggingFaceFW/fineweb-edu'"},
         "config": {"type": "string", "description": "Optional subset/config name, e.g. 'sample-10BT'"},
         "split": {"type": "string", "default": "train"},
         "text_field": {"type": "string", "description": "Optional. Field holding the text; auto-detected if omitted."},
         "n": {"type": "integer", "description": "Rows to sample (max 5000). Default 3000."},
         "is_reference": {"type": "boolean", "description": "True if this is a gold-standard reference, not a candidate."}},
         "required": ["dataset"]}},
    {"name": "check_language_mix",
     "description": "Share of each language and content type (prose/code/chat). Run this FIRST on a sample to learn its data type.",
     "input_schema": {"type": "object", "properties": {"sample_id": _sample}, "required": ["sample_id"]}},
    {"name": "check_duplication",
     "description": "Exact + near-duplicate rate between documents (near = ~80% similar, MinHash). Also reports "
                    "largest duplicate clusters; one huge cluster usually means template pages.",
     "input_schema": {"type": "object", "properties": {"sample_id": _sample}, "required": ["sample_id"]}},
    {"name": "check_readability",
     "description": "Average words per sentence, share of docs under 5 words/sentence, Flesch reading ease.",
     "input_schema": {"type": "object", "properties": {"sample_id": _sample}, "required": ["sample_id"]}},
    {"name": "check_junk",
     "description": "Share of docs that are binary/garbled, mostly non-text, repeated lines, HTML remnants, "
                    "too short, or boilerplate/broken pages, with a breakdown by reason.",
     "input_schema": {"type": "object", "properties": {"sample_id": _sample}, "required": ["sample_id"]}},
    {"name": "run_quality_benchmark",
     "description": "Compare a candidate with a reference sample of the SAME data type. use_classifier=true runs the "
                    "FineWeb-Edu classifier, valid ONLY for English prose/web text.",
     "input_schema": {"type": "object", "properties": {
         "sample_id": _sample, "reference_sample_id": _sample,
         "use_classifier": {"type": "boolean"}}, "required": ["sample_id", "reference_sample_id", "use_classifier"]}},
    {"name": "llm_judge",
     "description": "An LLM grades a random subset of the sample (1-5) on coherence, informativeness, boilerplate, "
                    "personal info and fit with the researcher's purpose. Costs money; default 100 docs, max 300.",
     "input_schema": {"type": "object", "properties": {
         "sample_id": _sample, "purpose": {"type": "string", "description": "The researcher's stated purpose"},
         "n": {"type": "integer"}}, "required": ["sample_id", "purpose"]}},
    {"name": "score_dataset",
     "description": "Compute the weighted 0-100 scorecard from the checks already run on this sample. The maths is "
                    "done in code. Lines with no check run are marked 'couldn't measure'.",
     "input_schema": {"type": "object", "properties": {
         "sample_id": _sample,
         "data_type": {"type": "string", "enum": ["prose", "code", "chat"]},
         "target_languages": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['en']"},
         "target_content_type": {"type": "string", "enum": ["prose", "code", "chat"]}},
         "required": ["sample_id", "data_type"]}},
    {"name": "write_report",
     "description": "Write the final markdown report. Scorecards are pulled from score_dataset results (you cannot "
                    "edit the numbers). You supply the ranking and the explanation.",
     "input_schema": {"type": "object", "properties": {
         "ranked_sample_ids": {"type": "array", "items": {"type": "string"}, "description": "Best first"},
         "researcher_need": {"type": "string"},
         "recommendation": {"type": "string", "description": "Why the top pick fits; caveats; what couldn't be measured"}},
         "required": ["ranked_sample_ids", "researcher_need", "recommendation"]}},
]


class GuardrailError(Exception):
    """Raised when a cap is hit. Not retried."""


class ToolBox:
    def __init__(self, judge_client=None, report_path="report.md", search_fn=None):
        self.results: dict[str, dict] = {}        # sample_id -> {check_name: result}
        self.meta: dict[str, dict] = {}
        self.searches = 0
        self.candidates: set[str] = set()
        self.judge_client = judge_client
        self.report_path = report_path
        self.search_fn = search_fn
        self.log: list[dict] = []                 # observability: every call, its outcome and time

    # -------- dispatch with retries + logging
    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, f"_{name}", None)
        if fn is None:
            return f"Error: unknown tool {name}"
        last = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            t0 = time.time()
            try:
                out = fn(**args)
                self.log.append({"tool": name, "args": args, "ok": True, "attempt": attempt, "secs": round(time.time() - t0, 2)})
                return out if isinstance(out, str) else json.dumps(out)
            except (GuardrailError, ValueError, TypeError) as e:     # not worth retrying
                last = e
                break
            except Exception as e:                                   # network blips etc.: retry
                last = e
                time.sleep(min(2 ** attempt, 8) if attempt < MAX_ATTEMPTS else 0)
        self.log.append({"tool": name, "args": args, "ok": False, "error": str(last)})
        return f"Error after {attempt} attempt(s): {last}. Continue with the other checks and mark this one as couldn't measure."

    def _docs(self, sample_id):
        return read_sample(sample_id)

    def _store(self, sample_id, key, value):
        self.results.setdefault(sample_id, {})[key] = value
        return value

    # -------- tools
    def _read_skill(self, name):
        path = SKILLS_DIR / name / "SKILL.md"
        if not path.exists():
            raise ValueError(f"No skill '{name}'. Available: {[p.name for p in SKILLS_DIR.iterdir() if p.is_dir()]}")
        return path.read_text()

    def _search_datasets(self, query, language=None):
        if self.searches >= MAX_SEARCHES:
            raise GuardrailError(f"Search cap of {MAX_SEARCHES} reached. Tell the researcher where you searched "
                                 f"and ask whether to continue.")
        self.searches += 1
        if self.search_fn:
            hits = self.search_fn(query, language)
        else:
            from huggingface_hub import HfApi
            kw = {"language": language} if language else {}
            hits = []
            for d in HfApi().list_datasets(search=query, sort="downloads", limit=8, **kw):
                tags = [t for t in (d.tags or []) if t.split(":")[0] in ("language", "task_categories", "size_categories", "modality")]
                hits.append({"id": d.id, "downloads": d.downloads, "tags": tags[:12]})
        return {"search_number": f"{self.searches}/{MAX_SEARCHES}", "query": query, "results": hits}

    def _load_sample(self, dataset, config=None, split="train", text_field=None, n=3000, is_reference=False):
        if not is_reference and dataset not in {m["dataset"] for m in self.meta.values()} \
                and len(self.candidates) >= MAX_DATASETS:
            raise GuardrailError(f"Dataset cap of {MAX_DATASETS} candidates reached.")
        info = load_sample(dataset, n=n, split=split, config=config, text_field=text_field)
        self.meta[info["sample_id"]] = {**info, "is_reference": is_reference}
        if not is_reference:
            self.candidates.add(info["sample_id"])
        return info

    def _check_language_mix(self, sample_id):
        return self._store(sample_id, "language_mix", checks.check_language_mix(self._docs(sample_id)))

    def _check_duplication(self, sample_id):
        return self._store(sample_id, "duplication", checks.check_duplication(self._docs(sample_id)))

    def _check_readability(self, sample_id):
        return self._store(sample_id, "readability", checks.check_readability(self._docs(sample_id)))

    def _check_junk(self, sample_id):
        return self._store(sample_id, "junk", checks.check_junk(self._docs(sample_id)))

    def _run_quality_benchmark(self, sample_id, reference_sample_id, use_classifier):
        if sample_id == reference_sample_id:
            raise ValueError("Candidate and reference must be different samples.")
        b = run_quality_benchmark(self._docs(sample_id), self._docs(reference_sample_id), use_classifier)
        b["reference_sample_id"] = reference_sample_id
        return self._store(sample_id, "benchmark", b)

    def _llm_judge(self, sample_id, purpose, n=100):
        out = llm_judge(self._docs(sample_id), purpose, n=n, client=self.judge_client)
        return self._store(sample_id, "llm_judge", out)

    def _score_dataset(self, sample_id, data_type, target_languages=None, target_content_type=None):
        if sample_id not in self.results:
            raise ValueError(f"No checks have been run on {sample_id} yet.")
        s = scoring.score(self.results[sample_id], data_type, target_languages, target_content_type)
        return self._store(sample_id, "score", s)

    def _write_report(self, ranked_sample_ids, researcher_need, recommendation):
        missing = [s for s in ranked_sample_ids if "score" not in self.results.get(s, {})]
        if missing:
            raise ValueError(f"Run score_dataset first for: {missing}")
        from .report import render
        md = render(ranked_sample_ids, self.results, self.meta, researcher_need, recommendation)
        Path(self.report_path).write_text(md)
        return f"Report written to {self.report_path}"
