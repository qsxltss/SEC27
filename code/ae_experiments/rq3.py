from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .cache import ExperimentCache
from .common import cache_root, output_root, write_csv, write_json
from .pipeline import SIGMA_RUNS, member_documents, prepare_dataset


WORD_PATTERN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
METHOD_LABELS = {"dcmi": "DCMI", "rag_mia": "RAG-MIA", "sigma": "SIGMA"}


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def tokens(value: str) -> List[str]:
    return WORD_PATTERN.findall(normalize(value).lower())


def target_text(doc: Dict[str, Any]) -> str:
    title, body = normalize(doc.get("title")), normalize(doc.get("text"))
    return f"{title}\n{body}" if title and body else title or body


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l_f1(query: str, target: str) -> float:
    q, t = tokens(query), tokens(target)
    if not q or not t:
        return 0.0
    lcs = lcs_length(q, t)
    if not lcs:
        return 0.0
    precision, recall = lcs / len(q), lcs / len(t)
    return 2 * precision * recall / (precision + recall)


def bleu_4(query: str, target: str) -> float:
    q, t = tokens(query), tokens(target)
    if not q or not t:
        return 0.0
    return float(sentence_bleu([t], q, smoothing_function=SmoothingFunction().method1))


def build_records(prepared: Dict[str, Dict[str, Dict[str, Any]]], seed: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for doc_id, doc in member_documents(prepared["dcmi"]):
        records.append({"method": "dcmi", "doc_id": doc_id, "query": doc["perturbed_query_before_rewrite"], "group": "primary", "target": target_text(doc)})
    for doc_id, doc in member_documents(prepared["rag_mia"]):
        records.append({"method": "rag_mia", "doc_id": doc_id, "query": doc["question_before_rewrite"], "group": "primary", "target": target_text(doc)})

    rng = random.Random(seed)
    sigma_members = member_documents(prepared["sigma"])
    rng.shuffle(sigma_members)
    sampled = sigma_members[:50]
    for run_name in SIGMA_RUNS:
        for doc_id, doc in sorted(sampled, key=lambda pair: pair[0]):
            candidates = []
            for item in doc.get("open_qa_items", []):
                run = item.get("runs", {}).get(run_name, {})
                query = normalize(
                    run.get("composed_query_before_rewrite", run.get("composed_query", ""))
                )
                if query:
                    candidates.append(query)
            query = candidates[rng.randrange(len(candidates))]
            records.append({"method": "sigma", "doc_id": doc_id, "query": query, "group": run_name, "target": target_text(doc)})
    return records


def run(output_dir: Path | None = None, seed: int = 42) -> Dict[str, Any]:
    destination = output_dir or (output_root() / "rq3")
    cache = ExperimentCache(cache_root())
    prepared = prepare_dataset("scifact", cache)
    records = build_records(prepared, seed)
    if len(records) != 300:
        raise ValueError(f"Expected 300 privacy-exposure queries, found {len(records)}")
    rows = []
    for index, record in enumerate(records):
        rows.append({
            "method": record["method"],
            "doc_id": record["doc_id"],
            "query_index": index,
            "sampling_group": record["group"],
            "query_text": record["query"],
            "rouge_l_f1": rouge_l_f1(record["query"], record["target"]),
            "bleu_4": bleu_4(record["query"], record["target"]),
        })
    methods: Dict[str, Any] = {}
    for method in METHOD_LABELS:
        selected = [row for row in rows if row["method"] == method]
        methods[method] = {
            metric: float(
                np.asarray(
                    [float(row[metric]) for row in selected],
                    dtype=np.float64,
                ).mean()
            )
            for metric in ("rouge_l_f1", "bleu_4")
        }
    result = {
        "meta": {"rq": "RQ3", "dataset": "SciFact", "retriever": "BGE", "seed": seed},
        "methods": methods,
    }
    write_json(destination / "results.json", result)
    write_csv(destination / "per_query.csv", rows, tuple(rows[0].keys()))
    return result
