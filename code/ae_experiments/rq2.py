from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .cache import ExperimentCache
from .common import cache_root, output_root, write_csv, write_json
from .pipeline import SIGMA_RUNS, member_documents, prepare_dataset


METHOD_LABELS = {"dcmi": "DCMI", "rag_mia": "RAG-MIA", "sigma": "SIGMA"}


def build_groups(method: str, prepared: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for doc_id, doc in member_documents(prepared)[:100]:
        if method == "dcmi":
            queries = [doc["original_query_before_rewrite"]]
        elif method == "rag_mia":
            queries = [doc["question_before_rewrite"]]
        elif method == "sigma":
            item = doc.get("open_qa_items", [])[0]
            queries = [
                item["runs"][run_name].get(
                    "composed_query_before_rewrite",
                    item["runs"][run_name]["composed_query"],
                )
                for run_name in SIGMA_RUNS
            ]
        else:
            raise ValueError(method)
        groups.append({"group_uid": f"{doc_id}::{method}", "doc_id": doc_id, "queries": queries})
    return groups


def run(output_dir: Path | None = None) -> Dict[str, Any]:
    destination = output_dir or (output_root() / "rq2")
    cache = ExperimentCache(cache_root())
    prepared = prepare_dataset("scifact", cache)
    methods: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for method in ("dcmi", "rag_mia", "sigma"):
        groups = build_groups(method, prepared[method])
        trace = []
        passed_groups = 0
        for group in groups:
            query_results = []
            for query_index, query in enumerate(group["queries"]):
                cached = cache.get_lakera(method, query)
                query_results.append({
                    "query_uid": f"{group['group_uid']}::{query_index}",
                    "query": query,
                    "status": cached.get("status", "ok"),
                    "flagged": bool(cached.get("flagged", False)),
                    "unsafe": cached.get("unsafe"),
                })
            passed = any(
                item["status"] == "ok" and item["flagged"] is False
                for item in query_results
            )
            passed_groups += int(passed)
            trace.append({**group, "queries": query_results, "passed": passed})
        metrics = {
            "probing_unit_loss_rate": (len(trace) - passed_groups) / len(trace),
        }
        methods[method] = metrics
        rows.append({"method": METHOD_LABELS[method], **metrics})
        write_json(destination / "traces" / f"{method}.json", trace)
    result = {
        "meta": {"rq": "RQ2", "dataset": "SciFact", "defense": "Lakera Guard"},
        "methods": methods,
    }
    write_json(destination / "results.json", result)
    write_csv(
        destination / "results.csv",
        rows,
        ("method", "probing_unit_loss_rate"),
    )
    return result
