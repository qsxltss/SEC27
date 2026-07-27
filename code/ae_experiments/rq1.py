from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .cache import ExperimentCache
from .common import (
    cache_root,
    data_root,
    index_root,
    normalize_text,
    output_root,
    score_summary,
    write_csv,
    write_json,
)
from .pipeline import (
    DATASETS,
    METHODS,
    RETRIEVERS,
    SIGMA_RUNS,
    evaluate_rq1_configuration,
    prepare_dataset,
)
from .prompts import rewrite_request
from .retrievers import DatasetRepository, RetrievalManager


METHOD_LABELS = {"dcmi": "DCMI", "rag_mia": "RAG-MIA", "sigma": "SIGMA"}


def rewrite_queries(
    method: str,
    dataset: str,
    prepared: Dict[str, Dict[str, Any]],
    cache: ExperimentCache,
) -> Dict[str, Dict[str, Any]]:
    data = copy.deepcopy(prepared)
    if method == "dcmi":
        for doc_id, doc in data.items():
            for label in ("original", "perturbed"):
                source = doc[f"{label}_query_before_rewrite"]
                doc[f"{label}_query"] = cache.get_rewrite(
                    method,
                    dataset,
                    f"{doc_id}::{label}",
                    rewrite_request(source),
                )
        return data
    if method == "rag_mia":
        for doc_id, doc in data.items():
            source = doc["question_before_rewrite"]
            doc["question"] = cache.get_rewrite(
                method,
                dataset,
                f"{doc_id}::0",
                rewrite_request(source),
            )
        return data
    if method == "sigma":
        for doc_id, doc in data.items():
            for fallback_index, item in enumerate(doc.get("open_qa_items", [])):
                question_id = item.get("question_id", fallback_index)
                for run_name in SIGMA_RUNS:
                    run = item.get("runs", {}).get(run_name, {})
                    source = normalize_text(
                        run.get(
                            "composed_query_before_rewrite",
                            run.get("composed_query", ""),
                        )
                    )
                    run["composed_query"] = cache.get_rewrite(
                        method,
                        dataset,
                        f"{doc_id}::{question_id}::{run_name}",
                        rewrite_request(source),
                    )
        return data
    raise ValueError(f"Unsupported method: {method}")


def run(
    output_dir: Path | None = None,
    datasets: Iterable[str] = DATASETS,
    retrievers: Iterable[str] = RETRIEVERS,
    methods: Iterable[str] = METHODS,
) -> Dict[str, Any]:
    cache_dir = cache_root()
    destination = output_dir or (output_root() / "rq1")
    cache = ExperimentCache(cache_dir)
    datasets_root = data_root() / "datasets"
    repository = DatasetRepository(datasets_root)
    retrieval = RetrievalManager(datasets_root, index_root())

    configurations: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    selected_datasets = tuple(datasets)
    selected_retrievers = tuple(retrievers)
    selected_methods = tuple(methods)
    for dataset in selected_datasets:
        print(f"[RQ1] dataset={dataset} preparing attack inputs", flush=True)
        base = prepare_dataset(dataset, cache)
        rewritten = {}
        for method in selected_methods:
            print(f"[RQ1] dataset={dataset} rewriting method={method}", flush=True)
            rewritten[method] = rewrite_queries(method, dataset, base[method], cache)
            write_json(
                destination / "prepared_queries" / method / f"{dataset}.json",
                rewritten[method],
            )
        configurations[dataset] = {}
        for retriever_name in selected_retrievers:
            configurations[dataset][retriever_name] = {}
            for method in selected_methods:
                print(f"[RQ1] dataset={dataset} retriever={retriever_name} method={method}", flush=True)
                trace, decisions = evaluate_rq1_configuration(
                    method=method,
                    dataset=dataset,
                    retriever=retriever_name,
                    prepared=rewritten[method],
                    cache=cache,
                    repository=repository,
                    retrieval=retrieval,
                )
                metrics = score_summary(decisions)
                configurations[dataset][retriever_name][method] = metrics
                rows.append({
                    "dataset": dataset,
                    "retriever": retriever_name.upper(),
                    "method": METHOD_LABELS[method],
                    "accuracy": metrics["accuracy"],
                    "correct_count": metrics["correct_count"],
                    "document_count": metrics["document_count"],
                })
                write_json(destination / "traces" / method / f"{dataset}_{retriever_name}.json", trace)
                write_json(destination / "decisions" / method / f"{dataset}_{retriever_name}.json", decisions)

    result = {
        "meta": {
            "rq": "RQ1",
            "generator": "DeepSeek-V3.2",
            "methods": list(selected_methods),
            "datasets": list(selected_datasets),
            "retrievers": list(selected_retrievers),
            "execution": (
                "query preparation -> query rewrite -> retrieval -> "
                "cache-through target response -> decision"
            ),
        },
        "configurations": configurations,
    }
    write_json(destination / "results.json", result)
    write_csv(
        destination / "results.csv",
        rows,
        (
            "dataset",
            "retriever",
            "method",
            "accuracy",
            "correct_count",
            "document_count",
        ),
    )
    return result
