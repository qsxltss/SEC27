from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .cache import ExperimentCache
from .common import (
    actual_membership,
    cache_root,
    data_root,
    load_json,
    normalize_text,
    normalize_yes_no,
)
from .prompts import dcmi_perturb_prompt, no_document_prompt, rag_prompt, sigma_judge_prompt
from .retrievers import DatasetRepository, RetrievalManager


METHODS = ("dcmi", "rag_mia", "sigma")
DATASETS = ("nfcorpus", "scifact", "trec-covid")
RETRIEVERS = ("bge", "bm25")
SIGMA_RUNS = ("with_summary_level_1", "without_summary_level_1")
TARGET_MAX_TOKENS = {"yes_no": 5, "open": 256}


def load_target_docs(dataset: str, root: Path | None = None) -> Dict[str, Dict[str, Any]]:
    value = load_json((root or data_root()) / "inputs" / f"{dataset}.json")
    if not isinstance(value, dict) or len(value) != 200:
        raise ValueError(f"Expected 200 fixed target documents for {dataset}")
    return value


def _dcmi_base_record(doc: Dict[str, Any], dataset: str, doc_id: str, cache: ExperimentCache) -> Dict[str, Any]:
    document = normalize_text(doc.get("text", ""))
    prompt = dcmi_perturb_prompt(document)
    perturbed_text = normalize_text(cache.get_dcmi_perturbation(dataset, doc_id, prompt))
    words = len(re.findall(r"\b\w+(?:[-']\w+)?\b", document))
    return {
        "word_count": words,
        "num_to_replace": max(1, round(0.06 * words)),
        "original_query_before_rewrite": f'Is this: "{document}" right? Answer with Yes or No.',
        "perturbed_query_before_rewrite": f'Is this: "{perturbed_text}" right? Answer with Yes or No.',
        "perturbed_text": perturbed_text,
    }


def _rag_mia_base_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    document = str(doc.get("text", ""))
    return {
        "question_before_rewrite": f'Does this:\n"{document}\n"appear in the context? Answer with Yes or No'
    }


def prepare_dataset(dataset: str, cache: ExperimentCache) -> Dict[str, Dict[str, Any]]:
    docs = load_target_docs(dataset)
    prepared: Dict[str, Dict[str, Any]] = {method: {} for method in METHODS}
    for doc_id, doc in docs.items():
        prepared["dcmi"][doc_id] = {**copy.deepcopy(doc), **_dcmi_base_record(doc, dataset, doc_id, cache)}
        prepared["rag_mia"][doc_id] = {**copy.deepcopy(doc), **_rag_mia_base_record(doc)}
        prepared["sigma"][doc_id] = {
            **copy.deepcopy(doc),
            "open_qa_items": cache.get_sigma_items(dataset, doc_id, normalize_text(doc.get("text", ""))),
        }
    return prepared


def _fallback_needed(response: str, answer_mode: str) -> bool:
    text = normalize_text(response).lower()
    if not text or "i don't know" in text or "i do not know" in text:
        return True
    return answer_mode == "yes_no" and normalize_yes_no(text) == "unknown"


def _evaluate_query(
    *,
    cache: ExperimentCache,
    repository: DatasetRepository,
    retrieval: RetrievalManager,
    target_docs: Dict[str, Dict[str, Any]],
    method: str,
    dataset: str,
    retriever: str,
    unit_id: str,
    query: str,
    answer_mode: str,
) -> Dict[str, Any]:
    retrieved_ids = retrieval.search(dataset, retriever, query, 3)
    contexts = repository.contexts(dataset, retrieved_ids, target_docs)
    rp = rag_prompt(query, contexts)
    np = no_document_prompt(query)
    cached, cache_origin = cache.find_target_response(
        method, dataset, retriever, unit_id, retrieved_ids, rp, np
    )
    if cached is None:
        print(
            f"[target cache] {cache_origin}: "
            f"{method}/{dataset}/{retriever}/{unit_id}; querying DeepSeek",
            flush=True,
        )
        max_tokens = TARGET_MAX_TOKENS[answer_mode]
        rag_response = cache.query_target(rp, max_tokens=max_tokens)
        use_fallback = _fallback_needed(rag_response, answer_mode)
        no_doc_response = (
            cache.query_target(np, max_tokens=max_tokens) if use_fallback else ""
        )
        source = "no_doc_fallback" if use_fallback else "rag"
        cached = cache.store_target_response(
            method=method,
            dataset=dataset,
            retriever=retriever,
            unit_id=unit_id,
            retrieved_doc_ids=retrieved_ids,
            rag_prompt=rp,
            no_doc_prompt=np,
            rag_response=rag_response,
            no_doc_response=no_doc_response,
            response_source=source,
        )
        cache_origin = "live"
    else:
        rag_response = str(cached.get("rag_response", ""))
        use_fallback = _fallback_needed(rag_response, answer_mode)
        no_doc_response = str(cached.get("no_doc_response", "")) if use_fallback else ""
        source = "no_doc_fallback" if use_fallback else "rag"
        expected_source = str(cached.get("response_source", source))
        if source != expected_source:
            raise RuntimeError(
                f"Fallback decision differs for {method}/{dataset}/{retriever}/{unit_id}: "
                f"computed={source}, cached={expected_source}"
            )
    final_response = no_doc_response if use_fallback else rag_response
    result = {
        "query": query,
        "retrieved_doc_ids": retrieved_ids,
        "rag_response": rag_response,
        "no_doc_response": no_doc_response,
        "response": final_response,
        "response_source": source,
        "target_response_cache": cache_origin,
    }
    if cached.get("provenance"):
        result["cache_provenance"] = str(cached["provenance"])
    return result


def _json_object_from_text(value: str) -> Dict[str, Any] | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _sigma_judgement(
    *,
    cache: ExperimentCache,
    dataset: str,
    retriever: str,
    unit_id: str,
    expected_answer: str,
    answer_type: str,
    response: str,
) -> Dict[str, Any]:
    judgement, origin = cache.find_sigma_judgement(
        dataset,
        retriever,
        unit_id,
        expected_answer,
        response,
    )
    if judgement is not None:
        return {**judgement, "cache_origin": origin}

    print(
        f"[SIGMA verifier cache] {origin}: "
        f"{dataset}/{retriever}/{unit_id}; querying DeepSeek",
        flush=True,
    )
    if normalize_text(response).casefold() == normalize_text(expected_answer).casefold():
        raw_response = ""
        matched = True
        status = "local_exact_match"
    else:
        prompt = sigma_judge_prompt(expected_answer, answer_type, response)
        raw_response = ""
        parsed = None
        for _ in range(3):
            raw_response = cache.query_target(prompt, max_tokens=4096)
            parsed = _json_object_from_text(raw_response)
            if parsed is not None:
                break
        matched = bool(parsed.get("matched", False)) if parsed else False
        status = "parsed" if parsed else "parse_failed"

    judgement = cache.store_sigma_judgement(
        dataset=dataset,
        retriever=retriever,
        unit_id=unit_id,
        expected_answer=expected_answer,
        response=response,
        raw_judge_response=raw_response,
        matched=matched,
        status=status,
    )
    return {**judgement, "cache_origin": "live"}


def evaluate_rq1_configuration(
    *,
    method: str,
    dataset: str,
    retriever: str,
    prepared: Dict[str, Dict[str, Any]],
    cache: ExperimentCache,
    repository: DatasetRepository,
    retrieval: RetrievalManager,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = copy.deepcopy(prepared)
    records: List[Dict[str, Any]] = []
    if method == "dcmi":
        for doc_id, doc in data.items():
            outputs = {}
            for label in ("original", "perturbed"):
                outputs[label] = _evaluate_query(
                    cache=cache,
                    repository=repository,
                    retrieval=retrieval,
                    target_docs=data,
                    method=method,
                    dataset=dataset,
                    retriever=retriever,
                    unit_id=f"{doc_id}::{label}",
                    query=doc[f"{label}_query"],
                    answer_mode="yes_no",
                )
            original_label = normalize_yes_no(outputs["original"]["response"])
            perturbed_label = normalize_yes_no(outputs["perturbed"]["response"])
            predicted = "yes" if original_label == "yes" and perturbed_label == "no" else "no"
            actual = actual_membership(doc)
            doc["evaluation"] = {
                **outputs,
                "original_answer_label": original_label,
                "perturbed_answer_label": perturbed_label,
                "predicted_mem": predicted,
            }
            records.append({
                "doc_id": doc_id,
                "actual_mem": actual,
                "predicted_mem": predicted,
                "is_correct": predicted == actual,
                "membership_score": 1.0 if predicted == "yes" else 0.0,
            })
        return data, records

    if method == "rag_mia":
        for doc_id, doc in data.items():
            output = _evaluate_query(
                cache=cache,
                repository=repository,
                retrieval=retrieval,
                target_docs=data,
                method=method,
                dataset=dataset,
                retriever=retriever,
                unit_id=f"{doc_id}::0",
                query=doc["question"],
                answer_mode="yes_no",
            )
            predicted = normalize_yes_no(output["response"])
            if predicted not in {"yes", "no"}:
                predicted = "no"
            actual = actual_membership(doc)
            doc["evaluation"] = {**output, "predicted_mem": predicted}
            records.append({
                "doc_id": doc_id,
                "actual_mem": actual,
                "predicted_mem": predicted,
                "is_correct": predicted == actual,
                "membership_score": 1.0 if predicted == "yes" else 0.0,
            })
        return data, records

    if method == "sigma":
        for doc_id, doc in data.items():
            valid_items = []
            correct_run_count = 0
            for fallback_index, item in enumerate(doc.get("open_qa_items", [])):
                question_id = item.get("question_id", fallback_index)
                valid = (
                    str(item.get("generation_status", "")).strip() == "ok"
                    and str(item.get("target_anchor_status", "")).strip() == "ok"
                    and str(item.get("target_anchor_audit_status", "")).strip() in {"", "ok"}
                )
                if not valid:
                    continue
                for run_name in SIGMA_RUNS:
                    run = item.get("runs", {}).get(run_name, {})
                    unit_id = f"{doc_id}::{question_id}::{run_name}"
                    output = _evaluate_query(
                        cache=cache,
                        repository=repository,
                        retrieval=retrieval,
                        target_docs=data,
                        method=method,
                        dataset=dataset,
                        retriever=retriever,
                        unit_id=unit_id,
                        query=run["composed_query"],
                        answer_mode="open",
                    )
                    judgement = _sigma_judgement(
                        cache=cache,
                        dataset=dataset,
                        retriever=retriever,
                        unit_id=unit_id,
                        expected_answer=normalize_text(item.get("answer", "")),
                        answer_type=normalize_text(item.get("answer_type", "")) or "term",
                        response=output["response"],
                    )
                    answered = bool(judgement.get("matched", False))
                    run.update(output)
                    run["answered_correctly"] = answered
                    run["judge_status"] = judgement.get("status", "cached")
                    correct_run_count += int(answered)
                if all(name in item.get("runs", {}) for name in SIGMA_RUNS):
                    valid_items.append(item)
            predicted = "yes" if valid_items and all(
                any(bool(item["runs"][name].get("answered_correctly")) for name in SIGMA_RUNS)
                for item in valid_items
            ) else "no"
            actual = actual_membership(doc)
            doc["predicted_mem"] = predicted
            records.append({
                "doc_id": doc_id,
                "actual_mem": actual,
                "predicted_mem": predicted,
                "is_correct": predicted == actual,
                "membership_score": float(correct_run_count),
            })
        return data, records
    raise ValueError(f"Unsupported method: {method}")


def member_documents(data: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    return sorted(
        ((str(doc_id), doc) for doc_id, doc in data.items() if actual_membership(doc) == "yes"),
        key=lambda pair: pair[0],
    )
