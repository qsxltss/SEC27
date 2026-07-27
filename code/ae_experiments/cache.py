from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .common import canonical_digest, load_json, normalize_text
from .deepseek import DeepSeekTargetClient, MODEL_NAME


class CacheMiss(RuntimeError):
    pass


class ExperimentCache:
    """External-response cache with immutable release and runtime layers."""

    def __init__(self, root: Path, target_client: Any = None):
        self.root = root
        self.dcmi_generation = load_json(root / "deepseek" / "dcmi_generation.json")
        self.sigma_generation = load_json(root / "deepseek" / "sigma_generation.json")
        self.rewrites = load_json(root / "deepseek" / "query_rewrites.json")
        self._target: Dict[str, Any] = {}
        self._runtime_target: Dict[str, Any] = {}
        self._judgements: Dict[str, Any] = {}
        self._runtime_judgements: Dict[str, Any] = {}
        self._lakera: Dict[str, Any] = {}
        self._target_client = target_client

    @staticmethod
    def _validated(record: Dict[str, Any], request: Any, label: str) -> Dict[str, Any]:
        actual = canonical_digest(request)
        expected = str(record.get("request_sha256", ""))
        if actual != expected:
            raise CacheMiss(
                f"Cache request mismatch for {label}: expected {expected}, generated {actual}. "
                "The code, input data, or retrieval result differs from the released experiment."
            )
        return record

    def get_dcmi_perturbation(self, dataset: str, doc_id: str, prompt: str) -> str:
        try:
            record = self.dcmi_generation[dataset][doc_id]
        except KeyError as error:
            raise CacheMiss(f"Missing DCMI generation cache: {dataset}/{doc_id}") from error
        self._validated(record, {"prompt": prompt}, f"DCMI generation {dataset}/{doc_id}")
        return str(record["response"])

    def get_sigma_items(self, dataset: str, doc_id: str, document: str) -> list:
        try:
            record = self.sigma_generation[dataset][doc_id]
        except KeyError as error:
            raise CacheMiss(f"Missing SIGMA generation cache: {dataset}/{doc_id}") from error
        self._validated(
            record,
            {"document": document, "pipeline": "open_qa_original_question"},
            f"SIGMA generation {dataset}/{doc_id}",
        )
        return record["response"]

    def get_rewrite(self, method: str, dataset: str, unit_id: str, request: Dict[str, Any]) -> str:
        try:
            record = self.rewrites[method][dataset][unit_id]
        except KeyError as error:
            raise CacheMiss(f"Missing query-rewrite cache: {method}/{dataset}/{unit_id}") from error
        self._validated(record, request, f"query rewrite {method}/{dataset}/{unit_id}")
        return str(record["response"])

    def _load_target(self, method: str, dataset: str, retriever: str) -> Dict[str, Any]:
        key = f"{method}/{dataset}/{retriever}"
        if key not in self._target:
            path = self.root / "deepseek" / "target_responses" / method / f"{dataset}_{retriever}.json"
            self._target[key] = load_json(path) if path.exists() else {}
        return self._target[key]

    def _runtime_target_path(self, method: str, dataset: str, retriever: str) -> Path:
        return (
            self.root
            / "deepseek"
            / "runtime_target_responses"
            / method
            / f"{dataset}_{retriever}.json"
        )

    def _load_runtime_target(
        self,
        method: str,
        dataset: str,
        retriever: str,
    ) -> Dict[str, Any]:
        key = f"{method}/{dataset}/{retriever}"
        if key not in self._runtime_target:
            path = self._runtime_target_path(method, dataset, retriever)
            self._runtime_target[key] = load_json(path) if path.exists() else {}
        return self._runtime_target[key]

    @staticmethod
    def target_request(
        retrieved_doc_ids: list,
        rag_prompt: str,
        no_doc_prompt: str,
    ) -> Dict[str, Any]:
        return {
            "retrieved_doc_ids": retrieved_doc_ids,
            "rag_prompt": rag_prompt,
            "no_doc_prompt": no_doc_prompt,
        }

    def find_target_response(
        self,
        method: str,
        dataset: str,
        retriever: str,
        unit_id: str,
        retrieved_doc_ids: list,
        rag_prompt: str,
        no_doc_prompt: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        request = self.target_request(retrieved_doc_ids, rag_prompt, no_doc_prompt)
        digest = canonical_digest(request)

        released = self._load_target(method, dataset, retriever).get(unit_id)
        if released and str(released.get("request_sha256", "")) == digest:
            return released, "released"

        runtime = self._load_runtime_target(method, dataset, retriever).get(unit_id)
        if runtime and str(runtime.get("request_sha256", "")) == digest:
            return runtime, "runtime"

        return None, "mismatch" if released else "missing"

    def query_target(self, prompt: str, max_tokens: int) -> str:
        if self._target_client is None:
            self._target_client = DeepSeekTargetClient()
        return str(self._target_client.query(prompt, max_tokens=max_tokens)).strip()

    @staticmethod
    def _atomic_write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=path.parent,
                suffix=".tmp",
            ) as handle:
                json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                temporary_name = handle.name
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def store_target_response(
        self,
        method: str,
        dataset: str,
        retriever: str,
        unit_id: str,
        retrieved_doc_ids: list,
        rag_prompt: str,
        no_doc_prompt: str,
        rag_response: str,
        no_doc_response: str,
        response_source: str,
    ) -> Dict[str, Any]:
        request = self.target_request(retrieved_doc_ids, rag_prompt, no_doc_prompt)
        digest = canonical_digest(request)
        record = {
            "rag_response": rag_response,
            "no_doc_response": no_doc_response,
            "response_source": response_source,
            "request_sha256": digest,
            "provenance": "live DeepSeek cache-through fallback",
            "model": MODEL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        records = self._load_runtime_target(method, dataset, retriever)
        records[unit_id] = record
        self._atomic_write_json(
            self._runtime_target_path(method, dataset, retriever),
            records,
        )
        return record

    def _load_judgements(self, dataset: str, retriever: str) -> Dict[str, Any]:
        key = f"{dataset}/{retriever}"
        if key not in self._judgements:
            path = self.root / "deepseek" / "sigma_judgements" / f"{dataset}_{retriever}.json"
            self._judgements[key] = load_json(path) if path.exists() else {}
        return self._judgements[key]

    def _runtime_judgement_path(self, dataset: str, retriever: str) -> Path:
        return (
            self.root
            / "deepseek"
            / "runtime_sigma_judgements"
            / f"{dataset}_{retriever}.json"
        )

    def _load_runtime_judgements(self, dataset: str, retriever: str) -> Dict[str, Any]:
        key = f"{dataset}/{retriever}"
        if key not in self._runtime_judgements:
            path = self._runtime_judgement_path(dataset, retriever)
            self._runtime_judgements[key] = load_json(path) if path.exists() else {}
        return self._runtime_judgements[key]

    @staticmethod
    def sigma_judgement_request(expected_answer: str, response: str) -> Dict[str, str]:
        return {
            "expected_answer": expected_answer,
            "response": response,
        }

    def find_sigma_judgement(
        self,
        dataset: str,
        retriever: str,
        unit_id: str,
        expected_answer: str,
        response: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        request = self.sigma_judgement_request(expected_answer, response)
        digest = canonical_digest(request)

        released = self._load_judgements(dataset, retriever).get(unit_id)
        if released and str(released.get("request_sha256", "")) == digest:
            return released, "released"

        runtime = self._load_runtime_judgements(dataset, retriever).get(unit_id)
        if runtime and str(runtime.get("request_sha256", "")) == digest:
            return runtime, "runtime"

        return None, "mismatch" if released else "missing"

    def store_sigma_judgement(
        self,
        dataset: str,
        retriever: str,
        unit_id: str,
        expected_answer: str,
        response: str,
        raw_judge_response: str,
        matched: bool,
        status: str,
    ) -> Dict[str, Any]:
        request = self.sigma_judgement_request(expected_answer, response)
        digest = canonical_digest(request)
        record = {
            "matched": bool(matched),
            "response": raw_judge_response,
            "status": status,
            "request_sha256": digest,
            "provenance": "live DeepSeek cache-through fallback",
            "model": MODEL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        records = self._load_runtime_judgements(dataset, retriever)
        records[unit_id] = record
        self._atomic_write_json(
            self._runtime_judgement_path(dataset, retriever),
            records,
        )
        return record

    def get_lakera(self, method: str, query: str) -> Dict[str, Any]:
        if method not in self._lakera:
            self._lakera[method] = load_json(self.root / "lakera" / f"{method}.json")
        normalized_query = normalize_text(query)
        key = canonical_digest({"query": normalized_query})
        try:
            record = self._lakera[method][key]
        except KeyError as error:
            raise CacheMiss(f"Missing Lakera response for {method}, query digest {key}") from error
        return self._validated(record, {"query": normalized_query}, f"Lakera {method}/{key}")
