from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class DatasetRepository:
    def __init__(self, datasets_root: Path):
        self.datasets_root = datasets_root
        self._corpora: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def corpus(self, dataset: str) -> Dict[str, Dict[str, Any]]:
        if dataset in self._corpora:
            return self._corpora[dataset]
        path = self.datasets_root / dataset / "corpus.jsonl"
        corpus: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                doc_id = str(record.get("_id", record.get("id", "")))
                if doc_id:
                    corpus[doc_id] = record
        self._corpora[dataset] = corpus
        return corpus

    def contexts(
        self,
        dataset: str,
        retrieved_doc_ids: List[str],
        target_docs: Dict[str, Dict[str, Any]],
        max_chars: int = 2048,
    ) -> List[str]:
        corpus = self.corpus(dataset)
        nonmembers = {
            str(doc_id)
            for doc_id, doc in target_docs.items()
            if str(doc.get("mem", "")).strip().lower() == "no"
        }
        return [
            str(corpus[doc_id].get("text", ""))[:max_chars]
            for doc_id in retrieved_doc_ids
            if doc_id in corpus and doc_id not in nonmembers
        ]


class BM25Retriever:
    def __init__(self, dataset: str, indexes_root: Path):
        index_dir = indexes_root / dataset / "bm25" / "indexes"
        with (index_dir / "doc_ids.pkl").open("rb") as handle:
            self.doc_ids = pickle.load(handle)
        self.index = None
        self.fast = (index_dir / "fast_metadata.json").exists()
        if self.fast:
            with (index_dir / "fast_metadata.json").open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            with (index_dir / "fast_vocab.pkl").open("rb") as handle:
                self.vocabulary = pickle.load(handle)
            self.offsets = np.load(index_dir / "fast_offsets.npy", mmap_mode="r")
            self.doc_indices = np.load(index_dir / "fast_doc_indices.npy", mmap_mode="r")
            self.term_frequencies = np.load(
                index_dir / "fast_term_frequencies.npy", mmap_mode="r"
            )
            self.doc_lengths = np.load(index_dir / "fast_doc_lengths.npy", mmap_mode="r")
            self.idf = np.load(index_dir / "fast_idf.npy", mmap_mode="r")
            self.average_document_length = float(metadata["average_document_length"])
            self.k1 = float(metadata["k1"])
            self.b = float(metadata["b"])
        else:
            with (index_dir / "bm25_index.pkl").open("rb") as handle:
                self.index = pickle.load(handle)

    def search(self, query: str, k: int) -> List[str]:
        tokens = query.lower().split()
        if not self.fast:
            scores = self.index.get_scores(tokens)
            indices = np.argsort(-scores, kind="stable")[:k]
            return [str(self.doc_ids[index]) for index in indices]

        scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        for token in tokens:
            term_index = self.vocabulary.get(token)
            if term_index is None:
                continue
            start = int(self.offsets[term_index])
            end = int(self.offsets[term_index + 1])
            docs = self.doc_indices[start:end]
            frequencies = self.term_frequencies[start:end]
            denominator = frequencies + self.k1 * (
                1.0 - self.b
                + self.b * self.doc_lengths[docs] / self.average_document_length
            )
            scores[docs] += self.idf[term_index] * (
                frequencies * (self.k1 + 1.0) / denominator
            )
        indices = np.argsort(-scores, kind="stable")[:k]
        return [str(self.doc_ids[index]) for index in indices]


class BGERetriever:
    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self, dataset: str, indexes_root: Path):
        try:
            import torch
            import faiss
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "BGE retrieval requires faiss, torch, and transformers. See code/requirements.txt."
            ) from error

        self.faiss = faiss
        self.torch = torch
        requested_device = os.getenv("AE_BGE_DEVICE", "cuda")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device).eval()
        index_dir = indexes_root / dataset / "bge" / "indexes" / f"{dataset}-index"
        self.index = faiss.read_index(str(index_dir / "corpus_index.faiss"))
        with (index_dir / "doc_ids.pkl").open("rb") as handle:
            self.doc_ids = pickle.load(handle)

    def search(self, query: str, k: int) -> List[str]:
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            embedding = self.torch.nn.functional.normalize(
                outputs.last_hidden_state[:, 0], p=2, dim=1
            )
        _, indices = self.index.search(embedding.cpu().numpy(), k)
        return [str(self.doc_ids[index]) for index in indices[0]]


class RetrievalManager:
    def __init__(self, datasets_root: Path, indexes_root: Path | None = None):
        self.datasets_root = datasets_root
        self.indexes_root = indexes_root or datasets_root
        self._retrievers: Dict[str, Any] = {}

    def search(self, dataset: str, retriever: str, query: str, k: int = 3) -> List[str]:
        key = f"{dataset}/{retriever}"
        if key not in self._retrievers:
            if retriever == "bm25":
                instance = BM25Retriever(dataset, self.indexes_root)
            elif retriever == "bge":
                instance = BGERetriever(dataset, self.indexes_root)
            else:
                raise ValueError(f"Unsupported retriever: {retriever}")
            self._retrievers[key] = instance
        return self._retrievers[key].search(query, k)
