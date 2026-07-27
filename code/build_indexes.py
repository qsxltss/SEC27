#!/usr/bin/env python3
"""Build the released BM25 and BGE indexes from the bundled corpora."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


DATASETS = ("nfcorpus", "scifact", "trec-covid")
RETRIEVERS = ("bm25", "bge")
BGE_MODEL = "BAAI/bge-large-en-v1.5"


def parse_choices(value: str, allowed: Sequence[str], label: str) -> Tuple[str, ...]:
    selected = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    invalid = sorted(set(selected) - set(allowed))
    if not selected or invalid:
        raise argparse.ArgumentTypeError(
            f"invalid {label}: {', '.join(invalid) if invalid else 'empty selection'}"
        )
    return selected


def filtered_documents(dataset_dir: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    selection_path = dataset_dir / "selected_indices.json"
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    excluded = {str(doc_id) for doc_id in selection["non_mem_indices"]}
    if len(excluded) != 1000:
        raise ValueError(f"Expected 1000 excluded non-members in {selection_path}")

    with (dataset_dir / "corpus.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            doc_id = str(record.get("_id", ""))
            if not doc_id:
                raise ValueError(
                    f"Missing _id in {dataset_dir / 'corpus.jsonl'} line {line_number}"
                )
            if doc_id not in excluded:
                yield doc_id, record


def stage_directory(index_root: Path, dataset: str, retriever: str) -> Path:
    parent = index_root / dataset
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".building-{retriever}-", dir=parent))


def install_stage(stage: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"Incomplete index destination already exists: {destination}. "
            "Remove it before rebuilding."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(destination)


def bm25_complete(destination: Path) -> bool:
    required = (
        "doc_ids.pkl",
        "fast_vocab.pkl",
        "fast_offsets.npy",
        "fast_doc_indices.npy",
        "fast_term_frequencies.npy",
        "fast_doc_lengths.npy",
        "fast_idf.npy",
        "fast_metadata.json",
    )
    return all((destination / name).is_file() for name in required)


def build_bm25(dataset_dir: Path, index_root: Path, dataset: str) -> None:
    destination = index_root / dataset / "bm25" / "indexes"
    if bm25_complete(destination):
        print(f"[index] BM25 {dataset}: already complete", flush=True)
        return
    if destination.exists():
        raise FileExistsError(f"Partial BM25 index exists: {destination}")

    print(f"[index] BM25 {dataset}: loading corpus", flush=True)
    documents = list(filtered_documents(dataset_dir))
    doc_ids = [doc_id for doc_id, _ in documents]
    tokenized = [
        str(record.get("text", "")).lower().split()
        for _, record in documents
    ]
    bm25 = BM25Okapi(tokenized)

    vocabulary = {term: index for index, term in enumerate(bm25.idf)}
    counts = np.zeros(len(vocabulary), dtype=np.int64)
    for document in bm25.doc_freqs:
        for term in document:
            counts[vocabulary[term]] += 1

    offsets = np.empty(len(vocabulary) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    doc_indices = np.empty(int(offsets[-1]), dtype=np.int32)
    term_frequencies = np.empty(int(offsets[-1]), dtype=np.int32)
    cursors = offsets[:-1].copy()
    for doc_index, document in enumerate(bm25.doc_freqs):
        for term, frequency in document.items():
            term_index = vocabulary[term]
            position = int(cursors[term_index])
            doc_indices[position] = doc_index
            term_frequencies[position] = int(frequency)
            cursors[term_index] += 1

    idf = np.empty(len(vocabulary), dtype=np.float64)
    for term, term_index in vocabulary.items():
        idf[term_index] = float(bm25.idf[term])

    stage = stage_directory(index_root, dataset, "bm25")
    with (stage / "doc_ids.pkl").open("wb") as handle:
        pickle.dump(doc_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (stage / "fast_vocab.pkl").open("wb") as handle:
        pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(stage / "fast_offsets.npy", offsets, allow_pickle=False)
    np.save(stage / "fast_doc_indices.npy", doc_indices, allow_pickle=False)
    np.save(stage / "fast_term_frequencies.npy", term_frequencies, allow_pickle=False)
    np.save(stage / "fast_doc_lengths.npy", np.asarray(bm25.doc_len), allow_pickle=False)
    np.save(stage / "fast_idf.npy", idf, allow_pickle=False)
    metadata = {
        "format": "rank_bm25_okapi_inverted_v1",
        "corpus_size": int(bm25.corpus_size),
        "average_document_length": float(bm25.avgdl),
        "k1": float(bm25.k1),
        "b": float(bm25.b),
        "epsilon": float(bm25.epsilon),
        "term_count": len(vocabulary),
        "posting_count": int(offsets[-1]),
    }
    with (stage / "fast_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    install_stage(stage, destination)
    print(
        f"[index] BM25 {dataset}: {len(doc_ids)} documents, "
        f"{len(vocabulary)} terms",
        flush=True,
    )


def bge_complete(destination: Path) -> bool:
    return (destination / "corpus_index.faiss").is_file() and (
        destination / "doc_ids.pkl"
    ).is_file()


def batches(
    records: Iterable[Tuple[str, Dict[str, Any]]],
    batch_size: int,
) -> Iterator[List[Tuple[str, Dict[str, Any]]]]:
    batch: List[Tuple[str, Dict[str, Any]]] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_bge(
    dataset_dir: Path,
    index_root: Path,
    dataset: str,
    tokenizer: Any,
    model: Any,
    torch: Any,
    faiss: Any,
    device: str,
    batch_size: int,
) -> None:
    destination = index_root / dataset / "bge" / "indexes" / f"{dataset}-index"
    if bge_complete(destination):
        print(f"[index] BGE {dataset}: already complete", flush=True)
        return
    if destination.exists():
        raise FileExistsError(f"Partial BGE index exists: {destination}")

    stage = stage_directory(index_root, dataset, "bge")
    index = None
    doc_ids: List[str] = []
    count = 0
    print(f"[index] BGE {dataset}: encoding corpus on {device}", flush=True)
    for batch_index, batch in enumerate(
        batches(filtered_documents(dataset_dir), batch_size),
        start=1,
    ):
        texts = [
            f"{record.get('title', '')} {record.get('text', '')}"
            for _, record in batch
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = torch.nn.functional.normalize(
                outputs.last_hidden_state[:, 0], p=2, dim=1
            )
        vectors = np.ascontiguousarray(embeddings.cpu().numpy(), dtype=np.float32)
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        doc_ids.extend(doc_id for doc_id, _ in batch)
        count += len(batch)
        if batch_index == 1 or batch_index % 100 == 0:
            print(f"[index] BGE {dataset}: encoded {count} documents", flush=True)

    if index is None or index.ntotal != len(doc_ids):
        raise RuntimeError(f"Failed to build complete BGE index for {dataset}")
    faiss.write_index(index, str(stage / "corpus_index.faiss"))
    with (stage / "doc_ids.pkl").open("wb") as handle:
        pickle.dump(doc_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    install_stage(stage, destination)
    print(f"[index] BGE {dataset}: {len(doc_ids)} documents", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 and BGE indexes from the released corpora"
    )
    artifact_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=artifact_root / "data")
    parser.add_argument(
        "--index-root",
        type=Path,
        default=None,
        help="Root containing per-dataset index directories; defaults to data/datasets",
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--retrievers", default=",".join(RETRIEVERS))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="BGE passage-encoding batch size; lower it if GPU memory is limited",
    )
    args = parser.parse_args()

    selected_datasets = parse_choices(args.datasets, DATASETS, "datasets")
    selected_retrievers = parse_choices(args.retrievers, RETRIEVERS, "retrievers")
    data_root = args.data_root.expanduser().resolve()
    index_root = (
        args.index_root.expanduser().resolve()
        if args.index_root
        else data_root / "datasets"
    )

    if "bm25" in selected_retrievers:
        for dataset in selected_datasets:
            build_bm25(data_root / "datasets" / dataset, index_root, dataset)

    pending_bge = []
    if "bge" in selected_retrievers:
        for dataset in selected_datasets:
            destination = (
                index_root
                / dataset
                / "bge"
                / "indexes"
                / f"{dataset}-index"
            )
            if bge_complete(destination):
                print(f"[index] BGE {dataset}: already complete", flush=True)
            else:
                pending_bge.append(dataset)

    if pending_bge:
        import torch
        import faiss
        from transformers import AutoModel, AutoTokenizer

        requested_device = os.getenv("AE_BGE_DEVICE", "cuda")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        print(f"[index] loading {BGE_MODEL} on {requested_device}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL)
        model = AutoModel.from_pretrained(BGE_MODEL).to(requested_device).eval()
        for dataset in pending_bge:
            build_bge(
                data_root / "datasets" / dataset,
                index_root,
                dataset,
                tokenizer,
                model,
                torch,
                faiss,
                requested_device,
                args.batch_size,
            )


if __name__ == "__main__":
    main()
