from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def artifact_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cache_root() -> Path:
    return artifact_root() / "cache"


def data_root() -> Path:
    return artifact_root() / "data"


def index_root() -> Path:
    configured = os.getenv("AE_INDEX_ROOT")
    return Path(configured).expanduser().resolve() if configured else data_root() / "datasets"


def output_root() -> Path:
    configured = os.getenv("AE_OUTPUT_ROOT")
    return Path(configured).expanduser().resolve() if configured else artifact_root() / "output"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_yes_no(value: Any) -> str:
    match = re.search(r"\b(yes|no)\b", normalize_text(value), re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


def actual_membership(doc: Any) -> str:
    if not isinstance(doc, dict):
        return "no"
    return "yes" if str(doc.get("mem", "")).strip().lower() == "yes" else "no"


def score_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    correct_count = sum(bool(record["is_correct"]) for record in records)
    document_count = len(records)
    return {
        "document_count": document_count,
        "correct_count": correct_count,
        "accuracy": correct_count / document_count if document_count else 0.0,
    }
