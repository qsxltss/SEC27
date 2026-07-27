#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BGE_BATCH_SIZE="${AE_BGE_BATCH_SIZE:-16}"
exec "${PYTHON_BIN}" "${ROOT}/code/build_indexes.py" \
  --batch-size "${BGE_BATCH_SIZE}" \
  "$@"
