#!/usr/bin/env bash
# TokAN inference convenience wrapper.
# Usage: ./run.sh <source.wav> [reference.wav] [output.wav] [extra infer_wav.py flags...]
#   reference.wav defaults to the source itself (no accent/voice conversion)
#   output.wav   defaults to outputs/<source-basename>_normalized.wav
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
SRC="$1"; shift
REF="${1:-$SRC}"; [ $# -gt 0 ] && shift
OUT="${1:-outputs/$(basename "${SRC%.*}")_normalized.wav}"; [ $# -gt 0 ] && shift
python infer_wav.py --source_wav "$SRC" --reference_wav "$REF" --output_wav "$OUT" --device cuda "$@"
