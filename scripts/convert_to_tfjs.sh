#!/usr/bin/env bash
# Convert build/web_saved_model (from build_web_model.py) into sharded TF.js graph models.
#
# Produces two candidates under build/tfjs/ (float16 and uint8 weight quantisation). The parity test
# (make_reference_predictions.py + web/tests/parity.node.test.ts) decides which one ships; copy it with
#   scripts/convert_to_tfjs.sh --install uint8     (or float16)
#
# Usage: scripts/convert_to_tfjs.sh [--install float16|uint8]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-tfjs"
SAVED_MODEL="$ROOT/build/web_saved_model"
OUT="$ROOT/build/tfjs"
WEB_MODEL="$ROOT/web/public/model"
SHARD_BYTES=4194304   # 4 MiB shards

if [[ "${1:-}" == "--install" ]]; then
  variant="${2:?usage: --install float16|uint8}"
  [[ -f "$OUT/$variant/model.json" ]] || { echo "No converted $variant model in $OUT" >&2; exit 1; }
  rm -rf "$WEB_MODEL" && mkdir -p "$WEB_MODEL"
  cp "$OUT/$variant"/* "$WEB_MODEL"/
  echo "Installed $variant model into $WEB_MODEL:"
  du -sh "$WEB_MODEL"
  exit 0
fi

if [[ -x "$VENV/Scripts/python.exe" ]]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi
if [[ ! -x "$PY" ]]; then
  echo "Creating isolated converter venv at $VENV"
  python3.11 -m venv "$VENV" 2>/dev/null || py -3.11 -m venv "$VENV"
  if [[ -x "$VENV/Scripts/python.exe" ]]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r "$ROOT/scripts/requirements-convert.txt"
  "$PY" -m pip install -q --no-deps tensorflowjs==4.22.0
fi

[[ -f "$SAVED_MODEL/saved_model.pb" ]] || { echo "Run scripts/build_web_model.py first" >&2; exit 1; }

export PYTHONPATH="$ROOT/scripts/tfjs_shims${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=2

convert() {  # $1 = variant name, remaining args = quantisation flags
  local name="$1"; shift
  rm -rf "${OUT:?}/$name"
  "$PY" -m tensorflowjs.converters.converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    "$@" \
    --weight_shard_size_bytes="$SHARD_BYTES" \
    "$SAVED_MODEL" "$OUT/$name"
  "$PY" "$ROOT/scripts/fix_tfjs_bias.py" "$OUT/$name"
  echo "== $name: $(du -sh "$OUT/$name" | cut -f1), $(ls "$OUT/$name" | wc -l) files"
}

convert float16 --quantize_float16="*"
convert uint8 --quantize_uint8="*"
