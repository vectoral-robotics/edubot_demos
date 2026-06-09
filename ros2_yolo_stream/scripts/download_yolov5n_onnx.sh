#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/workspace/models}
MODEL_FILE=${MODEL_FILE:-${MODEL_DIR}/yolov5n.onnx}
MODEL_URL=${MODEL_URL:-https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx}

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_FILE}" ]]; then
  echo "Model already exists: ${MODEL_FILE}"
  exit 0
fi

tmp_file="${MODEL_FILE}.tmp"
rm -f "${tmp_file}"

wget -O "${tmp_file}" "${MODEL_URL}"
mv "${tmp_file}" "${MODEL_FILE}"

echo "Downloaded ${MODEL_FILE}"
