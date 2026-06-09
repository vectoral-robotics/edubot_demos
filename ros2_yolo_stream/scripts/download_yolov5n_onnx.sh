#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/workspace/models}
MODEL_FILE=${MODEL_FILE:-${MODEL_DIR}/yolov5n_fp32.onnx}
MODEL_URL=${MODEL_URL:-https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_FILE}" ]]; then
  echo "Model already exists: ${MODEL_FILE}"
  exit 0
fi

download_file="${MODEL_DIR}/yolov5n_download.onnx"
tmp_file="${download_file}.tmp"
rm -f "${tmp_file}"

wget -O "${tmp_file}" "${MODEL_URL}"
mv "${tmp_file}" "${download_file}"

python3 "${SCRIPT_DIR}/onnx_fp16_to_fp32.py" "${download_file}" "${MODEL_FILE}"

echo "Downloaded ${MODEL_FILE}"
