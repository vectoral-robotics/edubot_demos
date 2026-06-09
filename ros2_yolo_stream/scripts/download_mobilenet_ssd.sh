#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/workspace/models}
MODEL_FILE=${MODEL_FILE:-${MODEL_DIR}/mobilenet_iter_73000.caffemodel}
CONFIG_FILE=${CONFIG_FILE:-${MODEL_DIR}/mobilenet_ssd.prototxt}
MODEL_URL=${MODEL_URL:-https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel}
CONFIG_URL=${CONFIG_URL:-https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt}

mkdir -p "${MODEL_DIR}"

download_if_missing() {
  local url=$1
  local target=$2

  if [[ -f "${target}" ]]; then
    echo "Already exists: ${target}"
    return
  fi

  local tmp_file="${target}.tmp"
  rm -f "${tmp_file}"
  wget -O "${tmp_file}" "${url}"
  mv "${tmp_file}" "${target}"
  echo "Downloaded ${target}"
}

download_if_missing "${CONFIG_URL}" "${CONFIG_FILE}"
download_if_missing "${MODEL_URL}" "${MODEL_FILE}"
