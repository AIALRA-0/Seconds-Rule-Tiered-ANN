#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-data/sift1m}"

echo "[get_sift1m] Checking dataset directory: ${DIR}"
if [ ! -d "${DIR}" ]; then
  echo "  Directory does not exist. Create it:"
  echo "    mkdir -p ${DIR}"
  echo "  Then place:"
  echo "    sift_base.fvecs"
  echo "    sift_query.fvecs"
  exit 1
fi

if [ ! -f "${DIR}/sift_base.fvecs" ] || [ ! -f "${DIR}/sift_query.fvecs" ]; then
  echo "  Missing files under ${DIR}."
  echo "  Please place:"
  echo "    ${DIR}/sift_base.fvecs"
  echo "    ${DIR}/sift_query.fvecs"
  exit 1
fi

echo "  OK: found sift_base.fvecs and sift_query.fvecs under ${DIR}"
