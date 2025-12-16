#!/usr/bin/env bash
set -euo pipefail

# Priority:
#   1. Environment variable SR_CONFIG (used by one_click)
#   2. First command-line argument
#   3. Default configs/default.yaml
CONFIG="${SR_CONFIG:-${1:-configs/default.yaml}}"

echo "[run_all] Using config: ${CONFIG}"

# 0) create venv if missing
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

# 1) activate
source .venv/bin/activate

# 2) install deps (if you find it slow, you can comment out these two lines later)
pip install -U pip
pip install -r requirements.txt

# 3) dataset check (only if sift1m)
python - <<PY
import yaml
from pathlib import Path

config_path = "${CONFIG}"
cfg = yaml.safe_load(open(config_path, "r"))

if cfg["data"]["dataset"] == "sift1m":
    d = Path(cfg["data"]["sift1m_dir"])
    base = d / "sift_base.fvecs"
    query = d / "sift_query.fvecs"
    if not base.exists() or not query.exists():
        print("[run_all] SIFT1M files missing. Run:")
        print("  bash scripts/get_sift1m.sh", d)
        raise SystemExit(1)

print(f"[run_all] dataset check OK (config={config_path})")
PY

# 4) run tests (recommended)
export PYTHONPATH="$(pwd)"
pytest -q

# 5) run everything
python -m src.run_all --config "${CONFIG}"

echo "[run_all] Done."
