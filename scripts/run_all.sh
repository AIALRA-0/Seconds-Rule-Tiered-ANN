#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

# 0) create venv if missing
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

# 1) activate
source .venv/bin/activate

# 2) install deps
pip install -U pip
pip install -r requirements.txt

# 3) dataset check (only if sift1m)
python - <<'PY'
import yaml
from pathlib import Path

cfg = yaml.safe_load(open("configs/default.yaml", "r"))
if cfg["data"]["dataset"] == "sift1m":
    d = Path(cfg["data"]["sift1m_dir"])
    base = d / "sift_base.fvecs"
    query = d / "sift_query.fvecs"
    if not base.exists() or not query.exists():
        print("[run_all] SIFT1M files missing. Run:")
        print("  bash scripts/get_sift1m.sh", d)
        raise SystemExit(1)
print("[run_all] dataset check OK")
PY

# 4) run tests (recommended)
export PYTHONPATH="$(pwd)"
pytest -q

# 5) run everything
python -m src.run_all --config "${CONFIG}"

echo "[run_all] Done."
