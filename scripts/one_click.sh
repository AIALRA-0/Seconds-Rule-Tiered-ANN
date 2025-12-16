#!/usr/bin/env bash
set -euo pipefail

# =========================
# Seconds-Rule-Tiered-ANN one-click pipeline
# - create/activate venv
# - install deps
# - smoke tests
# - full tests
# - run experiments (sweeps) + generate figures
# - bundle outputs for report writing
# =========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/results/logs"
FIG_DIR="${ROOT_DIR}/results/figures"
REPORT_DIR="${ROOT_DIR}/report"

mkdir -p "${LOG_DIR}" "${FIG_DIR}" "${REPORT_DIR}"

LOG_FILE="${LOG_DIR}/one_click_${RUN_ID}.log"
FREEZE_FILE="${LOG_DIR}/pip_freeze_${RUN_ID}.txt"
ENV_FILE="${LOG_DIR}/env_${RUN_ID}.txt"

# Redirect all stdout/stderr to both terminal and log file.
exec > >(tee -i "${LOG_FILE}") 2>&1

echo "============================================================"
echo "[one_click] RUN_ID=${RUN_ID}"
echo "[one_click] ROOT_DIR=${ROOT_DIR}"
echo "[one_click] LOG_FILE=${LOG_FILE}"
echo "============================================================"
echo

# -------------------------
# Helpers
# -------------------------
section () {
  echo
  echo "------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] $1"
  echo "------------------------------------------------------------"
}

run_cmd () {
  echo "+ $*"
  "$@"
}

try_cmd () {
  # Usage: try_cmd "description" command...
  local desc="$1"
  shift
  echo "+ (try) ${desc}: $*"
  if "$@"; then
    echo "  -> OK: ${desc}"
    return 0
  else
    echo "  -> FAIL: ${desc}"
    return 1
  fi
}

# -------------------------
# 0) Basic system info
# -------------------------
section "0) System / repo info"
echo "PWD: $(pwd)"
echo "Git status (short):"
git status --porcelain || true
echo
echo "Git HEAD:"
git rev-parse HEAD || true
echo
echo "uname -a:"
uname -a || true
echo

# -------------------------
# 1) Setup / activate venv
# -------------------------
section "1) Python venv + dependencies"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python or install python3."
  exit 1
fi

echo "Using PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" -V

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d ".venv" ]]; then
    run_cmd "${PYTHON_BIN}" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Detected already in a virtualenv: VIRTUAL_ENV=${VIRTUAL_ENV}"
fi

python -V
python -c "import sys; print('Python executable:', sys.executable)"

# Upgrade pip toolchain and install requirements
run_cmd python -m pip install --upgrade pip setuptools wheel

if [[ -f "requirements.txt" ]]; then
  run_cmd python -m pip install -r requirements.txt
else
  echo "ERROR: requirements.txt not found at repo root."
  exit 1
fi

# Ensure test runner exists even if not pinned
run_cmd python -m pip install -U pytest

# Save environment snapshot
{
  echo "RUN_ID=${RUN_ID}"
  echo "DATE=$(date -Is)"
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "PYTHON=$(python -V 2>&1)"
  echo "PYTHON_EXE=$(python -c 'import sys; print(sys.executable)')"
  echo "GIT_HEAD=$(git rev-parse HEAD 2>/dev/null || echo NA)"
} | tee "${ENV_FILE}"

run_cmd python -m pip freeze | tee "${FREEZE_FILE}"

# -------------------------
# 2) Data check (SIFT)
# -------------------------
section "2) Dataset presence check (SIFT1M / SIFTsmall)"

need_files=(
  "data/sift1m/sift_base.fvecs"
  "data/sift1m/sift_query.fvecs"
  "data/siftsmall/siftsmall_base.fvecs"
  "data/siftsmall/siftsmall_query.fvecs"
)

missing=0
for f in "${need_files[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing: ${f}"
    missing=1
  else
    echo "Found:   ${f}"
  fi
done

if [[ "${missing}" -eq 1 ]]; then
  echo
  echo "Some dataset files are missing."
  if [[ -f "scripts/get_sift1m.sh" ]]; then
    echo "Attempting to fetch dataset via scripts/get_sift1m.sh ..."
    run_cmd bash scripts/get_sift1m.sh
  else
    echo "ERROR: scripts/get_sift1m.sh not found, and dataset missing."
    exit 1
  fi
else
  echo "Dataset OK."
fi

# -------------------------
# 3) Minimal smoke tests
# -------------------------
section "3) Smoke tests (fast sanity)"

# (a) Faiss environment
try_cmd "faiss import smoke" python -m src.test_faiss_env

# (b) key unit tests (fast)
# You can tune the subset; keep it quick.
try_cmd "pytest smoke: policies + latency" python -m pytest -q \
  tests/test_policies.py tests/test_latency_model.py

# Optional: IVF alignment is important; keep it in smoke if it's fast.
# If it’s heavy on your machine, comment it out and keep it in full tests only.
try_cmd "pytest smoke: ivf alignment" python -m pytest -q tests/test_ivf_alignment.py

# -------------------------
# 4) Full test suite
# -------------------------
section "4) Full tests (all tests/)"
run_cmd python -m pytest -q

# -------------------------
# 5) Run experiments + generate figures
# -------------------------
section "5) Experiments + figures (full pipeline)"

export SR_CONFIG="${SR_CONFIG:-configs/default.yaml}"
export SR_RESULTS_DIR="${SR_RESULTS_DIR:-results}"

echo "SR_CONFIG=${SR_CONFIG}"
echo "SR_RESULTS_DIR=${SR_RESULTS_DIR}"

# Prefer python pipeline, but if you already have scripts/run_all.sh doing the right thing,
# we try it first (so your own orchestration stays authoritative).
PIPELINE_OK=0

if [[ -f "scripts/run_all.sh" ]]; then
  if try_cmd "pipeline via scripts/run_all.sh" bash scripts/run_all.sh; then
    PIPELINE_OK=1
  fi
fi

if [[ "${PIPELINE_OK}" -eq 0 ]]; then
  # Try common run_all entrypoints with/without --config
  if try_cmd "pipeline via python -m src.run_all --config" python -m src.run_all --config "${SR_CONFIG}"; then
    PIPELINE_OK=1
  elif try_cmd "pipeline via python -m src.run_all" python -m src.run_all; then
    PIPELINE_OK=1
  elif try_cmd "pipeline via python src/run_all.py --config" python src/run_all.py --config "${SR_CONFIG}"; then
    PIPELINE_OK=1
  elif try_cmd "pipeline via python src/run_all.py" python src/run_all.py; then
    PIPELINE_OK=1
  fi
fi

if [[ "${PIPELINE_OK}" -eq 0 ]]; then
  echo "ERROR: Could not find a working pipeline command."
  echo "Tried scripts/run_all.sh and src/run_all.py variants."
  exit 1
fi

# -------------------------
# 5b) Post-processing: compact dump + analysis figures
# -------------------------
section "5b) Post-processing: dump_results + plot_results"

# This step will automatically find the latest tiered_ann_seconds_rule_* directory under results/,
# and print the COMPACT summary you just saw (so you can copy it directly for me)
if [[ -f "scripts/dump_results.py" ]]; then
  run_cmd python scripts/dump_results.py
else
  echo "NOTE: scripts/dump_results.py does not exist, skipping dump."
fi

# This step will read the agg CSV from that run and generate clearer faceted plots,
# which are output by default to the figs/ subdirectory under the corresponding run directory
if [[ -f "scripts/plot_results.py" ]]; then
  run_cmd python scripts/plot_results.py
else
  echo "NOTE: scripts/plot_results.py does not exist, skipping plotting."
fi


# -------------------------
# 6) Verify outputs for report
# -------------------------
section "6) Verify report artifacts (CSVs + figures)"

echo "Listing results directory:"
ls -lah results || true
echo
echo "Listing figures directory:"
ls -lah results/figures || true
echo

# Recursively search for figures and CSVs under the entire results/ directory, no depth limit
NUM_PNG=$(find results -type f \( -name "*.png" -o -name "*.pdf" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l | tr -d ' ')
NUM_CSV=$(find results -type f -name "*.csv" | wc -l | tr -d ' ')

echo "Found figures: ${NUM_PNG}"
echo "Found CSVs:    ${NUM_CSV}"

if [[ "${NUM_PNG}" -lt 1 ]]; then
  echo "WARNING: No figures found under results/. "
  echo "The pipeline has finished running and generated CSVs, but there are currently no figures."
  echo "You can first use the CSVs for tables/analysis, or add the plotting script later (src/plotting.py)."
fi


# -------------------------
# 7) Bundle run outputs for easy report writing
# -------------------------
section "7) Create report bundle (md/json) + draft report template"

run_cmd python - <<'PY'
import json
import os
import platform
from pathlib import Path
from datetime import datetime

root = Path(os.getcwd())
run_id = os.environ.get("RUN_ID", None)

# If RUN_ID env not set here (bash doesn't export by default), infer from log name later.
# We'll still write files with a timestamp suffix to avoid overwriting.
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

results_dir = root / "results"
log_dir = results_dir / "logs"
report_dir = root / "report"
report_dir.mkdir(parents=True, exist_ok=True)

fig_dir = results_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Pick the newest one_click log as primary log
logs = sorted(log_dir.glob("one_click_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
log_file = logs[0] if logs else None

# Search All：results/**/{*.png,*.pdf,*.jpg,*.jpeg}
fig_files = sorted(
    [
        p
        for p in results_dir.rglob("*")
        if p.suffix.lower() in [".png", ".pdf", ".jpg", ".jpeg"]
    ]
)

csv_files = sorted(results_dir.rglob("*.csv"))

bundle = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "paths": {
        "root": str(root),
        "results_dir": str(results_dir),
        "figures_dir": str(fig_dir),
        "log_file": str(log_file) if log_file else None,
    },
    "artifacts": {
        "csv_files": [str(p) for p in csv_files],
        "figure_files": [str(p) for p in fig_files],
    },
}

bundle_json = results_dir / f"report_bundle_{ts}.json"
bundle_md = results_dir / f"report_bundle_{ts}.md"
draft_report = report_dir / f"draft_report_{ts}.md"

bundle_json.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

# Markdown bundle (human-friendly)
md_lines = []
md_lines.append(f"# Report Bundle ({ts})")
md_lines.append("")
md_lines.append("## Run Metadata")
md_lines.append(f"- Timestamp: {bundle['timestamp']}")
md_lines.append(f"- Platform: {bundle['platform']['system']} {bundle['platform']['release']} ({bundle['platform']['machine']})")
md_lines.append(f"- Python: {bundle['platform']['python']}")
md_lines.append(f"- Root: `{bundle['paths']['root']}`")
if log_file:
    md_lines.append(f"- Log: `{log_file}`")
md_lines.append("")
md_lines.append("## CSV Artifacts")
if csv_files:
    for p in csv_files:
        md_lines.append(f"- `{p}`")
else:
    md_lines.append("- (none found)")
md_lines.append("")
md_lines.append("## Figure Artifacts")
if fig_files:
    for p in fig_files:
        md_lines.append(f"- `{p}`")
else:
    md_lines.append("- (none found)")
md_lines.append("")
bundle_md.write_text("\n".join(md_lines), encoding="utf-8")

# Draft report skeleton (you can start writing immediately)
dr = []
dr.append(f"# Seconds-Rule Tiered ANN Report (Draft) — {ts}")
dr.append("")
dr.append("## Abstract")
dr.append("- (Write after results. 4–6 sentences: problem, method, key findings.)")
dr.append("")
dr.append("## 1. Introduction")
dr.append("- Motivation: tiered DRAM + Storage-Next SSD for ANN / vector search")
dr.append("- Research Question (RQ)")
dr.append("- Contributions (bullet list)")
dr.append("")
dr.append("## 2. Background & Related Work")
dr.append("- Five-minute rule → seconds-scale rule")
dr.append("- Hybrid ANN (DiskANN/SPANN etc.) vs. your policy-level focus")
dr.append("")
dr.append("## 3. System Model & Problem Formulation")
dr.append("- Data + ANN engine assumptions (Faiss IVF lists as clusters)")
dr.append("- Tiered storage model (DRAM vs SSD)")
dr.append("- Metrics: recall@k, p50/p95/p99 latency, QPS, I/O amplification, migration overhead")
dr.append("")
dr.append("## 4. Policies / Design")
dr.append("- Baselines: All-DRAM, All-SSD, Naive(LFU), Seconds-rule (+ optional LRU/Decayed-LFU)")
dr.append("- Rebalance mechanism & complexity")
dr.append("- (If modeled) migration cost")
dr.append("")
dr.append("## 5. Methodology")
dr.append("- Datasets: SIFT1M/SIFTsmall (and any synthetic)")
dr.append("- Workloads: stationary hotspot + drifting hotspot")
dr.append("- Sweep axes: DRAM fraction, SSD IOPS/latency, T*, rebalance interval, nprobe")
dr.append("")
dr.append("## 6. Results")
dr.append("### 6.1 Main plots generated by this run")
dr.append("")
if fig_files:
    for p in fig_files:
        rel = p.relative_to(root)
        dr.append(f"- ![]({rel.as_posix()})  ")
else:
    dr.append("- (No figures detected. Check pipeline.)")
dr.append("")
dr.append("### 6.2 Tables")
dr.append("- Add a summary table for representative configs (DRAM%, IOPS, workload) with latency/IO/recall.")
dr.append("")
dr.append("## 7. Analysis & Discussion")
dr.append("- When does seconds-rule win? When does it not? Why?")
dr.append("- Sensitivity to T*, rebalance interval, workload drift speed")
dr.append("- Limits of the model and what would change in a real system")
dr.append("")
dr.append("## 8. Lessons Learned")
dr.append("- What you tried, what worked, what failed, and what you'd do next.")
dr.append("")
dr.append("## 9. Reproducibility")
dr.append("- One-click script: `bash scripts/one_click.sh`")
dr.append("- Outputs: see `results/report_bundle_*.md` and `results/figures/`")
dr.append("")
draft_report.write_text("\n".join(dr), encoding="utf-8")

print(f"[bundle] Wrote: {bundle_json}")
print(f"[bundle] Wrote: {bundle_md}")
print(f"[bundle] Wrote: {draft_report}")
PY

# -------------------------
# 8) Final "copy-paste" block
# -------------------------
section "8) DONE — Copy-paste block for ChatGPT report generation"

NEWEST_BUNDLE_MD="$(ls -1t results/report_bundle_*.md | head -n 1 || true)"
NEWEST_BUNDLE_JSON="$(ls -1t results/report_bundle_*.json | head -n 1 || true)"
NEWEST_DRAFT_REPORT="$(ls -1t report/draft_report_*.md | head -n 1 || true)"

echo "-----BEGIN REPORT INPUT-----"
echo "RUN_ID=${RUN_ID}"
echo "LOG_FILE=${LOG_FILE}"
echo "ENV_FILE=${ENV_FILE}"
echo "FREEZE_FILE=${FREEZE_FILE}"
echo "BUNDLE_MD=${NEWEST_BUNDLE_MD}"
echo "BUNDLE_JSON=${NEWEST_BUNDLE_JSON}"
echo "DRAFT_REPORT=${NEWEST_DRAFT_REPORT}"
echo "FIGURES_DIR=results/figures"
echo "CSVS_UNDER=results"
echo "-----END REPORT INPUT-----"
echo
echo "Next step:"
echo "1) Open ${NEWEST_BUNDLE_MD} and/or paste the block above to me."
echo "2) If you want, also paste the tail of the log: tail -n 200 ${LOG_FILE}"
echo "3) Then I will generate the final report text + tables outline, and you just insert the figures."
echo
echo "All done."
