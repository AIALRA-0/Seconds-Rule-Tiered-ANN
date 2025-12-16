# Evaluating Caching Policies for IVF-based ANN Engines under Tiered DRAM and SSD Storage

## 1. Overview

Approximate nearest neighbor (ANN) search is a key primitive in modern vector retrieval systems (RAG, semantic search, recommendation, log analysis, etc.). Large-scale vector collections and high dimensionality make full-DRAM indexing expensive, while storing all index data on SSD pushes latency into the millisecond regime and severely hurts tail latency (p95/p99).

This project:

- Uses an IVF ANN engine as the backbone.
- Explicitly models a two-tier storage hierarchy: DRAM (fast/expensive) and SSD (slow/cheap).
- Treats each IVF inverted list (cluster list) as the caching/migration unit.
- Implements multiple list-level caching policies:
  - `all_dram`, `all_ssd`
  - `lru`
  - `naive_lfu`
  - `window_lfu`
  - `seconds_rule`
- Runs a systematic parameter sweep over:
  - DRAM fraction: `dram_fraction ∈ {0.05, 0.1, 0.2}`
  - Number of probed clusters: `nprobe ∈ {1, 2, 4, 8, 16, 32}`
  - SSD max IOPS: `max_iops ∈ {1M, 5M}`
  - Multiple workloads with different access distributions.

The goal is to understand:

1. How much tail latency (p95) can realistically be reduced by smarter caching under limited DRAM.
2. Under which workloads “seconds rule” (reuse-interval based) has an advantage over recency/frequency policies.
3. How much of the apparent policy benefit is eaten up by migration cost.


## 2. Key Contributions

- **Tiered IVF model**

  Abstracts an IVF-based ANN engine into:

  - ANN computation: `L_ann`
  - SSD I/O (page-level): `L_io`
  - A small fixed overhead: `L_fixed`

  With a simple page-latency model:

$$
t_{\text{page}} = 20 + \frac{10^6}{\text{max\_iops}} \ \text{µs}
$$


* **Multiple caching policies at IVF-list granularity**

  * `all_dram` / `all_ssd` as upper/lower bounds
  * `lru`: recency-based
  * `naive_lfu`: pure frequency-based
  * `window_lfu`: sliding-window / decayed frequency
  * `seconds_rule`: reuse-interval-aware policy

* **Multi-workload evaluation**

  * Default trace-like workload
  * Fast-moving hotspot workload
  * Slow-moving hotspot workload
  * Uniform access workload
  * Zipfian access workload (`s = 1.2`)

* **Unified pipeline and assets**

  * Raw per-query logs → aggregated CSVs (`ann_policy_agg.csv`)
  * SLA tables (`sla_reachable_recall.csv`)
  * Standardized plots:

    * p95 vs DRAM
    * IOAmp vs DRAM
    * Migration vs DRAM
    * Recall–latency frontier
    * SLA-reachable recall
  * One-click scripts to reproduce runs and generate report assets.


## 3. Main Findings 

* **Tail latency is dominated by SSD pages read**

  Under the page-level I/O model, p95 latency is essentially:

  ```math
  L(q) \approx L_{\text{ann}}(q) + t_{\text{page}} \cdot N_{\text{pages}}(q)
  ```

  So I/O amplification (IOAmp)—SSD bytes read per query divided by truly useful bytes (top-k vectors)—is the key intermediate metric.

* **For strict SLAs (200–500 µs), caching alone is not enough**

  With realistic SSD page costs, any tail query that touches SSD lists pushes p95 into the millisecond range. Under the current configuration, only the all-DRAM setting can meet a 200–500 µs SLA; all DRAM+SSD policies fail these tight budgets.

* **Policy behavior depends strongly on workload**

  * **Fast-changing hotspots**
    `seconds_rule` and `lru` can cut IOAmp significantly vs LFU-based policies, but at the cost of GB-scale migration.
  * **Slow-changing hotspots & Zipf**
    `window_lfu` / `naive_lfu` achieve almost the same p95 and IOAmp as `seconds_rule`, with 1–2 orders of magnitude less migration.
  * **Uniform / default**
    Hotspots are weak; “chasing hotspots” (LRU/seconds rule) mostly chases noise. `window_lfu` gives a better balance of reduced IOAmp and modest migration.

* **`nprobe` remains the fundamental recall–latency knob**

  Increasing `nprobe` monotonically increases recall, but also linearly increases list scans, SSD pages, IOAmp, and thus p95. In SSD scenarios, `nprobe` must be chosen under an explicit page budget derived from the SLA.

* **Window LFU is a robust default policy**

  Across workloads, `window_lfu` consistently offers:

  * Near-best p95 / IOAmp
  * Much lower migration overhead than `lru` / `seconds_rule`

  making it a safe default for real systems.

---

## 4. Repository Layout

```text
.
├── FINAL_REPORT.md                 # Long-form project report
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── configs/                        # Workload configs
│   ├── default.yaml                # Default workload
│   ├── exp_hotspot_fast.yaml       # Hotspot fast-fast
│   ├── exp_hotspot_slow.yaml       # Hotspot slow-fast
│   ├── exp_uniform.yaml            # Uniform access
│   └── exp_zipf_s1.2.yaml          # Zipf (s = 1.2)
├── data/
│   ├── get_sift1m.sh               # Wrapper script for SIFT download
│   ├── sift1m/                     # SIFT1M base / learn / query / groundtruth
│   └── siftsmall/                  # Small-scale test data
├── scripts/
│   ├── get_sift1m.sh               # Download and unpack SIFT1M
│   ├── run_all.sh                  # Run all workloads
│   ├── one_click.sh                # Run experiments + export results
│   ├── one_click_report.sh         # Stitch master report from results
│   ├── dump_results.py             # Export compact text summaries from agg/
│   ├── export_report_assets.py     # Generate report_assets for each run
│   └── validate_run.py             # Consistency checks for a single run
├── src/
│   ├── ann_engine.py               # IVF search + policy hooks
│   ├── policies.py                 # all_dram, all_ssd, LRU, LFU, seconds rule
│   ├── latency_model.py            # Page-level SSD latency model
│   ├── experiment_runner.py        # Orchestrate experiments from configs/
│   ├── plotting.py                 # Generate figures/*.png from agg/ CSVs
│   ├── run_all.py                  # Python entry for batch runs
│   └── workload.py                 # Workload definitions (default/hotspot/...)
├── results/
│   ├── sr_sift_default_fast_*/             # Default workload runs
│   ├── sr_sift_exp_hotspot_fast_fast_*/    # Hotspot fast-fast
│   ├── sr_sift_exp_hotspot_slow_fast_*/    # Hotspot slow-fast
│   ├── sr_sift_exp_uniform_fast_*/         # Uniform
│   ├── sr_sift_exp_zipf_s1.2_fast_*/       # Zipf s = 1.2
│   ├── report_master_*.md                  # Auto-stitched master reports
│   ├── log_*.txt                           # Run logs
│   ├── export_*.txt                        # Compact summaries per run
│   └── validate_*.txt                      # Validation outputs
└── tests/
    ├── test_ivf_alignment.py               # IVF index construction tests
    ├── test_latency_model.py               # Latency model tests
    └── test_policies.py                    # Policy behavior tests
```

---

## 5. Getting Started

### 5.1 Installation

Prerequisites:

* Python 3.9+ (recommended)
* Basic C/C++ toolchain if any dependency requires native extensions

Install dependencies:

```bash
pip install -r requirements.txt
```

### 5.2 Data

The experiments use SIFT1M (128-D float32 vectors).

Download and prepare:

```bash
bash scripts/get_sift1m.sh
```

After that, you should have:

* `data/sift1m/` — base / learn / query / groundtruth
* `data/siftsmall/` — small-scale sanity-check dataset

### 5.3 Running Experiments

#### One-click pipeline

Run all workloads and export key results:

```bash
bash scripts/one_click.sh
```

This will:

Run experiments for all configs in `configs/`.
Generate per-workload outputs under `results/`:

   * `raw/ann_policy_raw.csv`
   * `agg/ann_policy_agg.csv`
   * `agg/sla_reachable_recall.csv`
   * `figures/*.png`
   * `report_assets/*`

Generate master report:

```bash
bash scripts/one_click_report.sh
```

This will create `results/report_master_*.md` with stitched plots/tables.

#### Running a single workload

Example (exact CLI may vary slightly; check `experiment_runner.py`):

```bash
python -m src.experiment_runner --config configs/default.yaml
```

Or use:

```bash
bash scripts/run_all.sh
```

#### Validating a run

```bash
python scripts/validate_run.py --run_dir results/sr_sift_default_fast_...
```