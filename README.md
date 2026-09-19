# Survival Analysis of Underpricing in Indian SME IPOs (NSE Emerge 2013–2024)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ry-tZ/sme-ipo-survival-analysis/blob/master/notebooks/01_colab_reproducible_pipeline.ipynb)

Empirical econometric and machine-learning survival analysis investigating the duration of post-listing underpricing for **436 SME IPOs** listed on the **National Stock Exchange (NSE Emerge)** from 2013 to 2024.

---

## 📌 Executive Summary

Traditional IPO literature measures initial returns on day one. This research investigates the **temporal dynamics of underpricing persistence**:
* **Sample Size**: 577 total SME IPOs, of which **436 underpriced issues** form the primary survival cohort.
* **Granularity**: 330,631 daily post-listing trading records.
* **Econometric Models**:
  1. **Non-parametric**: Kaplan-Meier Estimator and Stratified Log-Rank tests (Hot vs. Cold issue cycles).
  2. **Semi-parametric**: Penalized Elastic-Net / Lasso Cox Proportional Hazards regression with Schoenfeld residual testing.
  3. **Non-linear Machine Learning**: Random Survival Forest (RSF) with 5-fold cross-validated Harrell's Concordance Index ($C$-index).

---

## 🏗️ Repository Architecture

```text
sme-ipo-survival-analysis/
├── data/
│   ├── raw/                        # Untouched Prowess extracts (gitignored)
│   └── processed/
│       └── sme_survival_data.csv   # Clean compiled 436-firm survival matrix
├── notebooks/
│   └── 01_colab_reproducible_pipeline.ipynb # One-click Google Colab runner
├── outputs/
│   ├── figures/                    # Publication-quality plots (KM, Cox, RSF)
│   └── tables/                     # LaTeX/CSV summary tables
├── src/
│   ├── config.py                   # Central paths and feature configurations
│   ├── data/
│   │   ├── make_dataset.js         # Fast streaming ingestion from raw sources
│   │   └── loader.py               # Clean Python data loader & feature engineering
│   ├── models/
│   │   ├── km_estimator.py         # Kaplan-Meier modeling & Log-rank tests
│   │   ├── cox_ph.py               # Penalized Cox regression & Hazard plots
│   │   └── random_survival_forest.py # RSF ensemble & CV C-index
│   └── pipeline.py                 # End-to-end batch execution runner
├── pyproject.toml / requirements.txt# Pinned dependency environment
└── README.md                       # Documentation
```

---

## 🚀 Quickstart

### Option A: 1-Click Run via Google Colab
Click the badge at the top or open `notebooks/01_colab_reproducible_pipeline.ipynb` in Colab to run without local setup.

### Option B: Local Execution

```bash
# 1. Clone repository
git clone https://github.com/Ry-tZ/sme-ipo-survival-analysis.git
cd sme-ipo-survival-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Compile data (if raw sources present)
node src/data/make_dataset.js

# 4. Run end-to-end econometric pipeline
python src/pipeline.py
```

---

## 📊 Empirical Data & Variable Definitions

| Variable | Type | Definition |
| :--- | :--- | :--- |
| **`time`** | Duration | Number of trading days from listing until closing price drops below or equal to issue price. |
| **`event`** | Binary | 1 if underpricing terminated during observation period; 0 if right-censored. |
| **`listing_gain_pct`** | Covariate | First-day listing percentage gain over offer price. |
| **`log_traded_qty`** | Covariate | Log-transformed Day 1 trading volume. |
| **`eps`** | Financial | Earnings Per Share on listing date. |
| **`pe_ratio_clipped`** | Financial | Price-to-Earnings ratio. |
| **`debt_to_asset_ratio`**| Financial | Total liabilities divided by total assets. |
| **`is_hot_period`** | Macro | Indicator for 2023–2024 SME boom period. |

---

## 👨‍🔬 Authors & Citation

* **Rajat Sambare** (BITS Pilani, Hyderabad Campus) — *ID: 2022A2PS1727H*
* **Supervisor**: Prof. Sravani Bharandev
* **Course**: ECON F266 Study Project
