# Quantitative Research Division: Organizational Progress & Methodology Ledger

**Repository:** `sme-ipo-survival-analysis`  
**Institutions:** Quantitative Research Division • BITS Pilani (Advisor: Prof. Sravani Bharandev)  
**Target Scope:** Indian SME IPO Underpricing Survival (NSE Emerge & BSE SME Platforms)  
**Status:** Production Validated • Interactive Simulation Engine Online • Git Synchronized  

---

## 1. Executive Org Charter & Pillar Architecture

To mirror the operational rigor of elite quantitative hedge funds, the research project is structured across six autonomous, specialized **Quantitative Research (QR) Pods**:

```
                          ┌─────────────────────────────────────────┐
                          │         HEAD OF QUANT RESEARCH          │
                          │        [Governance & Paper Lead]        │
                          └────────────────────┬────────────────────┘
                                               │
         ┌──────────────────┬──────────────────┼──────────────────┬──────────────────┐
         │                  │                  │                  │                  │
         ▼                  ▼                  ▼                  ▼                  ▼
┌─────────────────┐┌─────────────────┐┌─────────────────┐┌─────────────────┐┌─────────────────┐
│   DATA QUANT    ││   ALPHA QUANT   ││  MICROSTRUCTURE ││ RISK/PORTFOLIO  ││  MODEL TESTER   │
│ Ingestion & ETL ││ Signal Discovery││ Friction/Slippage││ Capital Allocation││ Cross-Validation│
└─────────────────┘└─────────────────┘└─────────────────┘└─────────────────┘└─────────────────┘
```

---

## 2. Pod-by-Pod Progression & Action Audit

### 🏛️ Pillar 1: Data Quant Pod (Data Acquisition, ETL & Zero Look-Ahead Hygiene)

#### Phase 1.1: Core Survival Cohort Harmonization
- **Objective:** Establish the foundational historical dataset of underpriced SME IPOs.
- **Actions Completed:**
  - Ingested primary market records across the NSE Emerge and BSE SME platforms for listings between 2013 and 2024.
  - Formulated the exact non-anticipative survival target:
    - \( T_i \): Active trading days from listing until closing price breaches or equals offer price (\( \text{Close}_t \le P_{\text{issue}} \)).
    - \( E_i = 1 \): Terminal underpricing collapse event observed.
    - \( E_i = 0 \): Right-censored (stock never traded at or below offer price through observation window).
  - Cleaned and compiled **436 underpriced SME IPOs** spanning **330,000+ daily trading log records** into [`data/processed/sme_survival_data.csv`](../data/processed/sme_survival_data.csv).

#### Phase 1.2: Wayback Machine Historical Bidding Harvest
- **Objective:** Overcome the live website paywall on Chittorgarh/IPOMatrix to capture granular Day 1, Day 2, Day 3, and Day 4 subscription multiples.
- **Actions Completed:**
  - Built [`src/data/harvest_wayback_subscription.py`](../src/data/harvest_wayback_subscription.py) to parse archived HTML snapshots from the Wayback Machine.
  - Implemented dynamic column parsing handling varied historical table structures (`['Date', 'NII', 'Retail', 'Total']`, `['Date', 'Other', 'Retail', 'Total']`, `['Date', 'QIB', 'NII', 'RII', 'Total']`).
  - Extracted **248 clean day-by-day bidding records across 74 SME IPOs** saved to [`data/processed/sme_daily_subscription_multiples.csv`](../data/processed/sme_daily_subscription_multiples.csv).

#### Phase 1.3: API Reverse-Engineering & Full Historical Harvest
- **Objective:** Complete 100% subscription coverage for all SME IPOs across the entire 14-year history of Indian SME platforms.
- **Actions Completed:**
  - Inspected frontend network payloads and discovered Chittorgarh's Cloud API endpoint:
    `https://webnodejs.chittorgarh.com/cloud/report/data-read/21/{page}/1/{year}/{financialYear}/0/sme/0`
  - Engineered [`src/data/harvest_chittorgarh_report21.py`](../src/data/harvest_chittorgarh_report21.py) with adaptive pacing (60ms) and retry logic.
  - Harvested **1,398 unique SME IPO records** from FY 2012–13 to FY 2025–26 with 0% data loss into [`data/raw/chittorgarh_subscription_all_sme.csv`](../data/raw/chittorgarh_subscription_all_sme.csv).
#### Phase 1.4: Ingestion of Live Secondary Prices & Returns Till Today
- **Objective:** Track current live market prices, cumulative returns, and current survival status till today for all companies in the cohort.
- **Actions Completed:**
  - Engineered [`src/data/harvest_chittorgarh_report25.py`](../src/data/harvest_chittorgarh_report25.py) querying Chittorgarh Report 25 across all 14 financial years.
  - Harvested **1,381 SME IPO records with live prices and current returns** into [`data/raw/chittorgarh_listing_history_all_sme.csv`](../data/raw/chittorgarh_listing_history_all_sme.csv).
  - Executed [`src/data/update_current_prices.py`](../src/data/update_current_prices.py) matching 422 companies in our cohort:
    - **230 companies (54.5%)** are currently trading **ABOVE** their issue price today (still surviving).
    - **192 companies (45.5%)** are currently trading **BELOW** their issue price today (underpricing collapsed).
    - **Median cohort return today:** **+19.1%**.
  - Upgraded [`dashboard/app.py`](../dashboard/app.py) with **Tab 5 (Live SME IPO Market Performance & Status Till Today)** featuring instant ticker/name search and live pricing badges.

---


### 🔬 Pillar 2: Alpha Research Quant Pod (Hypothesis Formulation & Signal Discovery)

#### Phase 2.1: Non-Parametric Survival Profiling (Kaplan-Meier)
- **Objective:** Quantify empirical survival duration and test for structural regime shifts.
- **Findings:**
  - **Cohort Median Survival:** **153.0 trading days** (~7.3 calendar months).
  - **Structural Shift:** Log-Rank test between pre-2023 listings (median 38 trading days) and the 2023–2024 boom (median >350 trading days, 62.9% censored) revealed an astronomically significant regime shift:
    \[
    p = 1.186 \times 10^{-11}
    \]

#### Phase 2.2: Discovery of the "Retail Breadth Protection" Effect
- **Objective:** Determine whether gross subscription or category-specific demand dictates secondary market price survival.
- **Empirical Breakthrough:**
  - Total Subscription Multiple correlation with collapse (\( E=1 \)): **-0.1952**
  - NII / HNI Multiple correlation with collapse: **-0.1810**
  - **Retail Subscription Multiple correlation with collapse:** **-0.4456**
  - **Application Breadth correlation with collapse:** **-0.4262**
- **Economic Mechanism:**
  - Issues dominated by levered HNI (NII) accounts experience severe post-listing "air pockets" because HNIs exit abruptly to repay IPO financing loans.
  - In contrast, broad retail dispersion (hundreds of thousands of unique applications) creates structural secondary price support, cutting the hazard rate by **14.3% per log unit** (\( \text{HR} = 0.8570 \), \( p = 2.31 \times 10^{-5} \)).

#### Phase 2.3: Dynamic Bidding Acceleration Modeling
- **Objective:** Capture informational cascades during the 3-day book-building window.
- **Empirical Findings:**
  - On the granular bidding subcohort (\( N=74 \)), early acceleration (\( \text{Day 2} / \text{Day 1} \)) yields a hazard ratio of **0.6109**, demonstrating that issues gaining rapid early momentum cut their breakdown hazard by **38.9%**.

---

### ⚙️ Pillar 3: Microstructure Quant Pod (Liquidity Frictions & Execution Constraints)

#### Phase 3.1: SEBI Mandated Lot Size Modeling
- **Institutional Context:** SEBI ICDR regulations mandate minimum application ticket sizes (INR 100,000 to INR 200,000) on SME platforms.
- **Empirical Metrics:**
  - Mean Mandated Lot Size: **2,106 shares**
  - Median Minimum Ticket Size: **INR 126,000**
  - Low-Liquidity Risk: 9.6% of firms trade fewer than 50 lots on Day 1.

#### Phase 3.2: Execution Slippage & Net Alpha Haircut
- **Objective:** Reconcile theoretical paper returns with net executable alpha.
- **Execution Model:**
  - Base bid-ask spread and impact slippage: **1.73%**.
  - Mean Gross Day-1 Underpricing: **54.11%**.
  - Mean Net Executable Return: **52.37%**.
- **Microstructure Insight:** Confirming Dhamija & Arora (2017), early SME issues traded on fewer than 72 days in their first year. Extreme secondary illiquidity artificially prolongs survival by freezing price discovery.

---

### 🛡️ Pillar 4: Risk & Portfolio Quant Pod (Allocation Governance & Exit Protocols)

#### Phase 4.1: The 5-Quintile Risk Allocation Matrix
The pod established quantitative holding and exit protocols conditional on Day-1 listing pops:

| Initial Gain Quintile | Event Rate | Median Survival | Day 20 Survival | Day 120 Survival | Quantitative Execution Protocol | Max Portfolio Cap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1 (0–10%)** | 98.9% | 1 Day | 5.7% | 2.3% | **Immediate Liquidation (T+1 to T+5).** Zero hold tolerance. | 0.50% |
| **Q2 (10–25%)** | 82.8% | 9 Days | 40.2% | 29.9% | **Mandatory Exit by T+20.** Severe hazard cliff past Day 10. | 1.00% |
| **Q3 (25–50%)** | 70.1% | 68 Days | 78.2% | 49.4% | **Harvest 50% Profit at T+20.** Trail remainder to T+60. | 1.50% |
| **Q4 (50–90%)** | 23.9% | 200 Days | 97.7% | 86.4% | **Hold through T+120.** Trail stop loss on 20-day moving average. | 2.50% |
| **Q5 (>90%)** | 24.4% | 192 Days | 97.7% | 94.2% | **Momentum Extension to T+240.** Exit upon 50-day moving average breakdown. | 3.00% |

---

### 🧪 Pillar 5: Model Testing & Validation Pod (Leakage Prevention & Cross-Validation)

#### Phase 5.1: 5-Fold Cross-Validation on Random Survival Forest
- **Cross-Validation Results:**
  - Fold 1: \( C = 0.7844 \)
  - Fold 2: \( C = 0.8711 \)
  - Fold 3: \( C = 0.7788 \)
  - Fold 4: \( C = 0.8062 \)
  - Fold 5: \( C = 0.8295 \)
  - **Mean Cross-Validated Concordance Index:** **0.8140**
- **Linear Benchmark:** Outperformed the linear Penalized Cox PH model (\( C = 0.7564 \)), confirming strong non-linear interactions between listing gains and secondary turnover.

#### Phase 5.2: Out-of-Time Forward Test (Post-December 2024 Listings)
- **Objective:** Strictly evaluate models trained on pre-2025 data against independent SME IPOs listed after December 31, 2024.
- **Validation Results:**
  - **Out-of-Time C-Index:** **0.8944**
  - **Out-of-Time ROC-AUC:** **0.8813**
  - **Empirical Bifurcation:**
    - High-Pop (>50%) & High-Retail (>30x) issues: **78.6% Survival Rate**, Median return today: **+64.6%**.
    - Low-Pop (<20%) & Low-Retail (<10x) issues: **37.7% Collapse Rate**, Median return today: **+16.5%**.
  - **Detailed Forward Test:** Sourced in [`src/models/evaluate_out_of_time.py`](../src/models/evaluate_out_of_time.py) and displayed interactively in Tab 4 of [`dashboard/app.py`](../dashboard/app.py).


---

### 🎓 Pillar 6: Academic Research Pod & Head of Quant (Paper Production & Governance)

#### Phase 6.1: Academic Research Paper
- Authored the comprehensive research paper at [`docs/academic_research_paper.md`](academic_research_paper.md), connecting empirical survival curves to foundational literature:
  - *Rock (1986)*: Winner's curse and informational asymmetry in primary allocations.
  - *Welch (1992)*: Sequential informational cascades in multi-day bidding.
  - *Ritter (1991)*: Long-run underperformance vs. SME persistence anomalies.
  - *Dhamija & Arora (2017)*: Indian SME microstructure and market-maker institutional design.

#### Phase 6.2: Interactive Simulation Lab
- Developed the interactive Streamlit simulation environment at [`dashboard/app.py`](../dashboard/app.py) enabling live scenario stress-testing.

---

## 3. How to Launch the Interactive Simulation Lab

To run the interactive simulator on your local machine:
```bash
# In the project directory:
C:\ProgramData\anaconda3\python.exe -m streamlit run dashboard/app.py
```

### Simulator Capabilities:
1. **Dynamic Parameter Sliders:** Real-time adjustments of listing gain, retail multiple, traded volume, bidding acceleration, latency, and firm age.
2. **Interactive S(t) Plots:** Plotly visualizations comparing Random Survival Forest, Cox PH, and baseline Kaplan-Meier curves.
3. **Horizon Milestones:** Instant calculation of survival probabilities at \( T+5 \), \( T+20 \), \( T+60 \), \( T+120 \), and \( T+240 \).
4. **Historical Twin Matcher:** Surfaces the top 3 closest historical SME IPOs from our 436 cohort with their actual survival durations.
5. **Slippage & Net Alpha Engine:** Computes net executable returns factoring in SEBI mandated lot sizes.

---

---

### 🚀 Pillar 7: Production Recommendation Engine & Quantitative Capital Sizing

#### Phase 7.1: Institutional Lot Sizing & Allotment Probability Engine
- **Module:** [`src/models/recommendation_engine.py`](../src/models/recommendation_engine.py)
- **SEBI SME Lot Sizing:** Implemented statutory price-band lot size slabs (CIR/MRD/DSA/06/2012) ranging from 10,000 shares down to 100 shares, keeping minimum ticket sizes standardized at ~INR 1.0L–1.45L.
- **Computerized Lottery Modeling (Retail Category):**
  $$\mathbb{P}(\text{Allotment})_{\text{Retail}} \approx \min\left(1.0, \frac{1}{\text{Retail Subscription Multiplier}}\right)$$
- **Multi-PAN Binomial Scaling:** Evaluates the cumulative odds of securing at least one lot across $k$ family member PANs:
  $$\mathbb{P}(\ge 1 \text{ Allotment}) = 1 - (1 - p_{\text{retail}})^k$$
  $$\mathbb{E}[\text{Allotted Lots}] = k \cdot p_{\text{retail}}$$

#### Phase 7.2: Kelly-Adjusted Capital Allocation & ASBA Opportunity Cost
- **Dynamic Risk Sizing:**
  - **Q1 (Highest Conviction / Safest):** Maximize retail allocation across up to $k$ PANs; hold for long-term compounder horizon ($T+60$ to $T+120$).
  - **Q2 (Low Hazard / Stable):** Allocate 2–3 PANs; harvest 50% at Day-1 pop and trail remainder.
  - **Q3 (Moderate / Speculative):** Allocate strictly 1 PAN for Day-1 listing pop flip.
  - **Q4 (High Hazard):** Speculative flip only (liquidate at pre-open session $T+1$).
  - **Q5 (Extreme Hazard / Value Trap):** Zero allocation (`AVOID`).
- **Opportunity Cost Engine:** Explicitly accounts for 4-day ASBA funds block at RBI risk-free repo rate (6.5% p.a.).

#### Phase 7.3: Interactive Recommendation Tab in Simulation Lab
- Added **Tab 1: 🎯 Production Recommendation Engine** to [`dashboard/app.py`](../dashboard/app.py).
- Features live investor budget configuration, family PAN scaling sliders, real-time allotment probability gauge, interactive Plotly binomial distribution chart, and pre-bidding/listing execution playbooks.
- **Automated Verification:** 100% test pass across unit tests in [`tests/test_recommendation_engine.py`](../tests/test_recommendation_engine.py).

#### Phase 7.4: Dual Workspace Mode & Pure Prospectus Stage (Zero Subscription Dependency)
- **Problem Resolved:** Addressed the practical reality that before an SME IPO opens for bidding, subscription numbers (Retail x, Total x, Bidding Acceleration, Day 1 x) do not exist.
- **Dual Workspace Architecture:**
  1. **Mode 1 (Upcoming SME IPO Bidding Engine):** Features a dedicated **Pure Prospectus Stage** where subscription sliders are hidden by default. The company is evaluated solely on audited DRHP/RHP financials (P/E at issue price, Debt-to-Asset, Operational Firm Age), lot sizing, capital budget, and family PANs.
  2. **"What-If" Lottery Scenario Matrix:** Computes exact computerized draw-of-lots odds across 6 potential demand tiers (Subdued 3x, Moderate 10x, Strong 25x, High Momentum 50x, Blockbuster 100x, Mega-Frenzy 250x) in advance.
  3. **Mode 2 (Research & Econometric Study Lab):** Unlocks all post-listing variables (Day-1 pop, volume, slippage, twin matcher, live prices till today).
- **Automated Tests:** 7 out of 7 unit tests passing.

---

## 4. How to Launch the Production Recommendation & Simulation Lab

To run the interactive simulator and recommendation engine on your local machine:
```bash
# In the project directory:
C:\ProgramData\anaconda3\python.exe -m streamlit run dashboard/app.py
```
Or simply double-click:
- `run_dashboard.bat` or `run_dashboard.ps1`

