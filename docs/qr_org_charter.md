# Quantitative Research (QR) Division Charter & Operational Framework
**Initiative:** SME IPO Underpricing Survival Dynamics (NSE Emerge)  
**Lead / Head of QR:** Antigravity (Managing Director)  
**Principal Investigator:** Rajat Sambare (2022A2PS1727H, BITS Pilani)  
**Academic Supervisor:** Prof. Sravani Bharandev  

---

## 1. Functional Org Chart

```
                          ┌──────────────────────────────────────────────┐
                          │          HEAD OF QUANT RESEARCH (MD)         │
                          │        Antigravity & Principal Lead          │
                          └──────────────────────┬───────────────────────┘
                                                 │
        ┌────────────────────────┬───────────────┴───────────────┬────────────────────────┐
        │                        │                               │                        │
┌───────▼──────────────┐ ┌───────▼──────────────┐ ┌──────────────▼──────────────┐ ┌───────▼──────────────┐
│ DATA & ALT DATA      │ │ ALPHA DISCOVERY      │ │ PORTFOLIO CONSTRUCTION      │ │ MARKET MICROSTRUCTURE│
│ QUANTS (L2/L3)       │ │ QUANTS (L3)          │ │ & RISK OPTIMIZERS (L3)      │ │ & EXECUTION QUANTS   │
│ Ingestion, PIT       │ │ Kaplan-Meier,        │ │ Risk budgets, optimal       │ │ Lot-size liquidity,  │
│ features, zero bias  │ │ Cox-Lasso, RSF       │ │ holding, survival Sharpe    │ │ market makers, slip  │
└──────────────────────┘ └──────────────────────┘ └─────────────────────────────┘ └──────────────────────┘
```

---

## 2. Functional Roles & Responsibilities

| Pillar | Role Title | Key Focus & Responsibilities | Deliverables |
| :--- | :--- | :--- | :--- |
| **Data & Alt Data** | `data_quant_agent` (L2/L3) | Ingest raw Prowess, NSE feeds, subscription books, and 2025+ test data. Enforce strict **Point-in-Time (PIT)** correctness, eliminate survivorship and lookahead bias. | `data/processed/sme_survival_data.csv`, feature data dictionaries. |
| **Alpha Discovery** | `alpha_quant_agent` (L3) | Model time-to-event survival curves ($T, E$), train regularized Cox PH models, execute Random Survival Forest with 5-fold cross-validation, identify non-linear duration drivers. | Hazard ratios, $C$-Index evaluation tables, variable importance (VIMP). |
| **Portfolio & Risk** | `risk_portfolio_quant_agent` (L3) | Translate survival curves $S(t)$ into optimal exit horizons, design stop-loss hazard triggers, evaluate drawdown risk and sector concentration limits. | Holding period optimization rules, survival-adjusted Sharpe metrics. |
| **Microstructure** | `microstructure_quant_agent` (L3) | Analyze NSE Emerge trading frictions: mandatory lot size illiquidity, bid-ask spreads, 3-year market maker inventory obligations, and post-listing volume slippage. | Net executable return models, liquidity discount factors. |
| **Academic Research** | `research_paper_agent` (L4/Fellow) | Benchmark global SME literature (AIM, ChiNext), evaluate Competing Risk formulations (Fine-Gray model for mainboard migration), draft literature reviews and APA citations. | LaTeX drafts, methodology justification memos, citations. |

---

## 3. Horizontal Assembly Line Handoff Protocol

```
[Raw Prowess & NSE Feeds]
          │
          ▼
┌───────────────────┐    Feature Matrix (19 vars)    ┌───────────────────┐
│  Data Quant Team  │ ─────────────────────────────► │ Alpha Signal Team │
└───────────────────┘                                └─────────┬─────────┘
                                                               │ S(t), h(t), C-Index
                                                               ▼
┌───────────────────┐      Net Executable Returns    ┌───────────────────┐
│ Execution & Micro │ ◄───────────────────────────── │ Portfolio & Risk  │
└───────────────────┘                                └───────────────────┘
```

1. **Gate 1 (Data Quant $\to$ Alpha Quant)**:
   * Data integrity check: zero lookahead bias, verified event flags ($E \in \{0, 1\}$), all survival times $T \ge 1$.
2. **Gate 2 (Alpha Quant $\to$ Risk Quant)**:
   * Statistical hurdle: Harrell's $C$-index $> 0.60$, Schoenfeld residual validation for proportional hazards.
3. **Gate 3 (Risk Quant $\to$ Execution Quant)**:
   * Practical hurdle: position sizing constrained by NSE lot size minimums (e.g. 1,000 shares) and market maker liquidity capacity.

---

## 4. Governance & Review Rhythms

* **Lookahead Bias Audit**: Verify that balance sheet metrics (EPS, Debt-to-Asset) are lagged by at least 90 days from fiscal year-end before associating them with listing date events.
* **Survivorship Bias Audit**: All 577 listed SME IPOs must be accounted for; firms that migrated to the mainboard or were suspended are right-censored rather than deleted.
* **Token & Compute Efficiency Protocol**: Researchers report structured summary tables, parameter estimates, and performance metrics—raw tabular data is persisted locally or in parquet storage.
