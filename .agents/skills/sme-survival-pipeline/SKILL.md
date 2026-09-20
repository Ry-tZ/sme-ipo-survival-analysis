---
name: sme-survival-pipeline
description: Run, evaluate, or extend the SME IPO Survival Analysis econometric pipeline (Kaplan-Meier, Penalized Cox PH, Random Survival Forest, and Microstructure metrics).
---

# SME Survival Pipeline Skill

Use this skill when you want to execute, modify, or extend the econometric survival models for Indian SME IPOs.

## Quick Execution
To run the full pipeline in PowerShell:
```powershell
& "C:\ProgramData\anaconda3\python.exe" src/pipeline.py
```

## Running Individual Modules
- **Kaplan-Meier Estimator & Log-Rank Tests**:
  ```powershell
  & "C:\ProgramData\anaconda3\python.exe" -c "from src.data.loader import load_survival_data; from src.models.km_estimator import run_kaplan_meier_analysis; run_kaplan_meier_analysis(load_survival_data())"
  ```
- **Penalized Cox PH Regression**:
  ```powershell
  & "C:\ProgramData\anaconda3\python.exe" -c "from src.data.loader import load_survival_data; from src.models.cox_ph import run_cox_ph_regression; run_cox_ph_regression(load_survival_data())"
  ```
- **Random Survival Forest (5-Fold CV)**:
  ```powershell
  & "C:\ProgramData\anaconda3\python.exe" -c "from src.data.loader import load_survival_data; from src.models.random_survival_forest import run_random_survival_forest; run_random_survival_forest(load_survival_data())"
  ```
- **Microstructure Slippage & Lot Sizing**:
  ```powershell
  & "C:\ProgramData\anaconda3\python.exe" src/models/microstructure_analysis.py
  ```
- **Portfolio Risk Allocation & Exit Horizons**:
  ```powershell
  & "C:\ProgramData\anaconda3\python.exe" src/models/portfolio_risk_engine.py
  ```

## Key Output Locations
- Figures: `outputs/figures/` (`km_overall_survival.png`, `km_hot_vs_cold.png`, `cox_hazard_ratios.png`, `rsf_predicted_curves.png`)
- Tables: `outputs/tables/` (`km_summary_stats.csv`, `cox_ph_summary.csv`, `rsf_cv_metrics.csv`, `optimal_exit_horizons.csv`, `microstructure_liquidity_summary.csv`)
