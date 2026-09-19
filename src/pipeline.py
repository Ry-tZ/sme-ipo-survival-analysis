import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_survival_data
from src.models.km_estimator import run_kaplan_meier_analysis
from src.models.cox_ph import run_cox_ph_regression
from src.models.random_survival_forest import run_random_survival_forest

def main():
    print("====================================================================")
    print("   SME IPO SURVIVAL ANALYSIS: REPRODUCIBLE ECONOMETRIC PIPELINE     ")
    print("====================================================================")
    
    print("\n[1/4] Loading compiled survival dataset (436 underpriced companies)...")
    df = load_survival_data()
    print(f"Loaded {len(df)} firms: {df['event'].sum()} events observed, {(1 - df['event']).sum()} right-censored.")
    
    print("\n[2/4] Executing Kaplan-Meier Survival Estimation & Log-Rank Tests...")
    km_res = run_kaplan_meier_analysis(df)
    print(f"Median survival duration: {km_res['median_survival_days']} trading days.")
    print(f"Log-Rank p-value (Hot vs Cold era): {km_res['logrank_hot_cold_p']:.4e}")
    
    print("\n[3/4] Fitting Penalized Cox Proportional Hazards Model...")
    cox_res = run_cox_ph_regression(df)
    print(f"Cox PH Concordance Index: {cox_res['concordance_index']:.4f}")
    
    print("\n[4/4] Training Random Survival Forest with 5-Fold Cross Validation...")
    rsf_res = run_random_survival_forest(df)
    print(f"RSF 5-Fold Cross-Validated C-Index: {rsf_res['mean_cv_c_index']:.4f}")
    
    print("\n====================================================================")
    print("Pipeline completed successfully! All tables and figures exported.")
    print("Figures -> outputs/figures/")
    print("Tables  -> outputs/tables/")
    print("====================================================================")

if __name__ == '__main__':
    main()
