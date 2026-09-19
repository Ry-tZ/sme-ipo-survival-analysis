import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from src.config import FIGURES_DIR, TABLES_DIR

def run_cox_ph_regression(df, penalizer=0.1):
    """
    Fits L1-penalized Cox Proportional Hazards regression and outputs hazard ratios.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    features = [
        'time',
        'event',
        'listing_gain_pct',
        'log_traded_qty',
        'eps',
        'pe_ratio_clipped',
        'debt_to_asset_ratio',
        'is_hot_period'
    ]
    
    cox_df = df[features].dropna().copy()
    
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.5) # Elastic Net / Lasso penalty
    cph.fit(cox_df, duration_col='time', event_col='event')
    
    # Save coefficients and Hazard Ratios
    summary_df = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].rename(columns={'exp(coef)': 'hazard_ratio'})
    summary_df.to_csv(TABLES_DIR / "cox_ph_summary.csv")
    
    # Plot Hazard Ratios
    plt.figure(figsize=(8, 4.5))
    cph.plot()
    plt.title("Cox Proportional Hazards: Covariate Effects on Underpricing Hazard", fontsize=11, fontweight='bold')
    plt.xlabel("log(Hazard Ratio)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cox_hazard_ratios.png", dpi=300)
    plt.close()
    
    # Schoenfeld residuals test
    try:
        assump = cph.check_assumptions(cox_df, show_plots=False)
    except Exception:
        pass
        
    return {
        "concordance_index": cph.concordance_index_,
        "summary": summary_df
    }
