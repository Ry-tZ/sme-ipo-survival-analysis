import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from src.config import FIGURES_DIR, TABLES_DIR

def run_cox_ph_regression(df, penalizer=0.1):
    """
    Fits Elastic-Net penalized Cox Proportional Hazards regression.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    features = [
        'time',
        'event',
        'listing_gain_pct',
        'firm_age',
        'diff_issue_list_dates',
        'log_traded_qty',
        'total_subs_times',
        'log_retail_subs',
        'log_day1_subs',
        'log_subs_accel',
        'log_closing_surge',
        'eps',
        'pe_ratio_clipped',
        'debt_to_asset_ratio',
        'is_hot_period'
    ]

    cox_df = df[features].dropna().copy()
    
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.5)
    cph.fit(cox_df, duration_col='time', event_col='event')
    
    summary_df = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].rename(columns={'exp(coef)': 'hazard_ratio'})
    summary_df.to_csv(TABLES_DIR / "cox_ph_summary.csv")
    
    plt.figure(figsize=(9, 6))
    cph.plot()
    plt.title("Cox Proportional Hazards: Impact on Underpricing Termination Hazard", fontsize=11, fontweight='bold')
    plt.xlabel("log(Hazard Ratio)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cox_hazard_ratios.png", dpi=300)
    plt.close()
    
    # Subcohort with actual daily bidding data
    sub_df = df[df['has_daily_bidding_data'] == 1][features].dropna().copy()
    cph_sub = CoxPHFitter(penalizer=penalizer, l1_ratio=0.5)
    cph_sub.fit(sub_df, duration_col='time', event_col='event')
    sub_summary = cph_sub.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].rename(columns={'exp(coef)': 'hazard_ratio'})
    sub_summary.to_csv(TABLES_DIR / "cox_ph_subcohort_summary.csv")

    return {
        "concordance_index": cph.concordance_index_,
        "summary": summary_df,
        "subcohort_concordance_index": cph_sub.concordance_index_,
        "subcohort_summary": sub_summary
    }

