import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from src.config import FIGURES_DIR, TABLES_DIR

def run_kaplan_meier_analysis(df):
    """
    Performs non-parametric Kaplan-Meier survival estimation and log-rank tests.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall Kaplan-Meier Curve
    kmf = KaplanMeierFitter()
    kmf.fit(df['time'], event_observed=df['event'], label='All SME IPOs (N=436)')
    
    plt.figure(figsize=(9, 5))
    kmf.plot_survival_function(ci_show=True, color='#0b57d0')
    plt.title("Kaplan-Meier Survival Curve: Duration of Underpricing in SME IPOs", fontsize=12, fontweight='bold')
    plt.xlabel("Trading Days Since Listing")
    plt.ylabel("Probability of Remaining Underpriced S(t)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    km_fig_path = FIGURES_DIR / "km_overall_survival.png"
    plt.savefig(km_fig_path, dpi=300)
    plt.close()
    
    # Summary stats
    median_survival = kmf.median_survival_time_
    summary_dict = {
        "Metric": ["Total Cohort", "Events (Underpricing Ended)", "Right Censored", "Median Days Underpriced"],
        "Value": [len(df), int(df['event'].sum()), int((1 - df['event']).sum()), f"{median_survival:.1f} days"]
    }
    km_summary_df = pd.DataFrame(summary_dict)
    km_summary_df.to_csv(TABLES_DIR / "km_summary_stats.csv", index=False)
    
    # 2. Stratified by Period (Hot Market: 2023-2024 vs Pre-2023)
    hot_mask = df['is_hot_period'] == 1
    kmf_hot = KaplanMeierFitter()
    kmf_cold = KaplanMeierFitter()
    
    kmf_hot.fit(df.loc[hot_mask, 'time'], df.loc[hot_mask, 'event'], label='Hot Era (2023-2024)')
    kmf_cold.fit(df.loc[~hot_mask, 'time'], df.loc[~hot_mask, 'event'], label='Pre-2023 (2013-2022)')
    
    lr_res = logrank_test(df.loc[hot_mask, 'time'], df.loc[~hot_mask, 'time'],
                          df.loc[hot_mask, 'event'], df.loc[~hot_mask, 'event'])
    
    plt.figure(figsize=(9, 5))
    ax = kmf_hot.plot_survival_function(color='#d93025')
    kmf_cold.plot_survival_function(ax=ax, color='#1e8e3e')
    plt.title(f"SME Underpricing Survival: Hot vs Cold Era (Log-Rank p={lr_res.p_value:.4e})", fontsize=11, fontweight='bold')
    plt.xlabel("Trading Days Since Listing")
    plt.ylabel("Probability of Remaining Underpriced")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "km_hot_vs_cold.png", dpi=300)
    plt.close()
    
    return {
        "median_survival_days": median_survival,
        "logrank_hot_cold_p": lr_res.p_value
    }
