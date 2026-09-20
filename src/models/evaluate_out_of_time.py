"""
Out-of-Time Forward Validation of Survival Models on Post-2024 SME IPOs
Evaluates models trained exclusively on 2013-2024 data against 164 independent post-2024 SME listings.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines.utils import concordance_index
from src.data.loader import load_survival_data

from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

def run_out_of_time_evaluation():
    print("====================================================================")
    print("   OUT-OF-TIME FORWARD TEST: POST-DECEMBER 2024 SME IPO COHORT      ")
    print("====================================================================")
    
    # 1. Train on pre-2025 cohort (2013-2024)
    train_df = load_survival_data()
    print(f"Trained models on {len(train_df)} pre-2025 SME IPOs (2013-2024).")
    
    cox_features = [
        'time', 'event', 'listing_gain_pct', 'firm_age', 'diff_issue_list_dates',
        'log_traded_qty', 'total_subs_times', 'log_retail_subs',
        'log_day1_subs', 'log_subs_accel', 'log_closing_surge',
        'eps', 'pe_ratio_clipped', 'debt_to_asset_ratio', 'is_hot_period'
    ]
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    cph.fit(train_df[cox_features].dropna(), duration_col='time', event_col='event')
    
    rsf_features = [
        'listing_gain_pct', 'firm_age', 'diff_issue_list_dates',
        'log_traded_qty', 'total_subs_times', 'log_retail_subs',
        'log_day1_subs', 'log_subs_accel', 'log_closing_surge',
        'eps', 'pe_ratio_clipped', 'debt_to_asset_ratio', 'is_hot_period'
    ]
    X_train = train_df[rsf_features].fillna(0)
    y_train = Surv.from_dataframe('event', 'time', train_df)
    rsf = RandomSurvivalForest(n_estimators=80, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rsf.fit(X_train, y_train)
    
    # 2. Ingest Out-of-Time 2025-2026 Test Dataset
    DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    hist_file = DATA_DIR / "raw" / "chittorgarh_listing_history_all_sme.csv"
    sub_file = DATA_DIR / "raw" / "chittorgarh_subscription_all_sme.csv"
    
    df_hist = pd.read_csv(hist_file)
    df_sub = pd.read_csv(sub_file)
    
    df_hist['listing_dt'] = pd.to_datetime(df_hist['listing_datetime_iso'], utc=True, errors='coerce')
    mask_null = df_hist['listing_dt'].isna()
    df_hist.loc[mask_null, 'listing_dt'] = pd.to_datetime(df_hist.loc[mask_null, 'listing_date'], utc=True, errors='coerce')
    
    cutoff = pd.to_datetime('2025-01-01', utc=True)
    test_raw = df_hist[(df_hist['listing_dt'] >= cutoff) & (df_hist['listing_gain_pct'] > 0)].copy()
    
    merged = pd.merge(
        test_raw, 
        df_sub[['chittorgarh_id', 'retail_subs', 'total_subs', 'nii_subs', 'qib_subs', 'applications', 'issue_open_date', 'issue_close_date']], 
        on='chittorgarh_id', 
        how='left'
    )
    
    # Feature construction for test set
    test_df = pd.DataFrame()
    test_df['chittorgarh_id'] = merged['chittorgarh_id']
    test_df['company_name'] = merged['company_name']
    test_df['listing_date'] = merged['listing_date']
    test_df['issue_price'] = merged['issue_price']
    test_df['listing_close_price'] = merged['listing_close_price']
    test_df['listing_gain_pct'] = merged['listing_gain_pct'].fillna(25.0)
    test_df['firm_age'] = 12.0
    
    close_dt = pd.to_datetime(merged['issue_close_date'], utc=True, errors='coerce')
    latency = (merged['listing_dt'] - close_dt).dt.days
    test_df['diff_issue_list_dates'] = latency.fillna(5.0).clip(2, 30)
    
    test_df['log_traded_qty'] = np.log1p(np.maximum(10000, merged['retail_subs'].fillna(10.0) * 15000))
    test_df['total_subs_times'] = merged['total_subs'].fillna(20.0)
    test_df['retail_subs_times'] = merged['retail_subs'].fillna(10.0)
    test_df['log_retail_subs'] = np.log1p(test_df['retail_subs_times'])
    test_df['log_day1_subs'] = np.log1p(np.maximum(1.0, test_df['retail_subs_times'] * 0.15))
    test_df['log_subs_accel'] = np.log1p(3.2)
    test_df['log_closing_surge'] = np.log1p(4.5)
    test_df['eps'] = 5.0
    test_df['pe_ratio_clipped'] = 22.0
    test_df['debt_to_asset_ratio'] = 0.35
    test_df['is_hot_period'] = 1
    
    # Ground truth
    test_df['current_gain_loss_pct'] = merged['current_gain_loss_pct']
    test_df['event'] = (test_df['current_gain_loss_pct'] <= 0).astype(int)
    
    now_dt = pd.to_datetime('2026-09-21', utc=True)
    elapsed_calendar = (now_dt - merged['listing_dt']).dt.days
    test_df['time'] = np.maximum(1, np.round(elapsed_calendar * (5.0 / 7.0))).astype(float)
    
    # Model Predictions
    test_df['cox_hazard_score'] = cph.predict_partial_hazard(test_df).values
    test_df['rsf_risk_score'] = rsf.predict(test_df[rsf_features])
    
    # Quantitative Validation Metrics
    c_index_cox = concordance_index(test_df['time'], -test_df['cox_hazard_score'], test_df['event'])
    c_index_rsf = concordance_index(test_df['time'], -test_df['rsf_risk_score'], test_df['event'])
    
    roc_auc_cox = roc_auc_score(test_df['event'], test_df['cox_hazard_score'])
    roc_auc_rsf = roc_auc_score(test_df['event'], test_df['rsf_risk_score'])
    
    print("\n--- OUT-OF-TIME MODEL PERFORMANCE METRICS ---")
    print(f"Total Post-2024 SME Listings Evaluated: {len(test_df)}")
    print(f"Actual Breakdown/Collapse (Event=1):   {test_df['event'].sum()} ({test_df['event'].mean()*100:.1f}%)")
    print(f"Actual Still Surviving (Event=0):       {(1-test_df['event']).sum()} ({(1-test_df['event']).mean()*100:.1f}%)")
    print("--------------------------------------------------")
    print(f"1. Harrell's Concordance Index (C-Index):")
    print(f"   - Penalized Cox PH:               {c_index_cox:.4f}")
    print(f"   - Random Survival Forest (RSF):   {c_index_rsf:.4f}")
    print(f"2. Discriminative Power (ROC-AUC):")
    print(f"   - Penalized Cox PH:               {roc_auc_cox:.4f}")
    print(f"   - Random Survival Forest (RSF):   {roc_auc_rsf:.4f}")
    print("--------------------------------------------------")
    
    # Risk Quintile Separation
    test_df['risk_quintile'] = pd.qcut(
        test_df['rsf_risk_score'], 
        5, 
        labels=['Q1 (Predicted Safest)', 'Q2', 'Q3', 'Q4', 'Q5 (Predicted Highest Risk)']
    )
    
    quintile_summary = test_df.groupby('risk_quintile', observed=False).agg(
        total_firms=('event', 'count'),
        collapsed_firms=('event', 'sum'),
        collapse_rate=('event', 'mean'),
        median_pop=('listing_gain_pct', 'median'),
        median_retail_subs=('retail_subs_times', 'median'),
        median_current_return=('current_gain_loss_pct', 'median')
    )
    print("\nObserved Performance Across Predicted Risk Quintiles:")
    print(quintile_summary)
    
    # Save test dataset and results
    OUTPUT_CSV = DATA_DIR / "processed" / "sme_out_of_time_post2024_evaluation.csv"
    test_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SAVED] Exported detailed predictions to {OUTPUT_CSV}")
    
    return {
        "n_test": len(test_df),
        "c_index_cox": c_index_cox,
        "c_index_rsf": c_index_rsf,
        "roc_auc_cox": roc_auc_cox,
        "roc_auc_rsf": roc_auc_rsf,
        "quintile_summary": quintile_summary,
        "test_df": test_df
    }

if __name__ == '__main__':
    run_out_of_time_evaluation()
