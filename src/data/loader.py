import pandas as pd
import numpy as np
from src.config import PROCESSED_SURVIVAL_CSV

def load_survival_data(csv_path=PROCESSED_SURVIVAL_CSV):
    """
    Loads and pre-processes the enriched SME IPO survival dataset.
    """
    df = pd.read_csv(csv_path)
    
    # Typing
    df['time'] = df['time'].astype(float)
    df['event'] = df['event'].astype(int)
    df['listing_year'] = df['listing_year'].astype(int)
    
    # Feature Engineering
    df['log_traded_qty'] = np.log1p(df['traded_qty_l1'].fillna(0))
    df['log_listing_gain'] = np.log1p(np.maximum(0, df['listing_gain_pct'].fillna(0)))
    df['is_hot_period'] = (df['listing_year'] >= 2023).astype(int)
    
    # Bidding Dynamics & Acceleration
    df['log_day1_subs'] = np.log1p(df['day1_subs'].fillna(2.71))
    df['log_subs_accel'] = np.log1p(df['subs_acceleration'].fillna(3.16))
    df['log_closing_surge'] = np.log1p(np.maximum(0, df['closing_day_surge'].fillna(5.74)))
    df['has_daily_bidding_data'] = df['has_daily_bidding_data'].fillna(0).astype(int)

    
    # Numerical stability
    df['pe_ratio_clipped'] = np.clip(df['pe_ratio'].fillna(15.0), -50, 200)
    df['debt_to_asset_ratio'] = df['debt_to_asset_ratio'].fillna(0.45)
    df['firm_age'] = df['firm_age'].fillna(12.0)
    df['diff_issue_list_dates'] = df['diff_issue_list_dates'].fillna(7.0)
    df['total_subs_times'] = df['total_subs_times'].fillna(3.5)
    df['qib_subs_times'] = df['qib_subs_times'].fillna(1.0)
    df['nii_subs_times'] = df['nii_subs_times'].fillna(1.5)
    
    return df
