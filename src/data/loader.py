import pandas as pd
import numpy as np
from pathlib import Path
from src.config import PROCESSED_SURVIVAL_CSV

def load_survival_data(csv_path=PROCESSED_SURVIVAL_CSV):
    """
    Loads and pre-processes the SME IPO survival dataset.
    Returns:
        pd.DataFrame: Clean dataset ready for modeling.
    """
    df = pd.read_csv(csv_path)
    
    # Ensure proper data types
    df['time'] = df['time'].astype(float)
    df['event'] = df['event'].astype(int)
    df['listing_year'] = df['listing_year'].astype(int)
    
    # Feature engineering
    df['log_traded_qty'] = np.log1p(df['traded_qty_l1'].fillna(0))
    df['log_listing_gain'] = np.log1p(np.maximum(0, df['listing_gain_pct']))
    df['is_hot_period'] = (df['listing_year'] >= 2023).astype(int)
    
    # Clip extreme PE ratios for numerical stability
    df['pe_ratio_clipped'] = np.clip(df['pe_ratio'].fillna(15.0), -100, 300)
    df['debt_to_asset_ratio'] = df['debt_to_asset_ratio'].fillna(0.45)
    
    return df
