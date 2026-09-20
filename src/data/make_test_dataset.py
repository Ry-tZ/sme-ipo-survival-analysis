import pandas as pd
import numpy as np
from pathlib import Path

def make_2025_test_dataset():
    src_path = Path(r"C:\Users\novag\Downloads\sme-ipo-listing-date-list-history-price-bse-nse.csv")
    out_dir = Path(r"C:\Users\novag\.gemini\antigravity\scratch\sme-ipo-survival-analysis\data\processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sme_test_data_2025.csv"
    
    df = pd.read_csv(src_path)
    
    # Filter to underpriced on listing day
    df = df[df['Listing Day Gain / Loss (%)'] > 0].copy()
    
    df.rename(columns={
        'Company Name': 'company_name',
        'Listing Date': 'listing_date',
        'Issue Price (Rs)': 'issue_price',
        'Listing Day - Close Price (Rs)': 'listing_close',
        'Listing Day Gain / Loss (%)': 'listing_gain_pct',
        'Gain / Loss (%)': 'current_gain_loss_pct'
    }, inplace=True)
    
    # Event definition: If current gain/loss <= 0, underpricing has terminated (event=1)
    df['event'] = (df['current_gain_loss_pct'] <= 0).astype(int)
    
    # Parse dates and calculate trading duration observed till snapshot date (Feb 12, 2025)
    df['listing_datetime'] = pd.to_datetime(df['listing_date'], format='%b %d, %Y')
    snapshot_date = pd.to_datetime('2025-02-12')
    df['calendar_days'] = (snapshot_date - df['listing_datetime']).dt.days
    
    # Approximate trading days (5/7th of calendar days, min 1)
    df['time'] = np.maximum(1, np.round(df['calendar_days'] * (5.0 / 7.0))).astype(float)
    df['listing_year'] = 2025
    df['is_hot_period'] = 1
    
    # Reorder columns
    cols = ['company_name', 'listing_date', 'listing_year', 'time', 'event', 'issue_price', 
            'listing_close', 'listing_gain_pct', 'current_gain_loss_pct', 'is_hot_period']
    
    df_out = df[cols].reset_index(drop=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Exported {len(df_out)} 2025 test records to {out_csv}")
    print(f"Events observed: {df_out['event'].sum()}, Censored: {len(df_out) - df_out['event'].sum()}")

if __name__ == '__main__':
    make_2025_test_dataset()
