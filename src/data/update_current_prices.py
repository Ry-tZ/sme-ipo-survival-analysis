"""
Update SME IPO Survival Dataset with Current Market Prices & Returns Till Today
Sourced from Chittorgarh Report 25 (NSE Emerge & BSE SME).
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "sme_survival_data.csv"
REPORT25_CSV = DATA_DIR / "raw" / "chittorgarh_listing_history_all_sme.csv"

def norm(name):
    import re
    s = str(name).lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    for w in ['ltd', 'limited', 'pvt', 'private', 'india', 'ipo']:
        s = s.replace(w, '')
    return s

def update_prices():
    sme = pd.read_csv(PROCESSED_CSV)
    cur = pd.read_csv(REPORT25_CSV)
    
    sme['norm_name'] = sme['company_name'].apply(norm)
    cur['norm_name'] = cur['company_name'].apply(norm)
    sme['symbol_str'] = sme['symbol'].astype(str).str.strip().str.upper()
    cur['nse_str'] = cur['nse_symbol'].astype(str).str.strip().str.upper()
    cur['bse_str'] = cur['bse_scrip_code'].dropna().astype(int, errors='ignore').astype(str).str.strip()
    
    curr_prices = []
    curr_gains = []
    matched = 0
    
    for _, row in sme.iterrows():
        m = cur[cur['nse_str'] == row['symbol_str']]
        if len(m) == 0:
            m = cur[cur['bse_str'] == row['symbol_str']]
        if len(m) == 0:
            m = cur[cur['norm_name'] == row['norm_name']]
        if len(m) > 0:
            matched += 1
            r = m.iloc[0]
            p = r['current_price_nse'] if pd.notna(r['current_price_nse']) and r['current_price_nse'] > 0 else r['current_price_bse']
            g = r['current_gain_loss_pct']
            curr_prices.append(p)
            curr_gains.append(g)
        else:
            curr_prices.append(np.nan)
            curr_gains.append(np.nan)
            
    print(f"Matched {matched} / {len(sme)} companies with live prices.")
    sme['current_price_today'] = curr_prices
    sme['current_gain_pct_today'] = curr_gains
    sme['is_surviving_today'] = (sme['current_gain_pct_today'] > 0).astype(int)
    
    # Drop temp cols
    sme = sme.drop(columns=['norm_name', 'symbol_str'])
    sme.to_csv(PROCESSED_CSV, index=False)
    print(f"[SUCCESS] Updated {PROCESSED_CSV} with current prices and gains till today.")
    return sme

if __name__ == "__main__":
    update_prices()
