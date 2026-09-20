"""
Enrich SME IPO Survival Dataset with Chittorgarh Cloud API Subscription Data
Merges retail subscription multiples, application counts, and retail-to-institutional ratios.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_CSV = DATA_DIR / "processed" / "sme_survival_data.csv"
CHITTORGARH_CSV = DATA_DIR / "raw" / "chittorgarh_subscription_all_sme.csv"

def norm_name(name):
    import re
    s = str(name).lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    for w in ['ltd', 'limited', 'pvt', 'private', 'india', 'ipo']:
        s = s.replace(w, '')
    return s

def clean_apps(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(',', '').strip()
    try:
        return float(s)
    except:
        return np.nan

def enrich_features():
    sme = pd.read_csv(PROCESSED_CSV)
    chit = pd.read_csv(CHITTORGARH_CSV)
    
    sme['norm_name'] = sme['company_name'].apply(norm_name)
    chit['norm_name'] = chit['company_name'].apply(norm_name)
    sme['symbol_str'] = sme['symbol'].astype(str).str.strip().str.upper()
    chit['nse_str'] = chit['nse_symbol'].astype(str).str.strip().str.upper()
    chit['bse_str'] = chit['bse_script_code'].dropna().astype(int, errors='ignore').astype(str).str.strip()
    
    retail_subs_list = []
    apps_list = []
    matched = 0
    
    for _, row in sme.iterrows():
        m = chit[chit['nse_str'] == row['symbol_str']]
        if len(m) == 0:
            m = chit[chit['bse_str'] == row['symbol_str']]
        if len(m) == 0:
            m = chit[chit['norm_name'] == row['norm_name']]
        if len(m) > 0:
            matched += 1
            retail_subs_list.append(m.iloc[0]['retail_subs'])
            apps_list.append(m.iloc[0]['applications'])
        else:
            retail_subs_list.append(np.nan)
            apps_list.append(np.nan)
            
    print(f"Matched {matched} / {len(sme)} firms with Chittorgarh Cloud data.")
    
    sme['retail_subs_times'] = pd.to_numeric(retail_subs_list, errors='coerce')
    # Fallback to total_subs_times where retail was missing (3 firms)
    sme['retail_subs_times'] = sme['retail_subs_times'].fillna(sme['total_subs_times'])
    
    sme['applications_count'] = [clean_apps(x) for x in apps_list]
    median_apps = sme['applications_count'].median()
    sme['applications_count_filled'] = sme['applications_count'].fillna(median_apps)
    
    # Feature transformations
    sme['log_retail_subs'] = np.log1p(sme['retail_subs_times'])
    sme['log_applications'] = np.log1p(sme['applications_count_filled'])
    sme['retail_to_nii_ratio'] = sme['retail_subs_times'] / (sme['nii_subs_times'] + 0.01)
    sme['qib_participated'] = (sme['qib_subs_times'] > 1.0).astype(int)
    
    # Drop temporary matching columns
    sme = sme.drop(columns=['norm_name', 'symbol_str'])
    sme.to_csv(PROCESSED_CSV, index=False)
    print(f"[SUCCESS] Updated {PROCESSED_CSV} with retail subscription and application breadth features.")
    return sme

if __name__ == "__main__":
    enrich_features()
