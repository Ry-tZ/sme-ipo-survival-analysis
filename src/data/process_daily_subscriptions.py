import pandas as pd
import numpy as np
import re
import os

def clean_name(name):
    if not isinstance(name, str): return ''
    n = name.upper()
    n = re.sub(r'\b(LIMITED|LTD\.?|INDIA|PVT|PRIVATE|\(INDIA\)|\.COM|CORP\.?|CORPORATION)\b', '', n)
    n = re.sub(r'[^A-Z0-9]', '', n)
    return n.strip()

def parse_day(val):
    if not isinstance(val, str): return np.nan
    m = re.search(r'Day\s*(\d+)', val, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return np.nan

def parse_date(val):
    if not isinstance(val, str): return None
    lines = val.split('\n')
    if len(lines) > 1:
        return lines[1].strip()
    return None

def clean_num(val):
    if pd.isna(val): return np.nan
    s = str(val).replace(',', '').strip()
    if s in ['', '[.]', '-', 'N/A', 'NA', 'None']: return np.nan
    try:
        return float(s)
    except:
        return np.nan

def main():
    print("=== Processing Day-by-Day Bidding Multiples ===")
    raw_file = 'data/processed/sme_daily_subscription_multiples.csv'
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Missing {raw_file}")
        
    df_raw = pd.read_csv(raw_file)
    print(f"Loaded raw records: {len(df_raw)}")
    
    # Filter valid rows
    valid = df_raw.dropna(subset=['day_or_date']).copy()
    valid['bidding_day'] = valid['day_or_date'].apply(parse_day)
    valid['bidding_date'] = valid['day_or_date'].apply(parse_date)
    valid = valid.dropna(subset=['bidding_day'])
    valid['bidding_day'] = valid['bidding_day'].astype(int)
    
    for col in ['total_multiple', 'qib_multiple', 'nii_multiple', 'retail_multiple']:
        valid[col] = valid[col].apply(clean_num)
        
    # Deduplicate
    valid = valid.drop_duplicates(subset=['company_name', 'bidding_day']).sort_values(['company_name', 'bidding_day'])
    
    # Save clean daily table
    clean_cols = ['company_name', 'bidding_day', 'bidding_date', 'total_multiple', 'qib_multiple', 'nii_multiple', 'retail_multiple', 'source']
    valid[clean_cols].to_csv(raw_file, index=False)
    print(f"Clean daily subscription table saved: {len(valid)} rows across {valid['company_name'].nunique()} companies.")
    
    # Compute company-level features
    pivot_records = []
    for cname, group in valid.groupby('company_name'):
        group = group.sort_values('bidding_day')
        rec = {'company_name': cname}
        rec['clean_name'] = clean_name(cname)
        max_day = group['bidding_day'].max()
        rec['total_bidding_days'] = max_day
        
        for d in [1, 2, 3]:
            d_row = group[group['bidding_day'] == d]
            if not d_row.empty:
                rec[f'day{d}_subs'] = d_row['total_multiple'].values[0]
                rec[f'day{d}_qib'] = d_row['qib_multiple'].values[0]
                rec[f'day{d}_nii'] = d_row['nii_multiple'].values[0]
                rec[f'day{d}_retail'] = d_row['retail_multiple'].values[0]
            else:
                rec[f'day{d}_subs'] = np.nan
                rec[f'day{d}_qib'] = np.nan
                rec[f'day{d}_nii'] = np.nan
                rec[f'day{d}_retail'] = np.nan
                
        final_row = group.iloc[-1]
        rec['final_day_subs'] = final_row['total_multiple']
        
        # Microstructure Dynamics
        d1 = rec.get('day1_subs')
        d2 = rec.get('day2_subs')
        dfin = rec.get('final_day_subs')
        
        # Acceleration Day 1 to Day 2
        if pd.notna(d1) and pd.notna(d2) and d1 > 0:
            rec['subs_acceleration'] = d2 / d1
        else:
            rec['subs_acceleration'] = np.nan
            
        # Closing Day Surge (Microstructure rush into final day)
        if pd.notna(d2) and pd.notna(dfin) and d2 > 0:
            rec['closing_day_surge'] = (dfin - d2) / d2
        else:
            rec['closing_day_surge'] = np.nan
            
        rec['has_daily_bidding_data'] = 1
        pivot_records.append(rec)
        
    features_df = pd.DataFrame(pivot_records)
    print(f"Engineered features for {len(features_df)} companies.")
    print("Sample engineered metrics:")
    print(features_df[['company_name', 'day1_subs', 'day2_subs', 'day3_subs', 'final_day_subs', 'subs_acceleration', 'closing_day_surge']].head(5))
    
    # 3. Merge into active survival analysis dataset
    surv_file = 'data/processed/sme_survival_data.csv'
    surv_df = pd.read_csv(surv_file)
    surv_df['clean_name'] = surv_df['company_name'].apply(clean_name)
    
    # Drop old placeholder columns if present
    for drop_col in ['day1_subs', 'day2_subs', 'day3_subs', 'subs_acceleration', 'closing_day_surge', 'has_daily_bidding_data']:
        if drop_col in surv_df.columns:
            surv_df = surv_df.drop(columns=[drop_col])
            
    # Merge features
    feature_cols = ['clean_name', 'day1_subs', 'day2_subs', 'day3_subs', 'subs_acceleration', 'closing_day_surge', 'has_daily_bidding_data']
    merged_surv = surv_df.merge(features_df[feature_cols], on='clean_name', how='left')
    
    merged_surv['has_daily_bidding_data'] = merged_surv['has_daily_bidding_data'].fillna(0).astype(int)
    
    # For missing daily values, impute with median from daily cohort to prevent dropouts in regression
    med_d1 = features_df['day1_subs'].median()
    med_d2 = features_df['day2_subs'].median()
    med_d3 = features_df['day3_subs'].median()
    med_acc = features_df['subs_acceleration'].median()
    med_surge = features_df['closing_day_surge'].median()
    
    print(f"\nCohort Medians for Daily Bidding: Day 1={med_d1:.2f}x, Day 2={med_d2:.2f}x, Day 3={med_d3:.2f}x, Accel={med_acc:.2f}x, Closing Surge={med_surge:.2f}x")
    
    merged_surv['day1_subs'] = merged_surv['day1_subs'].fillna(med_d1)
    merged_surv['day2_subs'] = merged_surv['day2_subs'].fillna(med_d2)
    merged_surv['day3_subs'] = merged_surv['day3_subs'].fillna(med_d3)
    merged_surv['subs_acceleration'] = merged_surv['subs_acceleration'].fillna(med_acc)
    merged_surv['closing_day_surge'] = merged_surv['closing_day_surge'].fillna(med_surge)
    
    merged_surv = merged_surv.drop(columns=['clean_name'])
    merged_surv.to_csv(surv_file, index=False)
    print(f"Updated {surv_file}: {len(merged_surv)} rows. Columns: {merged_surv.columns.tolist()}")

if __name__ == '__main__':
    main()
