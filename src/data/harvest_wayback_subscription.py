import os
import sys
import time
import random
import urllib.request
import pandas as pd
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load existing subscription data from user's directory
user_sub_file = r'C:\Users\novag\Desktop\python for finance\data types\ipo_subscription_data.xlsx'
wb_file = r'C:\Users\novag\Desktop\python for finance\data types\wayback_archived_urls.xlsx'
output_file = 'data/processed/sme_daily_subscription_multiples.csv'

os.makedirs('data/processed', exist_ok=True)

existing_records = []
if os.path.exists(user_sub_file):
    df_user = pd.read_excel(user_sub_file)
    valid_user = df_user.dropna(subset=['Day/Date'])
    for _, row in valid_user.iterrows():
        existing_records.append({
            'company_name': str(row['company_name']).strip(),
            'day_or_date': str(row['Day/Date']).strip(),
            'qib_multiple': row.get('QIB'),
            'nii_multiple': row.get('NII'),
            'retail_multiple': row.get('Retail'),
            'total_multiple': row.get('Total'),
            'source': 'user_scraped'
        })
    print(f"Loaded {len(existing_records)} existing rows across {len(set(r['company_name'] for r in existing_records))} companies from user excel.")

# Check if output file already has progress
already_done_companies = set(r['company_name'] for r in existing_records)
if os.path.exists(output_file):
    df_out = pd.read_csv(output_file)
    for _, row in df_out.iterrows():
        existing_records.append(row.to_dict())
        already_done_companies.add(str(row['company_name']).strip())
    print(f"Loaded additional progress from {output_file}. Total unique companies so far: {len(already_done_companies)}")

# 2. Load Wayback URLs
wb_df = pd.read_excel(wb_file)
existing_wb = wb_df[wb_df['status'] == 'existing'].copy()
print(f"Total available Wayback snapshots: {len(existing_wb)}")

to_process = existing_wb[~existing_wb['company_name'].str.strip().isin(already_done_companies)]
print(f"Companies remaining to process from Wayback: {len(to_process)}")

def parse_subscription_table(html, company_name):
    soup = BeautifulSoup(html, 'html.parser')
    rows_data = []
    
    for table in soup.find_all('table'):
        text = table.get_text()
        if 'Day 1' not in text and 'Day 2' not in text:
            continue
            
        # Parse headers
        all_trs = table.find_all('tr')
        if not all_trs:
            continue
            
        header_row = None
        for tr in all_trs[:3]:
            ths = [th.get_text(separator=' ', strip=True).lower() for th in tr.find_all(['th', 'td'])]
            if any('date' in h or 'day' in h for h in ths):
                header_row = ths
                break
                
        if not header_row:
            continue
            
        # Map column positions
        date_idx = -1
        qib_idx = -1
        nii_idx = -1
        retail_idx = -1
        total_idx = -1
        
        for idx, h in enumerate(header_row):
            if 'date' in h or 'day' in h:
                date_idx = idx
            elif 'qib' in h:
                qib_idx = idx
            elif 'nii' in h and 'bnii' not in h and 'snii' not in h:
                nii_idx = idx
            elif 'retail' in h or 'rii' in h:
                retail_idx = idx
            elif 'total' in h:
                total_idx = idx
                
        if total_idx == -1 and len(header_row) >= 4:
            total_idx = len(header_row) - 1
            
        # Now parse data rows
        for tr in all_trs:
            tds = [td.get_text(separator=' ', strip=True) for td in tr.find_all(['td', 'th'])]
            if not tds:
                continue
            first_cell = tds[0].strip()
            if not first_cell.startswith('Day'):
                continue
                
            # We found a day row!
            day_str = first_cell
            qib_val = tds[qib_idx] if qib_idx != -1 and qib_idx < len(tds) else None
            nii_val = tds[nii_idx] if nii_idx != -1 and nii_idx < len(tds) else None
            retail_val = tds[retail_idx] if retail_idx != -1 and retail_idx < len(tds) else None
            total_val = tds[total_idx] if total_idx != -1 and total_idx < len(tds) else None
            
            # Check if values are non-empty
            if any(v not in (None, '', '[.]', '-') for v in [qib_val, nii_val, retail_val, total_val]):
                rows_data.append({
                    'company_name': company_name,
                    'day_or_date': day_str,
                    'qib_multiple': qib_val,
                    'nii_multiple': nii_val,
                    'retail_multiple': retail_val,
                    'total_multiple': total_val,
                    'source': 'wayback_harvested'
                })
                
        if rows_data:
            break
            
    return rows_data

# Process loop with rate pacing
new_records = []
success_count = 0
empty_count = 0
error_count = 0

print("\nStarting batch harvester across Wayback snapshots...")

for i, (_, row) in enumerate(to_process.iterrows()):
    company = str(row['company_name']).strip()
    url = str(row['archive_url']).strip()
    
    print(f"[{i+1}/{len(to_process)}] Processing {company}...", end=" ", flush=True)
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            
        rows = parse_subscription_table(html, company)
        if rows:
            print(f"SUCCESS ({len(rows)} day rows)")
            new_records.extend(rows)
            success_count += 1
        else:
            print("EMPTY (no active day cells)")
            empty_count += 1
            
    except Exception as e:
        print(f"FAILED ({e})")
        error_count += 1
        
    # Incremental save every 5 companies
    if (i + 1) % 5 == 0 or (i + 1) == len(to_process):
        all_combined = existing_records + new_records
        df_save = pd.DataFrame(all_combined).drop_duplicates(subset=['company_name', 'day_or_date'])
        df_save.to_csv(output_file, index=False)
        print(f"--> Checkpointed {len(df_save)} total rows to {output_file}")
        
    # Paced delay between 1.5 and 3.0 seconds
    time.sleep(random.uniform(1.5, 3.0))

print("\n=== Harvest Finished ===")
print(f"Successfully scraped: {success_count}")
print(f"Empty/dynamic pages: {empty_count}")
print(f"Network/404 errors: {error_count}")
all_combined = existing_records + new_records
df_save = pd.DataFrame(all_combined).drop_duplicates(subset=['company_name', 'day_or_date'])
df_save.to_csv(output_file, index=False)
print(f"Final dataset written to {output_file} ({len(df_save)} rows across {df_save['company_name'].nunique()} companies)")
