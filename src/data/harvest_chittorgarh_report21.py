"""
Harvest Complete Chittorgarh SME IPO Subscription History (Report 21)
Fetches all SME IPO subscription data from FY 2012-13 to FY 2025-26 via Chittorgarh Cloud API.
"""

import urllib.request
import json
import time
import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_FILE = RAW_DIR / "chittorgarh_subscription_all_sme.csv"

def clean_html_name(html_str):
    if not html_str:
        return ""
    # Extract text from <a href="...">Name</a>
    match = re.search(r'>([^<]+)<', str(html_str))
    if match:
        return match.group(1).strip()
    return str(html_str).strip()

def harvest_report21():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    
    # Financial years 2012 to 2025
    years = list(range(2012, 2026))
    print(f"Starting harvest for {len(years)} financial years (2012-13 to 2025-26)...")
    
    for y in years:
        fy = f"{y}-{str(y+1)[2:]}"
        page = 1
        fy_count = 0
        
        while True:
            url = f"https://webnodejs.chittorgarh.com/cloud/report/data-read/21/{page}/1/{y}/{fy}/0/sme/0?search=0&v=12-16"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://www.chittorgarh.com/"
                }
            )
            
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    rows = payload.get("reportTableData", [])
                    if not rows:
                        break
                    
                    for r in rows:
                        clean_row = {
                            "chittorgarh_id": r.get("~id"),
                            "company_name": clean_html_name(r.get("Company")),
                            "url_slug": r.get("~URLRewrite_Folder_Name"),
                            "closing_date": r.get("Closing Date"),
                            "issue_open_date": r.get("~Issue_Open_Date"),
                            "issue_close_date": r.get("~Issue_Close_Date"),
                            "issue_amount_cr": r.get("Issue Amount (Rs.cr.)"),
                            "qib_subs": r.get("QIB (x)"),
                            "s_nii_subs": r.get("sNII (x)"),
                            "b_nii_subs": r.get("bNII (x)"),
                            "nii_subs": r.get("NII (x)"),
                            "retail_subs": r.get("Retail (x)"),
                            "employee_subs": r.get("Employee (x)"),
                            "shareholder_subs": r.get("Shareholder (x)"),
                            "others_subs": r.get("Others (x)"),
                            "total_subs": r.get("Total (x)"),
                            "applications": r.get("Applications"),
                            "isin": r.get("~isin"),
                            "bse_script_code": r.get("~bse_script_code"),
                            "nse_symbol": r.get("~nse_symbol"),
                            "financial_year": fy,
                            "report_year": y
                        }
                        all_records.append(clean_row)
                        fy_count += 1
                        
                    page += 1
                    time.sleep(0.06)
            except Exception as e:
                print(f"  Error on {fy} page {page}: {e}")
                time.sleep(1.0)
                # retry once
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                        rows = payload.get("reportTableData", [])
                        if not rows:
                            break
                        for r in rows:
                            all_records.append({
                                "chittorgarh_id": r.get("~id"),
                                "company_name": clean_html_name(r.get("Company")),
                                "url_slug": r.get("~URLRewrite_Folder_Name"),
                                "closing_date": r.get("Closing Date"),
                                "issue_open_date": r.get("~Issue_Open_Date"),
                                "issue_close_date": r.get("~Issue_Close_Date"),
                                "issue_amount_cr": r.get("Issue Amount (Rs.cr.)"),
                                "qib_subs": r.get("QIB (x)"),
                                "s_nii_subs": r.get("sNII (x)"),
                                "b_nii_subs": r.get("bNII (x)"),
                                "nii_subs": r.get("NII (x)"),
                                "retail_subs": r.get("Retail (x)"),
                                "employee_subs": r.get("Employee (x)"),
                                "shareholder_subs": r.get("Shareholder (x)"),
                                "others_subs": r.get("Others (x)"),
                                "total_subs": r.get("Total (x)"),
                                "applications": r.get("Applications"),
                                "isin": r.get("~isin"),
                                "bse_script_code": r.get("~bse_script_code"),
                                "nse_symbol": r.get("~nse_symbol"),
                                "financial_year": fy,
                                "report_year": y
                            })
                            fy_count += 1
                        page += 1
                except Exception as e2:
                    print(f"  Retry failed for {fy} page {page}: {e2}. Continuing to next year.")
                    break
        
        print(f"  FY {fy}: {fy_count} SME IPOs harvested.")
        
    df = pd.DataFrame(all_records)
    # Deduplicate on chittorgarh_id if present, or company_name + closing_date
    df = df.drop_duplicates(subset=["chittorgarh_id"]).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[DONE] Harvested {len(df)} unique SME IPO subscription records to {OUTPUT_FILE}")

    return df

if __name__ == "__main__":
    harvest_report21()
