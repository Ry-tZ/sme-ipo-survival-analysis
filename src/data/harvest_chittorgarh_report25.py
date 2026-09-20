"""
Harvest Complete Chittorgarh SME IPO Listing & Current Price History (Report 25)
Fetches all SME IPO current prices, listing returns, and cumulative gains till today.
"""

import urllib.request
import json
import time
import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_FILE = RAW_DIR / "chittorgarh_listing_history_all_sme.csv"

def clean_html(text):
    if not text:
        return ""
    # Strip HTML tags
    cleaned = re.sub(r'<[^>]+>', '', str(text)).strip()
    return cleaned

def harvest_report25():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    
    years = list(range(2012, 2026))
    print(f"Starting harvest for {len(years)} financial years (Report 25: Live Prices & Returns)...")
    
    for y in years:
        fy = f"{y}-{str(y+1)[2:]}"
        page = 1
        fy_count = 0
        
        while True:
            url = f"https://webnodejs.chittorgarh.com/cloud/report/data-read/25/{page}/1/{y}/{fy}/0/sme/0?search=0&v=12-16"
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
                            "company_name": clean_html(r.get("Company")),
                            "url_slug": r.get("~urlrewrite_folder_name"),
                            "opening_date": r.get("Opening Date"),
                            "listing_date": r.get("Listing Date"),
                            "listing_datetime_iso": r.get("~IL_IPO_Listing_date"),
                            "exchange": r.get("Listing At"),
                            "isin": r.get("ISIN"),
                            "bse_scrip_code": r.get("BSE Scrip Code"),
                            "nse_symbol": r.get("NSE Symbol"),
                            "issue_price": pd.to_numeric(clean_html(r.get("Issue Price (Rs.)")), errors="coerce"),
                            "listing_open_price": pd.to_numeric(clean_html(r.get("Open Price on Listing (Rs.)")), errors="coerce"),
                            "listing_close_price": pd.to_numeric(clean_html(r.get("Close Price on Listing (Rs.)")), errors="coerce"),
                            "listing_gain_pct": pd.to_numeric(clean_html(r.get("% Gain/Loss (Issue price v/s close price on Listing)")), errors="coerce"),
                            "current_price_bse": pd.to_numeric(clean_html(r.get("Current Price <br>at BSE (Rs.)")), errors="coerce"),
                            "current_price_nse": pd.to_numeric(clean_html(r.get("Current Price <br>at NSE (Rs.)")), errors="coerce"),
                            "current_gain_loss_pct": pd.to_numeric(clean_html(r.get("Gain / Loss (%)")), errors="coerce"),
                            "financial_year": fy,
                            "report_year": y
                        }
                        all_records.append(clean_row)
                        fy_count += 1
                        
                    page += 1
                    time.sleep(0.05)
            except Exception as e:
                print(f"  Error on {fy} page {page}: {e}")
                time.sleep(0.5)
                break
                
        print(f"  FY {fy}: {fy_count} SME IPOs recorded.")
        
    df = pd.DataFrame(all_records)
    # Deduplicate on chittorgarh_id
    df = df.drop_duplicates(subset=["chittorgarh_id"]).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[DONE] Harvested {len(df)} records with current prices to {OUTPUT_FILE}")
    return df

if __name__ == "__main__":
    harvest_report25()
