import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from src.config import TABLES_DIR, FIGURES_DIR

def run_microstructure_analysis():
    """
    Simulates and evaluates NSE Emerge & BSE SME microstructure constraints:
    1. Lot Size Friction: Minimum trade unit (typically 1,000 to 4,000 shares, ticket size >= Rs 100,000)
    2. Illiquidity Discount: As documented in Dhamija & Arora (2017), trading occurs on <72 days in Year 1.
    3. Slippage & Net Executable Return vs Theoretical Return.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path(r"C:\Users\novag\.gemini\antigravity\scratch\sme-ipo-survival-analysis\data\processed\sme_survival_data.csv")
    df = pd.read_csv(csv_path)
    
    # Representative lot sizing based on SEBI ICDR regulations:
    # Issue price <= 14: lot 10,000
    # 15 - 25: lot 6,000
    # 26 - 35: lot 4,000
    # 36 - 50: lot 3,000
    # 51 - 70: lot 2,000
    # 71 - 90: lot 1,600
    # 91 - 120: lot 1,200
    # 121 - 150: lot 1,000
    # 151 - 180: lot 800
    # 181 - 250: lot 600
    # 251 - 500: lot 400
    # > 500: lot 200
    
    def get_sebi_lot_size(price):
        if price <= 14: return 10000
        elif price <= 25: return 6000
        elif price <= 35: return 4000
        elif price <= 50: return 3000
        elif price <= 70: return 2000
        elif price <= 90: return 1600
        elif price <= 120: return 1200
        elif price <= 150: return 1000
        elif price <= 180: return 800
        elif price <= 250: return 600
        elif price <= 500: return 400
        else: return 200

    df['sebi_lot_size'] = df['issue_price'].apply(get_sebi_lot_size)
    df['lot_value_inr'] = df['sebi_lot_size'] * df['issue_price']
    
    # Traded volume relative to lot size on Listing Day (Day 1)
    df['traded_lots_l1'] = np.floor(df['traded_qty_l1'].fillna(0) / df['sebi_lot_size'])
    
    # Microstructure Liquidity Haircut (Amihud Proxy & Bid-Ask Spread Impact)
    # Illiquid issues with traded lots < 50 suffer 3-5% execution slippage; liquid issues suffer 0.75-1.5%
    df['slippage_pct'] = np.where(df['traded_lots_l1'] < 50, 4.5,
                          np.where(df['traded_lots_l1'] < 200, 2.5, 1.25))
    
    # Net Executable First Day Return
    df['net_executable_gain_pct'] = df['listing_gain_pct'] - df['slippage_pct']
    
    summary_stats = {
        'Metric': [
            'Mean SEBI Mandated Lot Size (Shares)',
            'Median Ticket Size Per Lot (INR)',
            'Mean Listing Day Traded Lots',
            'Median Listing Day Traded Lots',
            'Gross First-Day Mean Return (%)',
            'Net Executable Mean Return (%)',
            'Estimated Mean Execution Slippage (%)',
            'Firms with Severe Illiquidity (<50 lots Day 1)'
        ],
        'Value': [
            f"{df['sebi_lot_size'].mean():.0f}",
            f"INR {df['lot_value_inr'].median():,.0f}",
            f"{df['traded_lots_l1'].mean():.1f}",
            f"{df['traded_lots_l1'].median():.1f}",
            f"{df['listing_gain_pct'].mean():.2f}%",
            f"{df['net_executable_gain_pct'].mean():.2f}%",
            f"{df['slippage_pct'].mean():.2f}%",
            f"{(df['traded_lots_l1'] < 50).sum()} ({( (df['traded_lots_l1'] < 50).sum() / len(df) ) * 100:.1f}%)"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    out_table = TABLES_DIR / "microstructure_liquidity_summary.csv"
    summary_df.to_csv(out_table, index=False)
    print(f"[SUCCESS] Microstructure summary exported to {out_table}")
    print(summary_df.to_string(index=False))
    return summary_df

if __name__ == '__main__':
    run_microstructure_analysis()
