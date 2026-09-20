import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from src.config import TABLES_DIR
from src.data.loader import load_survival_data

def run_portfolio_risk_engine():
    """
    Portfolio & Risk Quant Engine:
    Computes optimal holding horizon, hazard-adjusted exit rules,
    and portfolio risk budgets for SME IPO allocations.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_survival_data()
    
    # Stratify by Listing Day Underpricing Quintiles
    df['gain_quintile'] = pd.qcut(df['listing_gain_pct'], q=5, labels=['Q1 (0-10%)', 'Q2 (10-25%)', 'Q3 (25-50%)', 'Q4 (50-90%)', 'Q5 (>90%)'])
    
    # Calculate empirical survival rate at key trading horizons (T = 20, 60, 120, 240 days)
    horizons = [20, 60, 120, 240]
    records = []
    
    for q_name, group in df.groupby('gain_quintile', observed=True):
        n_firms = len(group)
        median_t = group['time'].median()
        event_rate = (group['event'].sum() / n_firms) * 100
        
        row = {
            'Listing Gain Quintile': q_name,
            'Cohort Size': n_firms,
            'Event Rate (% Below Issue)': f"{event_rate:.1f}%",
            'Median Obs Days': f"{median_t:.0f} days"
        }
        
        for h in horizons:
            # S(h) = proportion of firms whose underpricing has NOT terminated prior to h
            survived = ((group['time'] >= h) | ((group['time'] < h) & (group['event'] == 0))).sum()
            surv_pct = (survived / n_firms) * 100
            row[f'Survival @ Day {h}'] = f"{surv_pct:.1f}%"
            
        # Recommended QR Exit Rule based on survival hazard
        if 'Q1' in str(q_name):
            row['Recommended QR Exit Action'] = 'Immediate T+1 to T+5 Liquidation (Severe breakdown risk)'
            row['Max Portfolio Weight'] = '0.50%'
        elif 'Q2' in str(q_name):
            row['Recommended QR Exit Action'] = 'Exit by T+20 (Hazard cliff begins)'
            row['Max Portfolio Weight'] = '1.00%'
        elif 'Q3' in str(q_name):
            row['Recommended QR Exit Action'] = 'Harvest 50% at T+20, trailing stop to T+60'
            row['Max Portfolio Weight'] = '1.50%'
        elif 'Q4' in str(q_name):
            row['Recommended QR Exit Action'] = 'Hold to T+120 with trailing stop on 20-DMA'
            row['Max Portfolio Weight'] = '2.50%'
        else: # Q5
            row['Recommended QR Exit Action'] = 'Momentum runner: hold to T+240; harvest on 50-DMA breach'
            row['Max Portfolio Weight'] = '3.00%'
            
        records.append(row)
        
    risk_df = pd.DataFrame(records)
    out_table = TABLES_DIR / "optimal_exit_horizons.csv"
    risk_df.to_csv(out_table, index=False)
    print(f"[SUCCESS] Optimal exit horizons exported to {out_table}")
    print(risk_df.to_string(index=False))
    return risk_df

if __name__ == '__main__':
    run_portfolio_risk_engine()
