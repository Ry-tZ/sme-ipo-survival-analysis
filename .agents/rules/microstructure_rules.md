# Rule: Market Microstructure & Execution Constraints

## Mandated SEBI Lot Sizing
SME shares cannot be traded in single units. They must be traded in mandatory lot sizes defined by SEBI ICDR regulations:
- Minimum ticket size per application / lot is typically ₹1,00,000 to ₹1,40,000.
- Mean lot size in cohort: **2,106 shares** (median ticket size: **INR 126,000**).

## Execution Slippage Model
When evaluating theoretical strategies vs real-world executable returns:
1. Low Liquidity Issues (Day-1 Traded Lots < 50): Apply a **4.5%** execution haircut.
2. Moderate Liquidity Issues (50 <= Day-1 Traded Lots < 200): Apply a **2.5%** execution haircut.
3. High Liquidity Issues (Day-1 Traded Lots >= 200): Apply a **1.25%** execution haircut.
4. Average Empirical Execution Slippage: **1.73%**.
5. Gross Mean Day-1 Return (54.11%) $\rightarrow$ Net Executable Mean Return (**52.37%**).
