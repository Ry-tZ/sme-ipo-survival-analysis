# Rule: Quantitative Research Governance & Scientific Integrity

## Objectives
1. Prevent Look-Ahead Bias: Never use aftermarket pricing data or future information to predict initial underpricing or Day-1 survival.
2. Event Definition Consistency:
   - Event $E = 1$: Occurs on the first trading day $t$ where $\text{Close}_t \le \text{Issue Price}$.
   - Censored $E = 0$: If the closing price remains strictly above issue price through observation end ($T = \text{last observed trading day}$).
3. Survivorship Bias Mitigation: Ensure all historical SME listings that failed, migrated, or delisted are preserved in the risk set until their exit day.
4. Reproducibility: Every empirical metric reported must be traceable to a script in `src/` and a table in `outputs/tables/`.
