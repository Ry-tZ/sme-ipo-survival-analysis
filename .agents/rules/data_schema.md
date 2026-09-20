# Rule: Data Schema & Covariate Dictionary

When handling, transforming, or feeding covariates to models in this repository, strictly adhere to the 23-column survival schema in `data/processed/sme_survival_data.csv`:

| Column | Type | Definition |
| :--- | :--- | :--- |
| `co_code` | int/str | Prowess company identifier |
| `company_name` | str | Registered corporate name |
| `symbol` | str | Official NSE/BSE ticker symbol |
| `listing_date` | YYYY-MM-DD | Date of initial public listing |
| `listing_year` | int | Calendar year of listing |
| `time` | float | Active trading duration in days until Close <= Issue Price |
| `event` | int | Binary indicator: 1 = underpricing ended; 0 = right-censored |
| `issue_price` | float | IPO offer price per share (INR) |
| `listing_close` | float | Secondary market closing price on Day 1 (INR) |
| `listing_gain_pct` | float | Percentage return from issue to Day 1 close: `((listing_close - issue_price) / issue_price) * 100` |
| `firm_age` | float | Age of the company in years at the time of listing |
| `diff_issue_list_dates` | float | Calendar days between issue closure and listing day |
| `traded_qty_l1` | float | Total shares traded on listing day |
| `total_subs_times` | float | Final aggregate oversubscription multiple across all categories |
| `qib_subs_times` | float | Qualified Institutional Buyer oversubscription multiple |
| `nii_subs_times` | float | Non-Institutional Investor (HNI) oversubscription multiple |
| `eps` | float | Earnings per share (INR) prior to listing |
| `pe_ratio` | float | Price-to-Earnings valuation multiple |
| `debt_to_asset_ratio` | float | Financial leverage: total liabilities divided by total assets |
| `day1_subs` | float | Day 1 bidding subscription multiple |
| `day2_subs` | float | Day 2 bidding subscription multiple |
| `day3_subs` | float | Day 3 bidding subscription multiple |
| `subs_acceleration` | float | Bidding momentum: `Day 3 multiple / Day 1 multiple` |
