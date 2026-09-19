from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# File Paths
PROCESSED_SURVIVAL_CSV = PROCESSED_DATA_DIR / "sme_survival_data.csv"

# Survival Definitions
EVENT_THRESHOLD_COL = "listing_close"
EVENT_BENCHMARK_COL = "issue_price"  # Event occurs when Close <= Issue_Price

# Enriched Covariates Matrix
ALL_COVARIATES = [
    "listing_gain_pct",
    "firm_age",
    "diff_issue_list_dates",
    "log_traded_qty",
    "total_subs_times",
    "qib_subs_times",
    "nii_subs_times",
    "eps",
    "pe_ratio_clipped",
    "debt_to_asset_ratio",
    "is_hot_period"
]
