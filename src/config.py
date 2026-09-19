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
PROCESSED_SURVIVAL_PARQUET = PROCESSED_DATA_DIR / "sme_survival_data.parquet"

# Survival Definitions
EVENT_THRESHOLD_COL = "Close_Price"
EVENT_BENCHMARK_COL = "Issue_Price"  # Event occurs when Close <= Issue_Price
TIME_HORIZON_MAX_DAYS = 2500         # Study window limit (2013-2024)

# Default Model Features
CORE_FEATURES = [
    "Firm_Age",
    "Offer_Size_Cr",
    "Total_Subs_Times",
    "Traded_Qty_L1",
    "Listing_Gain_Pct",
    "EPS",
    "PE_Ratio",
    "Debt_to_Asset_Ratio"
]
