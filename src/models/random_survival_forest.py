import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from src.config import FIGURES_DIR, TABLES_DIR

def run_random_survival_forest(df, n_estimators=100, n_splits=5):
    """
    Trains a Random Survival Forest with 5-fold cross-validation and extracts feature importances.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    feature_cols = [
        'listing_gain_pct',
        'log_traded_qty',
        'eps',
        'pe_ratio_clipped',
        'debt_to_asset_ratio',
        'is_hot_period'
    ]
    
    X = df[feature_cols].copy().fillna(0)
    y = Surv.from_dataframe('event', 'time', df)
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    c_indices = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rsf = RandomSurvivalForest(n_estimators=n_estimators, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rsf.fit(X_train, y_train)
        c_score = rsf.score(X_test, y_test)
        c_indices.append(c_score)
        
    mean_c_index = float(np.mean(c_indices))
    
    # Full fit for feature importance
    final_rsf = RandomSurvivalForest(n_estimators=n_estimators, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    final_rsf.fit(X, y)
    
    # Predict sample survival curves
    sample_X = X.iloc[:4]
    surv_funcs = final_rsf.predict_survival_function(sample_X)
    
    plt.figure(figsize=(9, 5))
    for i, fn in enumerate(surv_funcs):
        plt.step(fn.x, fn(fn.x), where="post", label=f"Firm {i+1} (Gain: {sample_X.iloc[i]['listing_gain_pct']:.1f}%)")
    plt.title("Random Survival Forest: Predicted Survival Trajectories for Representative SMEs", fontsize=11, fontweight='bold')
    plt.xlabel("Trading Days Since Listing")
    plt.ylabel("Predicted Probability S(t)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rsf_predicted_curves.png", dpi=300)
    plt.close()
    
    rsf_metrics = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(n_splits)] + ["Mean CV"],
        "C_Index": [round(c, 4) for c in c_indices] + [round(mean_c_index, 4)]
    })
    rsf_metrics.to_csv(TABLES_DIR / "rsf_cv_metrics.csv", index=False)
    
    return {
        "mean_cv_c_index": mean_c_index,
        "fold_c_indices": c_indices
    }
