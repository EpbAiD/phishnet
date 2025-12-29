# ===============================================================
# src/training/whois_train.py  🌐 Binary WHOIS Model Trainer v3
# ---------------------------------------------------------------
# ✓ Binary labels (phishing vs. legit/other)
# ✓ Preserves NaN for tree-based models
# ✓ Implicit imputation for linear/logistic models
# ✓ Factorize categorical columns safely
# ✓ Subset model selection supported
# ===============================================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)

from src.training.model_zoo import get_models_for_dataset


# ----------------------------- Model Subset -----------------------------
WHOIS_MODEL_SUBSET = ["lgbm", "xgb", "rf", "catboost"]


# ----------------------------- Helpers -----------------------------
def collapse_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure labels are binary 0/1.
    Labels are already normalized by dataset_builder, but this provides backwards compatibility.
    """
    df = df.copy()

    # Check if labels are already numeric 0/1
    if pd.api.types.is_numeric_dtype(df["label"]):
        # Already normalized by dataset_builder
        df["label"] = df["label"].astype(int)
        # Verify only 0 and 1
        unique_labels = df["label"].unique()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"Expected labels 0/1, got: {unique_labels}")
        return df

    # Legacy: convert string labels (if any remain)
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    df["label"] = df["label"].replace({
        "legitimate": "legit",
        "benign": "legit",
        "safe": "legit",
        "0": "legit"
    })
    df["label"] = df["label"].replace({
        "phish": "phishing",
        "malicious": "phishing",
        "1": "phishing"
    })

    # enforce binary only
    df = df[df["label"].isin(["legit", "phishing"])].copy()
    return df


def make_non_negative_if_needed(model_name: str, X: pd.DataFrame) -> pd.DataFrame:
    """Some models like CNB cannot handle negatives."""
    if "cnb" in model_name.lower():
        X = X.copy()
        X[X < 0] = 0
    return X


def safe_factorize(df: pd.DataFrame, keep_nan: bool = False) -> pd.DataFrame:
    """Encode object columns; preserve NaNs for tree models."""
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns

    for col in obj_cols:
        s = df[col]

        if keep_nan:  # Preserve missingness
            mask = s.isna()
            filled = s.fillna("__MISSING_CAT__")
            codes, _ = pd.factorize(filled)
            codes = codes.astype("float64")
            codes[mask] = np.nan
            df[col] = codes
        else:  # Missingness as separate category
            filled = s.fillna("MISSING")
            df[col], _ = pd.factorize(filled)

    return df


def needs_imputation(model_name: str) -> bool:
    """Tree models can handle NaNs directly."""
    name = model_name.lower().strip()
    if any(k in name for k in ["xgb", "lgbm", "histgb", "catboost"]):
        return False
    return True


# ----------------------------- Paths -----------------------------
DATA_MAIN = "data/processed/whois_features_modelready.csv"
DATA_IMPUTED = "data/processed/whois_features_modelready_imputed.csv"
MODELDIR = "models"
ANADIR = "analysis"


# ----------------------------- Metrics -----------------------------
def _phish_metrics(y_true, y_prob, y_pred):
    roc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary"
    )
    return {
        "roc_auc": roc,
        "acc": acc,
        "phish_precision": p,
        "phish_recall": r,
        "phish_f1": f1,
    }


# ----------------------------- Main Training -----------------------------
def train_all_whois_models(subset=None):
    os.makedirs(MODELDIR, exist_ok=True)
    os.makedirs(ANADIR, exist_ok=True)

    df_main = collapse_binary_labels(pd.read_csv(DATA_MAIN))
    df_imp = collapse_binary_labels(pd.read_csv(DATA_IMPUTED))

    assert set(df_main.columns) == set(df_imp.columns), "❌ Column mismatch between WHOIS datasets!"

    y_binary = {
        "legit": 0,
        "phishing": 1
    }

    n = len(df_main)
    all_models = get_models_for_dataset("whois", n_rows=n, allow_heavy=True, subset=subset)
    models = all_models

    print(f"\n🔐 Training WHOIS models → {list(models.keys())}")

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        use_imp = needs_imputation(name)
        df = df_imp if use_imp else df_main
        version = "imputed" if use_imp else "native"

        print(f"\n🚀 WHOIS: {name} → using {version}")

        X = df.drop(columns=["url", "label","bucket"], errors="ignore")
        keep_nan_flag = (version == "native") and (not needs_imputation(name))
        X = safe_factorize(X, keep_nan=keep_nan_flag)

        # If labels are numeric (0/1), use directly; otherwise map from strings
        if pd.api.types.is_numeric_dtype(df["label"]):
            y = df["label"].astype(int).values
        else:
            y = df["label"].map(y_binary).values

        fold_stats = []
        for tr, va in skf.split(X, y):
            X_train = make_non_negative_if_needed(name, X.iloc[tr])
            X_val = make_non_negative_if_needed(name, X.iloc[va])

            model.fit(X_train, y[tr])

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_val)[:, 1]
            else:
                score = model.decision_function(X_val)
                mn, mx = score.min(), score.max()
                prob = (score - mn) / (mx - mn + 1e-12)

            pred = (prob >= 0.5).astype(int)
            fold_stats.append(_phish_metrics(y[va], prob, pred))

        mean_stats = {k: np.mean([d[k] for d in fold_stats]) for k in fold_stats[0]}
        mean_stats["model"] = name
        mean_stats["dataset_version"] = version
        results.append(mean_stats)

        X_full = make_non_negative_if_needed(name, X)
        model.fit(X_full, y)
        joblib.dump(model, f"{MODELDIR}/whois_{name}.pkl")

        print(f"💾 saved whois_{name}.pkl | ROC={mean_stats['roc_auc']:.4f} | F1={mean_stats['phish_f1']:.4f}")

    out = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    out.to_csv(f"{ANADIR}/whois_cv_results.csv", index=False)
    print(f"\n📊 WHOIS CV results → {ANADIR}/whois_cv_results.csv")

    return out


# ----------------------------- Entry -----------------------------
if __name__ == "__main__":
    print("🚀 Training selected WHOIS models from each family...")

    # CLEANUP: Remove old WHOIS models before training
    import glob
    old_models = glob.glob(f"{MODELDIR}/whois_*.pkl")
    if old_models:
        print(f"🗑️  Removing {len(old_models)} old WHOIS models...")
        for old_model in old_models:
            try:
                os.remove(old_model)
                print(f"   Removed: {os.path.basename(old_model)}")
            except Exception as e:
                print(f"   ⚠️  Could not remove {old_model}: {e}")

    # Selected models: one from each relevant family
    # 1. Linear: logreg_elasticnet
    # 2. Ensemble-Bagging: rf (Random Forest)
    # 3. Ensemble-Boosting: catboost, lgbm, xgb
    # 4. Kernel: svm_rbf (if dataset allows)
    # 5. Neural: mlp
    subset = ["logreg_elasticnet", "rf", "catboost", "lgbm", "xgb", "svm_rbf", "mlp"]
    print(train_all_whois_models(subset=subset))