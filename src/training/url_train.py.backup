# ===============================================================
# src/train_url_models.py  🚀 Model-Aware NaN Handling v3 (Binary)
# ---------------------------------------------------------------
# ✓ Uses final URL feature matrix (model-ready + imputed)
# ✓ Binary labels: phishing = 1, everything else = 0
# ✓ Automatically selects native vs imputed dataset per model
# ✓ Supports selective training using a subset of model names
# ✓ Preserves missingness as a feature (not a bug)
# ✓ Full cross-validation (ROC, F1, Precision, Recall)
# ===============================================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)

from src.training.model_zoo import get_models_for_dataset


# ----------------------------- Utilities -----------------------------
def make_non_negative_if_needed(model_name: str, X: pd.DataFrame) -> pd.DataFrame:
    """Ensure feature matrix is non-negative if model (e.g. CNB) requires it."""
    if "cnb" in model_name.lower():
        X = X.copy()
        X[X < 0] = 0
    return X


def safe_factorize(df: pd.DataFrame, keep_nan: bool = False) -> pd.DataFrame:
    """
    Encode all object columns safely.

    - If keep_nan=True:
        * Missing values stay as actual NaN (for tree models).
        * We factorize strings but restore NaNs in the encoded array.
    - If keep_nan=False:
        * Missingness becomes its own category (for linear/logistic models).
        * NaNs → 'MISSING' → integer code via factorize.
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns

    for col in obj_cols:
        col_series = df[col]

        if keep_nan:
            mask_nan = col_series.isna()
            filled = col_series.fillna("__MISSING_CAT__")
            codes, _ = pd.factorize(filled)
            codes = codes.astype("float64")
            codes[mask_nan] = np.nan
            df[col] = codes
        else:
            filled = col_series.fillna("MISSING")
            codes, _ = pd.factorize(filled)
            df[col] = codes

    return df


# ----------------------------- Paths -----------------------------
DATA_MAIN = "data/processed/url_features_modelready.csv"
DATA_IMPUTED = "data/processed/url_features_modelready_imputed.csv"
MODELDIR = "models"
ANADIR = "analysis"


# ----------------------------- Metrics -----------------------------
def _phish_metrics(y_true, y_prob, y_pred):
    """Compute ROC-AUC + phishing-class precision/recall/F1."""
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


# ----------------------------- Dataset Chooser -----------------------------
def needs_imputation(model_name: str) -> bool:
    """Check if model requires NaN-free data."""
    name = model_name.lower().strip()
    if any(k in name for k in ["xgb", "lgbm", "histgb"]):
        return False  # tree-based models handle NaNs
    return True


# ----------------------------- Main Training -----------------------------
def train_all_url_models(subset: list[str] | None = None):
    """
    Train all (or a chosen subset of) URL models as a binary classifier.

    Labels:
      - phishing   → 1
      - everything else (legit, legitimate, etc.) → 0
    """
    os.makedirs(MODELDIR, exist_ok=True)
    os.makedirs(ANADIR, exist_ok=True)

    # Load datasets
    df_main = pd.read_csv(DATA_MAIN)
    df_imp = pd.read_csv(DATA_IMPUTED)

    # Normalize label column on both
    for df in (df_main, df_imp):
        if "label" not in df.columns:
            raise ValueError("❌ 'label' column missing in URL feature datasets!")
        df["label"] = df["label"].astype(str).str.lower()
        # Collapse 'legit' variants into 'legitimate'
        df["label"] = df["label"].replace({
            "legit": "legitimate",
            "Legit": "legitimate",
            "Legitimate": "legitimate",
        })

    # Check alignment
    if set(df_main.columns) != set(df_imp.columns):
        raise ValueError("❌ Column mismatch between native and imputed URL datasets!")

    n = len(df_main)
    models = get_models_for_dataset("url", n_rows=n, allow_heavy=True, subset=subset)

    if not models:
        raise ValueError("❌ No models selected! Check your subset list.")

    print(f"\n📌 Training models: {list(models.keys())}")

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # NaN ratio on native dataset
    nan_ratio_native = df_main.isna().mean().mean()
    print(f"[INFO] Native dataset NaN ratio: {nan_ratio_native:.3%}")

    for name, model in models.items():
        # Dataset choice
        use_imputed = needs_imputation(name)
        df = df_imp if use_imputed else df_main
        version = "imputed" if use_imputed else "native"

        # Auto fallback if native too messy
        if not use_imputed and nan_ratio_native > 0.05:
            print(f"⚠️ {name}: too many NaNs in native → switching to imputed.")
            df = df_imp
            version = "imputed"
            use_imputed = True

        print(f"\n🚀 Training {name} → using {version}")

        # ---------------- Features ----------------
        # Drop non-feature columns
        X = df.drop(columns=["url", "label", "bucket"], errors="ignore")

        # Factorize categoricals with/without preserving NaNs
        keep_nan = (version == "native") and (not needs_imputation(name))
        X = safe_factorize(X, keep_nan=keep_nan)

        # ---------------- Labels (binary) ----------------
        # Labels are already normalized to 0/1 by dataset_builder
        y = df["label"].astype(int).values

        fold_stats = []
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
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

        # Aggregate metrics
        mean_stats = {k: np.mean([d[k] for d in fold_stats]) for k in fold_stats[0]}
        mean_stats["model"] = name
        mean_stats["dataset_version"] = version
        results.append(mean_stats)

        # Final fit on full data and save model
        X_full = make_non_negative_if_needed(name, X)
        model.fit(X_full, y)
        model_path = f"{MODELDIR}/url_{name}.pkl"
        joblib.dump(model, model_path)

        print(
            f"💾 Saved → {model_path} | "
            f"ROC={mean_stats['roc_auc']:.4f} | "
            f"F1(phish)={mean_stats['phish_f1']:.4f} | "
            f"ver={version}"
        )

    # Save results summary
    out = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    out_path = f"{ANADIR}/url_cv_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\n📊 URL CV results → {out_path}")
    return out


# ----------------------------- Entry Point -----------------------------
if __name__ == "__main__":
    print("🚀 Training selected URL models from each family...")

    # CLEANUP: Remove old URL models before training
    import glob
    old_models = glob.glob(f"{MODELDIR}/url_*.pkl")
    if old_models:
        print(f"🗑️  Removing {len(old_models)} old URL models...")
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
    results = train_all_url_models(subset=["logreg_elasticnet", "rf", "catboost", "lgbm", "xgb", "svm_rbf", "mlp"])
    print("\n🎯 Completed. Top results:")
    print(results.head(10))