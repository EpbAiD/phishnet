# ===============================================================
# src/model_zoo.py
# Industry-grade model zoo for URL / DNS / WHOIS classification
# ---------------------------------------------------------------
# ✓ Supports heavy vs. lightweight models automatically
# ✓ Handles optional families (CatBoost, Imbalanced Learn)
# ✓ Allows explicit model subset control in training scripts
# ✓ Optimized for phishing detection (class_weight balanced)
# ===============================================================

from __future__ import annotations
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Optional families ------------------------------------------------------------
try:
    from catboost import CatBoostClassifier
    CATBOOST = True
except Exception:
    CATBOOST = False

try:
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    IMB = True
except Exception:
    IMB = False


# =============================================================================
# Helpers to build individual classifiers
# =============================================================================

def _logreg_elasticnet():
    return LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        solver="saga",
        max_iter=4000,
        class_weight="balanced",
        n_jobs=None,
        random_state=42
    )

def _logreg_l2():
    return LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=4000,
        class_weight="balanced",
        n_jobs=None,
        random_state=42
    )

def _svm_linear_calibrated():
    base = LinearSVC(class_weight="balanced", random_state=42)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)

def _svm_rbf():
    return SVC(
        C=2.0, gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    )


# =============================================================================
# Master Factory Function
# =============================================================================

def get_models_for_dataset(
    dataset: str,
    n_rows: int,
    allow_heavy: bool = True,
    subset: list[str] | None = None,
):
    """
    Return available models tuned for phishing detection.

    Parameters
    ----------
    dataset : {"url","dns","whois"}
    n_rows  : dataset size for scaling / pruning decisions
    allow_heavy : allow models with high training cost (e.g., SVM-RBF)
    subset : optional list of model names to restrict the zoo

    Returns
    -------
    dict[str, sklearn.Model]
    """

    # Decide availability of heavy models
    heavy_ok = allow_heavy and (n_rows <= 120_000)

    models = {
        # Linear family ---------------------------------------------------------
        "logreg_elasticnet": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", _logreg_elasticnet())
        ]),
        "logreg_l2": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", _logreg_l2())
        ]),
        "sgd_log": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                max_iter=5000,
                tol=1e-3,
                random_state=42,
            ))
        ]),

        # Trees / Ensembles -----------------------------------------------------
        "rf":         RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42),
        "extratrees": ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42),
        "gb":         GradientBoostingClassifier(random_state=42),
        "histgb":     HistGradientBoostingClassifier(learning_rate=0.08, random_state=42),

        # Gradient Boosting (industry strong) ----------------------------------
        "xgb": XGBClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        ),

        "lgbm": LGBMClassifier(
            n_estimators=800, num_leaves=48, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            class_weight="balanced",
            random_state=42,
        ),

        # Baselines -------------------------------------------------------------
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=31))
        ]),
        "cnb": ComplementNB(),

        # Neural Networks -------------------------------------------------------
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ]),
    }

    # Optional IMBENS models ---------------------------------------------------
    if IMB:
        models["brf"] = BalancedRandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
        models["easy_ensemble"] = EasyEnsembleClassifier(n_estimators=50, random_state=42, n_jobs=-1)

    # Optional CatBoost --------------------------------------------------------
    if CATBOOST:
        models["catboost"] = CatBoostClassifier(
            iterations=800, depth=6, learning_rate=0.07,
            verbose=False,
            auto_class_weights="Balanced",
            random_state=42,
        )

    # Add heavy SVMs only if dataset small enough ------------------------------
    if heavy_ok:
        models["linear_svm_cal"] = _svm_linear_calibrated()
        models["svm_rbf"] = _svm_rbf()

    # Light pruning for extremely large URL feature sets -----------------------
    if dataset == "url" and n_rows > 150_000:
        models.pop("svm_rbf", None)
        models.pop("linear_svm_cal", None)

    # Remove unavailable optional models safely --------------------------------
    if not IMB:
        models.pop("brf", None)
        models.pop("easy_ensemble", None)
    if not CATBOOST:
        models.pop("catboost", None)

    # Optional subset mask ------------------------------------------------------
    if subset:
        models = {k: v for k, v in models.items() if k in subset}

    return models