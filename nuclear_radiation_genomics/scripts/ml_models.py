from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional
    XGBRegressor = None


def train_classifier(
    features: pd.DataFrame,
    target_col: str = "outcome",
    model_name: str = "Random Forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    X = features.drop(columns=[target_col])
    # keep only numeric columns (drop identifiers like patient_id)
    X = X.select_dtypes(include=[np.number])
    y = features[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if model_name == "SVM":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ])
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state)

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")
    fpr, tpr, _ = roc_curve(y_test, proba)
    cm = confusion_matrix(y_test, preds)

    # CV accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    try:
        cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
    except Exception:
        cv_acc = float("nan")

    feature_importances = None
    if hasattr(clf, "feature_importances_"):
        feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    elif hasattr(clf, "named_steps") and "svc" in clf.named_steps:
        # SVM has no direct feature importances
        feature_importances = None

    return {
        "model": clf,
        "X_test": X_test,
        "y_test": y_test,
        "proba": proba,
        "preds": preds,
        "acc": acc,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "cm": cm,
        "cv_acc": cv_acc,
        "feature_importances": feature_importances,
    }


def train_regressor(
    features: pd.DataFrame,
    target_col: str = "mean_delta_expr",
    model_name: str = "XGBoost",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    X = features.drop(columns=[target_col, "outcome"]) if "outcome" in features else features.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    y = features[target_col].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_name == "Linear Regression" or XGBRegressor is None:
        reg = LinearRegression()
    else:
        reg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state)

    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    mse = float(np.mean((preds - y_test) ** 2))
    return {
        "model": reg,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
        "mse": mse,
    }


def cluster_patients(features: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    X = features.drop(columns=["outcome"]) if "outcome" in features else features.copy()
    X = X.select_dtypes(include=[np.number])
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return labels, km


