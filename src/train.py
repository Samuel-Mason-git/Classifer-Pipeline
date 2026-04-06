import json
from pathlib import Path
from hashlib import md5

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # friendly message if dependency missing
    raise SystemExit("xgboost is not installed. Run `poetry add xgboost` and reinstall deps.") from exc

from utils.ml_helpers import split_data_by_time, log_test_run_md

project_root = Path(__file__).resolve().parent.parent


def load_data():
    df = pd.read_csv(project_root / "data" / "processed" / "btc_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def hash_dataset(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode()
    return md5(payload).hexdigest()[:12]


def train_model(X_train, y_train, model_params):
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, label="SET"):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
    }

    print(f"\n=== {label} METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def save_artifacts(model, metrics, model_name):
    model_path = project_root / "models" / f"{model_name}.joblib"
    metrics_path = project_root / "models" / f"{model_name}_metrics.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    df = load_data()

    feature_cols = [
        "return_1d",
        "return_3d",
        "return_7d",
        "volume_change_1d",
        "market_cap_change_1d",
        "volatility_7d",
        "price_ma_7",
        "price_ma_14",
    ]
    target_col = "target_next_day_up"

    train_ratio = 0.7
    val_ratio = 0.15
    
    X_train, X_val, X_test, y_train, y_val, y_test, train, val, test = split_data_by_time(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    MODEL_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = train_model(X_train, y_train, MODEL_PARAMS)

    val_metrics = evaluate_model(model, X_val, y_val, label="VALIDATION")
    test_metrics = evaluate_model(model, X_test, y_test, label="TEST")

    all_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
    }
    save_artifacts(model, all_metrics, model_name="xgboost_model")

    data_hash = hash_dataset(df)
    params = {
        "model": "XGBClassifier",
        **MODEL_PARAMS,
        "features": feature_cols,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }

    run_id = log_test_run_md(
        model_name="xgboost_classifier",
        params=params,
        metrics=test_metrics,
        data_hash=data_hash,
        notes="Xgboost classifier on BTC daily feature set",
        primary_metric="f1",
    )

    print(f"Logged test run: {run_id}")
