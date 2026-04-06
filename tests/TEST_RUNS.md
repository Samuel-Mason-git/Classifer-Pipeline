# Test Runs

Describe validation/test runs here (kept in Git; data artifacts stay ignored).

## Template
- Date/commit:
- Dataset slice / hash:
- Primary metric (e.g., F1):
- Secondary metrics:
- Model + params:
- Notes:

## Latest
- 2026-04-06T13:04:29Z | commit: N/A | model: logistic_regression | f1: 0.5769230769230769 | params: {"features": ["return_1d", "return_3d", "return_7d", "volume_change_1d", "market_cap_change_1d", "volatility_7d", "price_ma_7", "price_ma_14"], "max_iter": 1000, "model": "LogisticRegression", "train_ratio": 0.7, "val_ratio": 0.15} | data: 4a7993d7da9c | notes: Baseline logistic regression on BTC daily feature set | id: 56c69ab7a1
- 2026-04-06T13:03:08Z | commit: N/A | model: xgboost_classifier | f1: 0.35 | params: {"colsample_bytree": 0.8, "eval_metric": "logloss", "features": ["return_1d", "return_3d", "return_7d", "volume_change_1d", "market_cap_change_1d", "volatility_7d", "price_ma_7", "price_ma_14"], "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 3, "model": "XGBClassifier", "n_estimators": 100, "n_jobs": -1, "objective": "binary:logistic", "random_state": 42, "reg_alpha": 1.0, "reg_lambda": 2.0, "subsample": 0.8, "train_ratio": 0.7, "val_ratio": 0.15} | data: 4a7993d7da9c | notes: Xgboost classifier on BTC daily feature set | id: ce8e310af8
- 2026-04-06T12:52:51Z | commit: N/A | model: xgboost_classifier | f1: 0.5483870967741935 | params: {"colsample_bytree": 0.9, "features": ["return_1d", "return_3d", "return_7d", "volume_change_1d", "market_cap_change_1d", "volatility_7d", "price_ma_7", "price_ma_14"], "learning_rate": 0.05, "max_depth": 4, "model": "XGBClassifier", "n_estimators": 400, "reg_lambda": 1.0, "subsample": 0.9, "train_ratio": 0.7, "val_ratio": 0.15} | data: 4a7993d7da9c | notes: Xgboost classifier on BTC daily feature set | id: e754109e4c
- (add entries as you run tests)
