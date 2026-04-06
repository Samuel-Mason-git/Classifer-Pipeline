from __future__ import annotations
import json
from hashlib import md5
from pathlib import Path
from datetime import datetime

TEST_LOG = Path("tests/TEST_RUNS.md")

def log_test_run_md(
    model_name: str,
    params: dict,
    metrics: dict,
    data_hash: str,
    notes: str = "",
    commit: str | None = None,
    primary_metric: str = "f1",
    ):
    """
    Appends/updates a bullet under '## Latest' in tests/TEST_RUNS.md.
    De-duplicates by run_id (model+params+data_hash).
    """
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    TEST_LOG.touch(exist_ok=True)

    run_id = md5((model_name + json.dumps(params, sort_keys=True) + data_hash).encode()).hexdigest()[:10]
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    score = metrics.get(primary_metric, "n/a")

    line = (
        f"- {ts} | commit: {commit or 'N/A'} | model: {model_name} | "
        f"{primary_metric}: {score} | params: {json.dumps(params, sort_keys=True)} | "
        f"data: {data_hash} | notes: {notes} | id: {run_id}"
    )

    text = TEST_LOG.read_text(encoding="utf-8")
    if "## Latest" not in text:
        text = "# Test Runs\n\n## Latest\n"

    lines = text.splitlines()
    latest_idx = next(i for i, l in enumerate(lines) if l.strip() == "## Latest")
    # drop any existing line with same run_id
    lines = [l for l in lines if run_id not in l]
    # insert after the header
    lines.insert(latest_idx + 1, line)

    TEST_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_id

def split_data_by_time(df, feature_cols, target_col, train_ratio=0.7, val_ratio=0.15):
    # assumes df is sorted oldest → newest
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    X_train, y_train = train[feature_cols], train[target_col]
    X_val,   y_val   = val[feature_cols], val[target_col]
    X_test,  y_test  = test[feature_cols], test[target_col]

    return X_train, X_val, X_test, y_train, y_val, y_test, train, val, test