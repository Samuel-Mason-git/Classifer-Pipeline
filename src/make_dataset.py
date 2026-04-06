import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# ---------------------------------------------
# Create Daily dataset
# ---------------------------------------------
def daily_dataset(debug=False):
    '''
    Creates a daily dataset with a date field, sorted by date.

    args:
        debug: boolean for if to pass dataframe into the sanity check

    returns: 
        daily dataset
    '''
    raw = pd.read_csv(project_root / 'data' / 'raw' / 'btc_data.csv')
    raw['date_time'] = pd.to_datetime(raw['date_time'])
    raw['date'] = raw['date_time'].dt.date

    # Group by date and take the last observation
    daily = (
        raw.sort_values('date_time')
        .groupby('date', as_index=False)
        .last()
    )
    daily = daily.drop(columns=["timestamp", "date_time"])
    daily["target_next_day_up"] = (daily["price"].shift(-1) > daily["price"]).astype(int)
    daily = daily.iloc[:-1].copy() # Remove last day since it cant have a label yet
    if debug:
        debug_dataset(df=daily, 
                      target='target_next_day_up')
    return daily

# -----------------------
# Sanity check (Optional)
# -----------------------
def debug_dataset(df, target=None):
    '''
    Provides basic descriptive sanity checks for the quality of the dataset.

    args:
        dataframe: dataframe for sanity check.
        target: target column of data for a distribution check.
    '''
    print("\n==============================")
    print("=== DATASET OVERVIEW ===")
    print("==============================")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # -----------------------
    # Dtypes
    # -----------------------
    print("\n=== DTYPES ===")
    print(df.dtypes)

    # -----------------------
    # Nulls
    # -----------------------
    print("\n=== NULL CHECK ===")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("No nulls")
    else:
        print(nulls[nulls > 0])

    # -----------------------
    # Duplicates
    # -----------------------
    print("\n=== DUPLICATES ===")
    total_dupes = df.duplicated().sum()
    print(f"Total duplicate rows: {total_dupes}")

    if "date" in df.columns:
        date_dupes = df["date"].duplicated().sum()
        print(f"Duplicate dates: {date_dupes}")

    # -----------------------
    # Infinite values
    # -----------------------
    print("\n=== INF CHECK ===")
    inf_count = (df == float("inf")).sum().sum()
    ninf_count = (df == float("-inf")).sum().sum()
    print(f"Inf values: {inf_count}")
    print(f"-Inf values: {ninf_count}")

    # -----------------------
    # Basic stats
    # -----------------------
    print("\n=== BASIC STATS ===")
    print(df.describe())

    # -----------------------
    # Target distribution
    # -----------------------
    if target and target in df.columns:
        print("\n=== TARGET DISTRIBUTION ===")
        print(df[target].value_counts(normalize=True))

    # -----------------------
    # Outliers (z-score)
    # -----------------------
    print("\n=== OUTLIER CHECK (Z-SCORE > 3) ===")
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        std = df[col].std()
        if std == 0:
            continue
        z = (df[col] - df[col].mean()) / std
        outliers = (z.abs() > 3).sum()
        if outliers > 0:
            print(f"{col}: {outliers} potential outliers")

    # -----------------------
    # Date continuity (time series integrity)
    # -----------------------
    if "date" in df.columns:
        print("\n=== DATE CONTINUITY ===")
        df_sorted = df.sort_values("date")
        date_series = pd.to_datetime(df_sorted["date"])
        gaps = date_series.diff().dt.days
        missing_gaps = gaps[gaps > 1]
        if len(missing_gaps) == 0:
            print("No missing dates")
        else:
            print(f"Missing date gaps: {len(missing_gaps)}")
            print(missing_gaps.head())

    print("\n=== DEBUG COMPLETE ===")

# -----------------------
# Feature creation
# -----------------------
def feature_creation(df):
    '''
    Creates features for the model.

    args:
        df: dataframe passed in
    
    returns:
        return % change for: 1, 3 & 7 days
        volume change 1 day
        market cap change 1 day
        volatility 7 days
        rolling 7, 14 day average price

    '''
    df = df.copy()
    df["return_1d"] = df["price"].pct_change()
    df["return_3d"] = df["price"].pct_change(3)
    df["return_7d"] = df["price"].pct_change(7)

    df["volume_change_1d"] = df["volume"].pct_change()
    df["market_cap_change_1d"] = df["market_cap"].pct_change()

    df["volatility_7d"] = df["return_1d"].rolling(7).std()
    df["price_ma_7"] = df["price"].rolling(7).mean()
    df["price_ma_14"] = df["price"].rolling(14).mean()
    df = df.dropna().reset_index(drop=True)
    return df

# -----------------------
# Save processed dataset
# -----------------------
def save_processed_data(df):
    output_path = project_root / "data" / "processed" / "btc_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    daily = daily_dataset(debug=True)
    featured = feature_creation(daily)
    debug_dataset(featured, target="target_next_day_up")
    save_processed_data(featured)
