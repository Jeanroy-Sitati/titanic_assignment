"""
data_cleaning.py
----------------
Handles all the messy stuff in the raw Titanic dataset before we do any
real analysis. Missing values, outliers, duplicates - the usual suspects.
"""

import pandas as pd
import numpy as np

def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    print(f"[load] shape: {df.shape}")
    return df


def inspect_missing(df):
    # just a quick look at what's missing and how bad it is
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "pct": pct})
    print("\n--- Missing Value Report ---")
    print(report[report["missing_count"] > 0].to_string())
    return report


def handle_missing(df):
    """
    Strategy per column:
    - Age: ~20% missing → median impute (median is robust to the skew we know exists)
    - Embarked: only 2-3 rows missing → mode impute, no biggie
    - Cabin: ~77% missing → way too many to impute meaningfully, extract deck letter
             then drop the raw column. We'll add a HasCabin indicator too.
    - Fare: 1 missing → median impute
    """
    df = df.copy()

    # Age - median impute grouped by Pclass+Sex gives better estimates than overall median
    age_median = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_median)
    # fallback if a group had all NaN somehow
    df["Age"] = df["Age"].fillna(df["Age"].median())
    print(f"[age] missing after impute: {df['Age'].isnull().sum()}")

    # Fare - just one row, use median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    print(f"[fare] missing after impute: {df['Fare'].isnull().sum()}")

    # Embarked - 2-3 rows, mode is fine here
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    print(f"[embarked] missing after impute: {df['Embarked'].isnull().sum()}")

    # Cabin - extract deck, add binary indicator, drop raw Cabin
    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    df["Deck"] = df["Cabin"].str[0]          # first letter = deck
    df["Deck"] = df["Deck"].fillna("Unknown")
    df.drop(columns=["Cabin"], inplace=True)
    print(f"[cabin] deck distribution:\n{df['Deck'].value_counts().to_string()}")

    return df


def handle_outliers(df):
    """
    Fare and Age are the numerical columns worth checking.
    - Fare: right-skewed, a few values > 300 exist (genuine first class). 
            Cap at 99th percentile to avoid distorting models.
    - Age: pretty clean 1-80 range after imputation, no action needed.
    """
    df = df.copy()

    # Fare outlier capping
    fare_99 = df["Fare"].quantile(0.99)
    print(f"[fare] 99th pct = {fare_99:.2f}, max = {df['Fare'].max():.2f}")
    df["Fare"] = df["Fare"].clip(upper=fare_99)
    print(f"[fare] max after cap = {df['Fare'].max():.2f}")

    # Age sanity check
    print(f"[age] range: {df['Age'].min():.1f} - {df['Age'].max():.1f}")
    # nothing alarming, leave as is

    return df


def fix_consistency(df):
    """
    Check for stuff like typos in categorical columns, duplicate rows, etc.
    """
    df = df.copy()

    # lowercase sex just in case
    df["Sex"] = df["Sex"].str.lower().str.strip()

    # drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[duplicates] removed {before - len(df)} rows")

    # make sure Pclass is integer
    df["Pclass"] = df["Pclass"].astype(int)

    return df


def clean_pipeline(input_path="data/train.csv", output_path="data/train_cleaned.csv"):
    df = load_data(input_path)
    inspect_missing(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df = fix_consistency(df)
    df.to_csv(output_path, index=False)
    print(f"\n[done] cleaned dataset saved → {output_path}")
    print(f"[done] final shape: {df.shape}")
    return df


if __name__ == "__main__":
    clean_pipeline()
