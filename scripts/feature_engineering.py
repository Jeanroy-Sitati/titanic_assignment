"""
feature_engineering.py
-----------------------
All the new features we create from the existing columns. This is where
most of the signal for the model actually comes from in this dataset.
"""

import pandas as pd
import numpy as np
import re


def create_family_features(df):
    """
    FamilySize and IsAlone - two of the more predictive features in Titanic data.
    Solo travellers and very large families both had lower survival rates.
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1   # +1 for the person themselves
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"[family] FamilySize range: {df['FamilySize'].min()} - {df['FamilySize'].max()}")
    print(f"[family] IsAlone: {df['IsAlone'].value_counts().to_dict()}")
    return df


def extract_title(df):
    """
    Pull the title out of the Name column. Title is surprisingly informative -
    it encodes gender, social class, and age group all at once.
    Rare titles get bucketed into 'Rare' to avoid sparse one-hot columns.
    """
    df = df.copy()

    # grab everything between comma and period - that's the title
    df["Title"] = df["Name"].str.extract(r',\s*([^\.]+)\.', expand=False)

    # also try just matching common patterns directly if the above misses anything
    # (some name formats skip the comma)
    fallback_mask = df["Title"].isnull()
    if fallback_mask.any():
        df.loc[fallback_mask, "Title"] = df.loc[fallback_mask, "Name"].str.extract(
            r'\b(Mr|Mrs|Miss|Master|Dr|Rev|Col|Capt|Lady|Major)\b', expand=False
        )

    # normalize the messy ones
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Countess": "Rare", "Capt": "Rare",
        "Col": "Rare", "Don": "Rare", "Dr": "Rare",
        "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
        "Jonkheer": "Rare", "Dona": "Rare"
    }
    df["Title"] = df["Title"].replace(title_map)

    # anything left that's not in our main categories → Rare
    main_titles = {"Mr", "Mrs", "Miss", "Master"}
    df["Title"] = df["Title"].apply(
        lambda t: t if pd.notna(t) and t in main_titles else "Rare"
    )

    print(f"[title] distribution:\n{df['Title'].value_counts().to_string()}")
    return df


def create_age_groups(df):
    """
    Bin Age into groups - Child/Teen/Adult/Senior.
    Helps the model pick up non-linear age effects.
    """
    df = df.copy()
    bins = [0, 12, 17, 60, 100]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    print(f"[age_group] distribution:\n{df['AgeGroup'].value_counts().to_string()}")
    return df


def create_fare_per_person(df):
    """
    The raw Fare for grouped tickets is often for the whole group,
    not per person. Dividing by FamilySize gives a more honest per-person cost.
    """
    df = df.copy()
    # avoid divide by zero just in case
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)
    return df


def log_transform_skewed(df):
    """
    Fare is very right-skewed (long tail of expensive first-class tickets).
    Log transform brings it closer to normal which helps linear/distance models.
    We use log1p to safely handle any zero fares.
    """
    df = df.copy()
    df["Fare_log"] = np.log1p(df["Fare"])
    df["FarePerPerson_log"] = np.log1p(df["FarePerPerson"])
    # Age is less skewed but still worth transforming
    df["Age_log"] = np.log1p(df["Age"])
    return df


def encode_categoricals(df):
    """
    One-hot encode nominal features. Pclass we leave as integer since it's
    already ordinal (1 > 2 > 3 in terms of class) - the model can use that ordering.
    """
    df = df.copy()
    ohe_cols = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # convert boolean cols to int (cleaner for models)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[encode] shape after OHE: {df.shape}")
    return df


def create_interaction_features(df):
    """
    Optional but useful: Pclass x Fare interaction captures the idea that
    price within a class is more meaningful than raw fare across classes.
    """
    df = df.copy()
    df["Pclass_x_Fare"] = df["Pclass"] * df["Fare_log"]
    return df


def feature_engineering_pipeline(input_path="data/train_cleaned.csv", output_path="data/train_engineered.csv"):
    df = pd.read_csv(input_path)
    print(f"[load] shape: {df.shape}")

    df = create_family_features(df)
    df = extract_title(df)
    df = create_age_groups(df)
    df = create_fare_per_person(df)
    df = log_transform_skewed(df)
    df = encode_categoricals(df)
    df = create_interaction_features(df)

    # drop columns that have served their purpose or add noise
    drop_cols = ["Name", "Ticket", "PassengerId"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df.to_csv(output_path, index=False)
    print(f"\n[done] engineered dataset saved → {output_path}")
    print(f"[done] final shape: {df.shape}")
    print(f"[done] columns:\n{list(df.columns)}")
    return df


if __name__ == "__main__":
    feature_engineering_pipeline()
