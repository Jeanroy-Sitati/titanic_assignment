"""
feature_selection.py
--------------------
After all the engineering, we need to decide what actually goes into the model.
Uses correlation analysis + Random Forest feature importance to narrow it down.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (headless environment)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def remove_high_correlation(df, threshold=0.90):
    """
    Drop one of any pair of features with correlation > threshold.
    No point feeding two almost-identical columns to the model.
    """
    # only check numeric columns
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["Survived"], errors="ignore")
    corr_matrix = num_df.corr().abs()

    # upper triangle only - avoid double counting pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # find columns where any correlation exceeds threshold
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"[corr] dropping {len(to_drop)} high-correlation features: {to_drop}")

    df = df.drop(columns=to_drop)
    return df, to_drop


def get_feature_importance(df, top_n=15):
    """
    Quick Random Forest to rank features. Not tuned - just using defaults
    to get an idea of what's actually predictive.
    """
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # handle any remaining non-numeric cols (shouldn't be many after OHE)
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n[rf_importance] top {top_n} features:")
    print(importance_df.head(top_n).to_string(index=False))

    # save a bar chart of importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top = importance_df.head(top_n)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Random Forest Feature Importance (top {top_n})")
    plt.tight_layout()
    plt.savefig("notebooks/feature_importance.png", dpi=150)
    plt.close()
    print("[rf_importance] plot saved → notebooks/feature_importance.png")

    return importance_df


def select_features(df, importance_df, min_importance=0.01):
    """
    Keep features with importance >= min_importance threshold.
    Also always keep a few hand-picked ones we know are important from domain knowledge.
    """
    must_keep = {"Pclass", "Fare_log", "Age_log", "FamilySize", "IsAlone"}
    
    selected = set(importance_df[importance_df["importance"] >= min_importance]["feature"].tolist())
    selected = selected | (must_keep & set(df.columns))  # union with must-keep

    # always keep target
    selected.add("Survived")

    selected_cols = [c for c in df.columns if c in selected]
    print(f"\n[select] keeping {len(selected_cols)} features")
    print(selected_cols)
    return df[selected_cols], selected_cols


def feature_selection_pipeline(input_path="data/train_engineered.csv", output_path="data/train_selected.csv"):
    df = pd.read_csv(input_path)
    print(f"[load] shape: {df.shape}")

    df, dropped_corr = remove_high_correlation(df)
    importance_df = get_feature_importance(df)
    df_selected, final_features = select_features(df, importance_df)

    df_selected.to_csv(output_path, index=False)
    print(f"\n[done] selected dataset saved → {output_path}")
    print(f"[done] final shape: {df_selected.shape}")

    # summary report
    print("\n" + "="*50)
    print("FEATURE SELECTION SUMMARY")
    print("="*50)
    print(f"  Dropped for high correlation: {dropped_corr}")
    print(f"  Final features ({len(final_features)}):")
    for f in sorted(final_features):
        imp = importance_df[importance_df['feature']==f]['importance'].values
        imp_str = f"{imp[0]:.4f}" if len(imp)>0 else "N/A"
        print(f"    {f:<35} importance={imp_str}")

    return df_selected, importance_df


if __name__ == "__main__":
    feature_selection_pipeline()
