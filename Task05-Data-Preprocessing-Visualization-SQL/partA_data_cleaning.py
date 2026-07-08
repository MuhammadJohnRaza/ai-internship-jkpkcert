import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 65)
print("  TASK 05 - Part A: Data Cleaning & Preprocessing")
print("  Dataset: Titanic (seaborn built-in)")
print("=" * 65)

df = sns.load_dataset("titanic")
print(f"\n[LOAD]  Shape: {df.shape}  |  Columns: {list(df.columns)}")

print("\n-- Step 1: Initial Inspection --")
print(df.dtypes)
print("\nNull counts:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDescriptive stats:")
print(df.describe(include="all").T.to_string())

before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n-- Step 2: Duplicates Removed --")
print(f"  Rows before: {before}  |  Rows after: {len(df)}")

print("\n-- Step 3: Handling Missing Values --")

age_miss = df["age"].isnull().sum()
df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
    lambda x: x.fillna(x.median())
)
print(f"  age:      {age_miss} missing -> filled with grouped median (pclass x sex)")

emb_miss = df["embarked"].isnull().sum()
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
df["embark_town"].fillna(df["embark_town"].mode()[0], inplace=True)
print(f"  embarked: {emb_miss} missing -> filled with mode '{df['embarked'].mode()[0]}'")

deck_miss = df["deck"].isnull().sum()
df.drop(columns=["deck"], inplace=True)
print(f"  deck:     {deck_miss} missing ({deck_miss/891*100:.1f}%) -> column dropped (too sparse)")

df.drop(columns=["alive"], inplace=True)
print("  alive:    dropped (string duplicate of 'survived')")

print("\nRemaining nulls:\n", df.isnull().sum())

print("\n-- Step 4: Outlier Detection --")

def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in ["age", "fare", "sibsp", "parch"]:
    low, high = iqr_bounds(df[col])
    outliers = df[(df[col] < low) | (df[col] > high)].shape[0]
    print(f"  {col:8s}  IQR bounds: [{low:.2f}, {high:.2f}]  Outliers: {outliers}")

fare_cap = df["fare"].quantile(0.99)
df["fare"] = df["fare"].clip(upper=fare_cap)
print(f"\n  fare capped at 99th-percentile ({fare_cap:.2f}) to reduce skew")

print("\n-- Step 5: Data Type Corrections --")
df["pclass"] = df["pclass"].astype("category")
df["survived"] = df["survived"].astype(int)
print("  pclass -> category  |  survived -> int")

print("\n-- Step 6: Final Cleaned Dataset --")
print(f"  Shape: {df.shape}")
print(f"  Nulls remaining: {df.isnull().sum().sum()}")
print(df.dtypes)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(
    sns.load_dataset("titanic").isnull(),
    cbar=False, yticklabels=False, cmap="viridis", ax=axes[0]
)
axes[0].set_title("Before Cleaning - Missing Values", fontweight="bold")
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=axes[1])
axes[1].set_title("After Cleaning - Missing Values", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/partA_01_missing_heatmap.png", dpi=150)
plt.close()
print(f"\n  [SAVED] {PLOTS_DIR}/partA_01_missing_heatmap.png")

df.to_csv("titanic_cleaned.csv", index=False)
print("  [SAVED] titanic_cleaned.csv")

print("\n[DONE] Part A complete.\n")
