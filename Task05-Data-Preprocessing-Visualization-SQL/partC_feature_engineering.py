import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import os
import warnings

warnings.filterwarnings("ignore")

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 65)
print("  TASK 05 - Part C: Feature Engineering")
print("=" * 65)

if os.path.exists("titanic_cleaned.csv"):
    df = pd.read_csv("titanic_cleaned.csv")
    print("\n[LOAD] Loaded titanic_cleaned.csv")
else:
    df = sns.load_dataset("titanic")
    print("\n[LOAD] titanic_cleaned.csv not found - using raw seaborn dataset")

print(f"  Shape: {df.shape}")

print("\n-- Step 1: Identify Categorical Features --")
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"  Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"  Numerical  columns ({len(num_cols)}): {num_cols}")

print("\n-- Step 2: Derived Features --")
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"]    = (df["family_size"] == 1).astype(int)
df["age_group"]   = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 120],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
)
df["fare_band"] = pd.qcut(df["fare"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
print("  Added: family_size, is_alone, age_group (5 bins), fare_band (quartiles)")

print("\n-- Step 3: Label Encoding --")
le = LabelEncoder()
label_targets = ["sex", "embarked"]
df_encoded = df.copy()
for col in label_targets:
    df_encoded[col + "_le"] = le.fit_transform(df_encoded[col].astype(str))
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col:12s} -> {col}_le  |  Mapping: {mapping}")
print("\n  [Note] Label Encoding is suitable for binary/ordinal features.")
print("         'sex' (binary) and 'embarked' (nominal, low cardinality) demonstrated here.")

print("\n-- Step 4: One-Hot Encoding --")
ohe_targets = ["embarked", "age_group", "fare_band"]
df_ohe = pd.get_dummies(df_encoded, columns=ohe_targets, drop_first=True, dtype=int)
new_cols = [c for c in df_ohe.columns if c not in df_encoded.columns]
print(f"  Columns OHE-expanded: {ohe_targets}")
print(f"  New dummy columns created ({len(new_cols)}): {new_cols}")
print(f"  DataFrame shape after OHE: {df_ohe.shape}")
print("\n  [Note] One-Hot Encoding avoids imposing ordinal relationships on nominal categories.")
print("         drop_first=True prevents multicollinearity.")

print("\n-- Step 5: Normalization - Min-Max Scaling --")
scale_cols = ["age", "fare", "sibsp", "parch", "family_size"]
existing_scale = [c for c in scale_cols if c in df_ohe.columns]
mm_scaler = MinMaxScaler()
df_norm = df_ohe.copy()
df_norm[existing_scale] = mm_scaler.fit_transform(df_ohe[existing_scale].fillna(0))
print(f"  Columns normalised: {existing_scale}")
print(f"  Each value scaled to [0, 1] range.")
print(df_norm[existing_scale].describe().round(3).to_string())
print("\n  [Importance] Normalization ensures features with large ranges (e.g., fare: 0-512)")
print("               do not dominate distance-based or gradient-descent algorithms.")

print("\n-- Step 6: Standardization - Z-Score Scaling --")
ss_scaler = StandardScaler()
df_std = df_ohe.copy()
df_std[existing_scale] = ss_scaler.fit_transform(df_ohe[existing_scale].fillna(0))
print(f"  Columns standardised: {existing_scale}")
print(f"  Each value transformed to mean~0, std~1.")
print(df_std[existing_scale].describe().round(3).to_string())
print("\n  [Importance] Standardization is preferred when data follows a Gaussian distribution")
print("               or for algorithms like SVM, PCA, Logistic Regression.")

print("\n-- Generating Scaling Comparison Plot --")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
col_demo = "age"
axes[0].hist(df[col_demo].dropna(), bins=25, color="#3498db", edgecolor="white")
axes[0].set_title(f"Original '{col_demo}'", fontweight="bold")
axes[0].set_xlabel("Value")
axes[1].hist(df_norm[col_demo].dropna(), bins=25, color="#2ecc71", edgecolor="white")
axes[1].set_title(f"Min-Max Normalised '{col_demo}'", fontweight="bold")
axes[1].set_xlabel("Value [0, 1]")
axes[2].hist(df_std[col_demo].dropna(), bins=25, color="#e67e22", edgecolor="white")
axes[2].set_title(f"Z-Score Standardised '{col_demo}'", fontweight="bold")
axes[2].set_xlabel("Value (z-score)")
for ax in axes:
    ax.set_ylabel("Count")
plt.suptitle("Feature Scaling Comparison - Age Column", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/partC_01_scaling_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [SAVED] {PLOTS_DIR}/partC_01_scaling_comparison.png")

print("\n-- Encoded Column Distributions --")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["sex_le", "embarked_le", "is_alone"]):
    if col in df_ohe.columns:
        counts = df_ohe[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values,
               color=["#9b59b6", "#3498db", "#1abc9c"][:len(counts)])
        ax.set_title(f"'{col}' Distribution", fontweight="bold")
        ax.set_xlabel("Encoded Value")
        ax.set_ylabel("Count")
plt.suptitle("Label-Encoded & Binary Feature Distributions", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/partC_02_encoded_distributions.png", dpi=150)
plt.close()
print(f"  [SAVED] {PLOTS_DIR}/partC_02_encoded_distributions.png")

df_ohe.to_csv("titanic_features.csv", index=False)
print("\n  [SAVED] titanic_features.csv  (OHE-encoded, ready for modelling)")

print("\n[DONE] Part C complete.\n")
