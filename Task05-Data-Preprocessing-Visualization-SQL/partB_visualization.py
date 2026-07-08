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
print("  TASK 05 - Part B: Data Visualization")
print("=" * 65)

if os.path.exists("titanic_cleaned.csv"):
    df = pd.read_csv("titanic_cleaned.csv")
    print("\n[LOAD] Loaded titanic_cleaned.csv")
else:
    df = sns.load_dataset("titanic")
    print("\n[LOAD] titanic_cleaned.csv not found - using raw seaborn dataset")

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)

print("\n-- Plot 1: Histogram - Age Distribution --")
fig, ax = plt.subplots(figsize=(10, 5))
for val, label, color in zip([1, 0], ["Survived", "Not Survived"], ["#2ecc71", "#e74c3c"]):
    ax.hist(df[df["survived"] == val]["age"].dropna(), bins=30,
            alpha=0.65, label=label, color=color, edgecolor="white")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Count")
ax.set_title("Histogram - Age Distribution by Survival", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/partB_01_histogram_age.png", dpi=150)
plt.close()
print("  Insight: Children (<12) and young adults show higher survival; middle-aged men were most affected.")
print(f"  [SAVED] {PLOTS_DIR}/partB_01_histogram_age.png")

print("\n-- Plot 2: Boxplot - Fare by Passenger Class --")
fig, ax = plt.subplots(figsize=(9, 5))
df["pclass"] = df["pclass"].astype(int)
sns.boxplot(data=df, x="pclass", y="fare", palette="Set2",
            order=[1, 2, 3], ax=ax, linewidth=1.5)
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Fare (GBP, capped at 99th pct)")
ax.set_title("Boxplot - Fare Distribution by Passenger Class", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/partB_02_boxplot_fare_class.png", dpi=150)
plt.close()
print("  Insight: 1st class fares are significantly higher and more spread; 3rd class is tightly clustered near zero.")
print(f"  [SAVED] {PLOTS_DIR}/partB_02_boxplot_fare_class.png")

print("\n-- Plot 3: Correlation Heatmap --")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap - Numeric Features", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/partB_03_correlation_heatmap.png", dpi=150)
plt.close()
print("  Insight: 'fare' positively correlated with survival (r~+0.26); 'pclass' negatively correlated.")
print(f"  [SAVED] {PLOTS_DIR}/partB_03_correlation_heatmap.png")

print("\n-- Plot 4: Bar Chart - Survival Rate by Sex --")
surv_sex = df.groupby("sex")["survived"].mean().reset_index()
surv_sex["pct"] = surv_sex["survived"] * 100
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(surv_sex["sex"], surv_sex["pct"],
              color=["#e74c3c", "#3498db"], edgecolor="white", width=0.5)
for bar, pct in zip(bars, surv_sex["pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{pct:.1f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 100)
ax.set_ylabel("Survival Rate (%)")
ax.set_xlabel("Sex")
ax.set_title("Bar Chart - Survival Rate by Sex", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/partB_04_survival_by_sex.png", dpi=150)
plt.close()
print("  Insight: Female survival (~74%) was nearly 4x the male rate (~19%) - 'women and children first' policy.")
print(f"  [SAVED] {PLOTS_DIR}/partB_04_survival_by_sex.png")

print("\n-- Plot 5: Violin Plot - Age by Survival & Passenger Class --")
fig, ax = plt.subplots(figsize=(11, 6))
plot_df = df.copy()
plot_df["pclass_str"] = plot_df["pclass"].astype(str)
plot_df["survived_label"] = plot_df["survived"].map({1: "Survived", 0: "Not Survived"})
sns.violinplot(data=plot_df, x="pclass_str", y="age", hue="survived_label",
               split=True,
               palette={"Survived": "#2ecc71", "Not Survived": "#e74c3c"},
               inner="quartile", ax=ax, order=["1", "2", "3"])
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Age (years)")
ax.set_title("Violin Plot - Age Distribution by Class & Survival", fontweight="bold")
ax.legend(title="Outcome")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/partB_05_violin_age_class_survival.png", dpi=150)
plt.close()
print("  Insight: 1st class survivors skew older; 3rd class non-survivors concentrated in young working-age adults.")
print(f"  [SAVED] {PLOTS_DIR}/partB_05_violin_age_class_survival.png")

print("\n[DONE] Part B complete - 5 plots saved to", PLOTS_DIR, "\n")
