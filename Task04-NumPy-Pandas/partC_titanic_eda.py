

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")                                                   
import matplotlib.pyplot as plt
import os

                                                        
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

                                                        
print("=" * 60)
print("STEP 1: Loading Titanic Dataset (seaborn)")
print("=" * 60)

df = sns.load_dataset("titanic")
print(f"\n  Shape         : {df.shape}")
print(f"  Columns       : {df.columns.tolist()}")
print(f"\n[First 5 rows]\n{df.head()}")
print(f"\n[Data types]\n{df.dtypes}")

                                                        
print("\n" + "=" * 60)
print("STEP 2: Missing Value Analysis & Data Cleaning")
print("=" * 60)

missing = df.isnull().sum()
pct     = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "Pct%": pct})
missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)
print(f"\n[Missing values before cleaning]\n{missing_df}")

                                                        
df["age"] = df.groupby(["pclass","sex"])["age"].transform(
    lambda x: x.fillna(x.median())
)

                                                  
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
df["embark_town"].fillna(df["embark_town"].mode()[0], inplace=True)

                                                                         
df.drop(columns=["deck"], inplace=True)

                                                               
df.drop(columns=["alive"], inplace=True)

           
remaining = df.isnull().sum()
remaining = remaining[remaining > 0]
print(f"\n[Missing values after cleaning]\n{remaining if len(remaining) else '  None -- dataset is clean!'}")
print(f"\n  Final shape   : {df.shape}")

                                                         
df["survived"] = df["survived"].astype(int)
df["pclass"]   = df["pclass"].astype("category")
df["sex"]      = df["sex"].astype("category")
df["embarked"] = df["embarked"].astype("category")

                                                        
df["family_size"]  = df["sibsp"] + df["parch"] + 1
df["is_alone"]     = (df["family_size"] == 1).astype(int)
df["age_group"]    = pd.cut(df["age"],
                             bins=[0, 12, 17, 35, 60, 100],
                             labels=["Child","Teen","Adult","Middle-aged","Senior"])
df["fare_band"]    = pd.qcut(df["fare"], q=4,
                              labels=["Low","Mid","High","Premium"])

print(f"\n[New derived features added]: family_size, is_alone, age_group, fare_band")

                                                        
print("\n" + "=" * 60)
print("STEP 3: Summary Statistics")
print("=" * 60)

print("\n[Overall describe (numeric)]\n", df.describe().round(2))
print("\n[Categorical describe]\n", df.describe(include="category"))

               
surv_rate = df["survived"].mean() * 100
print(f"\n  Overall survival rate : {surv_rate:.1f}%")
print(f"  Total passengers      : {len(df)}")
print(f"  Survivors             : {df['survived'].sum()}")
print(f"  Non-survivors         : {(1-df['survived']).sum()}")

        
print("\n[Survival rate by Sex]\n",
      df.groupby("sex")["survived"].agg(["sum","mean","count"])
        .rename(columns={"sum":"Survived","mean":"Rate","count":"Total"})
        .assign(Rate=lambda x: (x["Rate"]*100).round(1)))

          
print("\n[Survival rate by Passenger Class]\n",
      df.groupby("pclass", observed=True)["survived"].agg(["sum","mean","count"])
        .rename(columns={"sum":"Survived","mean":"Rate","count":"Total"})
        .assign(Rate=lambda x: (x["Rate"]*100).round(1)))

              
print("\n[Survival rate by Age Group]\n",
      df.groupby("age_group", observed=True)["survived"].agg(["sum","mean","count"])
        .rename(columns={"sum":"Survived","mean":"Rate","count":"Total"})
        .assign(Rate=lambda x: (x["Rate"]*100).round(1)))

                     
print("\n[Fare Statistics by Class]\n",
      df.groupby("pclass", observed=True)["fare"]
        .agg(["min","mean","median","max"]).round(2))

                                                        
print("\n" + "=" * 60)
print("STEP 4: Exploratory Data Analysis (EDA)")
print("=" * 60)

sns.set_theme(style="darkgrid", palette="muted")
BLUE, RED = "#4C72B0", "#C44E52"

                                                         
fig, ax = plt.subplots(figsize=(6, 4))
survived_counts = df["survived"].value_counts().sort_index()
bars = ax.bar(["Did Not Survive", "Survived"],
              survived_counts.values,
              color=[RED, BLUE], edgecolor="white", width=0.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center", fontweight="bold")
ax.set_title("Passenger Survival Count", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Passengers")
save(fig, "01_survival_count.png")

                                                        
fig, ax = plt.subplots(figsize=(7, 4))
sex_surv = df.groupby(["sex","survived"], observed=True).size().unstack()
sex_surv.columns = ["Did Not Survive", "Survived"]
sex_surv.plot(kind="bar", ax=ax, color=[RED, BLUE],
              edgecolor="white", rot=0)
ax.set_title("Survival by Sex", fontsize=14, fontweight="bold")
ax.set_xlabel("Sex")
ax.set_ylabel("Count")
ax.legend(title="Outcome")
save(fig, "02_survival_by_sex.png")

                                                       
fig, ax = plt.subplots(figsize=(7, 4))
class_surv = df.groupby(["pclass","survived"], observed=True).size().unstack()
class_surv.columns = ["Did Not Survive", "Survived"]
class_surv.plot(kind="bar", ax=ax, color=[RED, BLUE],
                edgecolor="white", rot=0)
ax.set_title("Survival by Passenger Class", fontsize=14, fontweight="bold")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Count")
ax.legend(title="Outcome")
save(fig, "03_survival_by_class.png")

                                                       
fig, ax = plt.subplots(figsize=(9, 4))
for surv, color, label in [(0, RED, "Did Not Survive"), (1, BLUE, "Survived")]:
    ax.hist(df[df["survived"]==surv]["age"].dropna(),
            bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
ax.axvline(df[df["survived"]==0]["age"].median(), color=RED,
           linestyle="--", linewidth=1.5,
           label=f"Non-surv median ({df[df['survived']==0]['age'].median():.0f})")
ax.axvline(df[df["survived"]==1]["age"].median(), color=BLUE,
           linestyle="--", linewidth=1.5,
           label=f"Surv median ({df[df['survived']==1]['age'].median():.0f})")
ax.set_title("Age Distribution by Survival", fontsize=14, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()
save(fig, "04_age_distribution.png")

                                                       
fig, ax = plt.subplots(figsize=(9, 4))
for surv, color, label in [(0, RED, "Did Not Survive"), (1, BLUE, "Survived")]:
    data = df[df["survived"]==surv]["fare"].dropna() + 0.1
    ax.hist(data, bins=40, alpha=0.6, color=color,
            label=label, edgecolor="white", log=True)
ax.set_title("Fare Distribution by Survival (log scale)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Fare (£)")
ax.set_ylabel("Count (log)")
ax.legend()
save(fig, "05_fare_distribution.png")

                                                       
fig, ax = plt.subplots(figsize=(9, 4))
age_rate = (df.groupby("age_group", observed=True)["survived"].mean() * 100).round(1)
bars = ax.bar(age_rate.index.astype(str), age_rate.values,
              color=BLUE, edgecolor="white")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title("Survival Rate by Age Group", fontsize=14, fontweight="bold")
ax.set_xlabel("Age Group")
ax.set_ylabel("Survival Rate (%)")
ax.set_ylim(0, 80)
save(fig, "06_survival_by_age_group.png")

                                                        
fig, ax = plt.subplots(figsize=(9, 6))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap (Numeric Features)",
             fontsize=14, fontweight="bold")
save(fig, "07_correlation_heatmap.png")

                                                       
fig, ax = plt.subplots(figsize=(9, 4))
fam_rate = (df.groupby("family_size")["survived"].mean() * 100).round(1)
ax.plot(fam_rate.index, fam_rate.values,
        marker="o", color=BLUE, linewidth=2, markersize=8)
for x, y in zip(fam_rate.index, fam_rate.values):
    ax.text(x, y + 1.5, f"{y:.0f}%", ha="center", fontsize=8)
ax.set_title("Survival Rate by Family Size", fontsize=14, fontweight="bold")
ax.set_xlabel("Family Size (self + siblings + parents/children)")
ax.set_ylabel("Survival Rate (%)")
ax.set_xticks(fam_rate.index)
ax.set_ylim(0, 100)
save(fig, "08_family_size_survival.png")

                                                       
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
embark_count = df.groupby(["embarked","survived"], observed=True).size().unstack(fill_value=0)
embark_rate  = (df.groupby("embarked", observed=True)["survived"].mean() * 100).round(1)

embark_count.columns = ["Did Not Survive","Survived"]
embark_count.plot(kind="bar", ax=axes[0], color=[RED, BLUE],
                  edgecolor="white", rot=0)
axes[0].set_title("Passenger Count by Port")
axes[0].set_xlabel("Port (C=Cherbourg, Q=Queenstown, S=Southampton)")

embark_rate.plot(kind="bar", ax=axes[1], color=BLUE, edgecolor="white", rot=0)
for i, v in enumerate(embark_rate.values):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
axes[1].set_title("Survival Rate by Port")
axes[1].set_xlabel("Port")
axes[1].set_ylabel("Survival Rate (%)")
axes[1].set_ylim(0, 80)
fig.suptitle("Embarkation Port Analysis", fontsize=14, fontweight="bold")
save(fig, "09_embarkation_analysis.png")

                                                        
fig, ax = plt.subplots(figsize=(7, 4))
pivot = df.pivot_table(values="survived", index="sex",
                        columns="pclass", aggfunc="mean",
                        observed=True) * 100
sns.heatmap(pivot.round(1), annot=True, fmt=".1f",
            cmap="Blues", ax=ax, linewidths=0.5,
            cbar_kws={"label": "Survival Rate (%)"})
ax.set_title("Survival Rate (%) - Sex x Passenger Class",
             fontsize=14, fontweight="bold")
save(fig, "10_sex_class_heatmap.png")

                                                        
print("\n" + "=" * 60)
print("STEP 5: Key Findings Summary")
print("=" * 60)

female_surv = df[df["sex"]=="female"]["survived"].mean()*100
male_surv   = df[df["sex"]=="male"]["survived"].mean()*100
p1_surv = df[df["pclass"]==1]["survived"].mean()*100
p3_surv = df[df["pclass"]==3]["survived"].mean()*100
child_surv  = df[df["age_group"]=="Child"]["survived"].mean()*100
senior_surv = df[df["age_group"]=="Senior"]["survived"].mean()*100
alone_surv  = df[df["is_alone"]==1]["survived"].mean()*100
family_surv = df[df["is_alone"]==0]["survived"].mean()*100

print(f"""
  1. Overall Survival Rate    : {surv_rate:.1f}%
     (Less than 1-in-2 survived the disaster)

  2. Gender Gap (most critical factor):
     - Females survived at {female_surv:.1f}% vs Males at {male_surv:.1f}%
     - "Women and children first" policy clearly reflected in data

  3. Class Inequality:
     - 1st-class passengers: {p1_surv:.1f}% survival rate
     - 3rd-class passengers: {p3_surv:.1f}% survival rate
     - Proximity to lifeboats and evacuation priority differed by class

  4. Age:
     - Children (<12): {child_surv:.1f}% survival rate
     - Seniors (>60) : {senior_surv:.1f}% survival rate

  5. Family Size matters:
     - Travelling alone   : {alone_surv:.1f}% survival
     - Travelling w/ family: {family_surv:.1f}% survival
     - Very large families (>5) had lower survival odds

  6. Embarkation Port:
     - Cherbourg (C) passengers had the highest survival rate
       (linked to higher proportion of 1st-class passengers)

  7. Fare & Class correlation:
     - Higher fare strongly correlated with higher class
     - Higher class -> higher survival (Pearson r ~= {corr['survived']['fare']:.2f} for fare)

  8. Data Quality:
     - 'deck' column (77% missing) was dropped
     - 'age' (~20% missing) imputed using grouped median
       (grouped by pclass x sex for accuracy)
""")

print("=" * 60)
print(f"Part C - Titanic EDA: COMPLETE  |  Plots saved to: {PLOT_DIR}")
print("=" * 60)
