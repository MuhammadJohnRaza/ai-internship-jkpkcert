# Task 05 – Data Preprocessing, Visualization & SQL Fundamentals

> **PKCERT AI & Software Development Internship**
> **Total Marks: 100**

---

## 📁 Project Structure

```
Task05-Data-Preprocessing-Visualization-SQL/
├── partA_data_cleaning.py       # Part A – Data Cleaning & Preprocessing (30 marks)
├── partB_visualization.py       # Part B – Data Visualization (25 marks)
├── partC_feature_engineering.py # Part C – Feature Engineering (20 marks)
├── partD_sql_fundamentals.py    # Part D – SQL Fundamentals with Python (25 marks)
├── titanic_cleaned.csv          # Auto-generated: cleaned dataset (after Part A)
├── titanic_features.csv         # Auto-generated: feature-engineered dataset (after Part C)
├── titanic_db.sqlite            # Auto-generated: relational database (after Part D)
├── plots/                       # Auto-generated visualizations
│   ├── partA_01_missing_heatmap.png
│   ├── partB_01_histogram_age.png
│   ├── partB_02_boxplot_fare_class.png
│   ├── partB_03_correlation_heatmap.png
│   ├── partB_04_survival_by_sex.png
│   ├── partB_05_violin_age_class_survival.png
│   ├── partC_01_scaling_comparison.png
│   └── partC_02_encoded_distributions.png
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

> Python 3.8+ required. `sqlite3` is part of the Python standard library — no extra install needed.

---

## ▶️ Running All Parts (in order)

```bash
python partA_data_cleaning.py
python partB_visualization.py
python partC_feature_engineering.py
python partD_sql_fundamentals.py
```

> **Part A must run first** — it generates `titanic_cleaned.csv` which Parts B, C, and D consume.
> All other parts can detect the file and fall back gracefully if it is absent.

---

## Dataset

**Titanic** — loaded via `seaborn.load_dataset("titanic")` (originally Kaggle / Encyclopedia Titanica).

| Attribute | Value |
|-----------|-------|
| Source | Seaborn built-in (Kaggle Titanic) |
| Rows | 891 passengers |
| Original columns | 15 |
| Target variable | `survived` (0 = No, 1 = Yes) |

---

## Part A – Data Cleaning & Preprocessing (30 Marks)

**File:** `partA_data_cleaning.py`

### Preprocessing Steps

| Step | Issue | Action | Reason |
|------|-------|--------|--------|
| 1 | Initial inspection | `df.dtypes`, `isnull()`, `describe()` | Baseline audit |
| 2 | Duplicate rows | `drop_duplicates()` | Prevent data leakage / bias |
| 3 | `age` — ~20% missing | Imputed with **grouped median** (pclass × sex) | Preserves demographic age patterns better than global median |
| 3 | `embarked` — 2 missing | Filled with **mode** | Only 2 rows; safest option |
| 3 | `deck` — 77% missing | **Dropped** column | Too sparse to impute reliably |
| 3 | `alive` — string duplicate | **Dropped** column | Redundant with `survived` |
| 4 | `fare` outliers | **Capped** at 99th percentile | Reduces right-skew; retains data points |
| 5 | Data types | `pclass` → category, `survived` → int | Correct types for downstream analysis |

### Key Finding
After cleaning: **0 null values remain** across all retained columns.

---

## Part B – Data Visualization (25 Marks)

**File:** `partB_visualization.py`

| # | Plot Type | File | Insight |
|---|-----------|------|---------|
| 1 | **Histogram** | `partB_01_histogram_age.png` | Children & young adults had higher survival; middle-aged men were most affected |
| 2 | **Boxplot** | `partB_02_boxplot_fare_class.png` | 1st class fares are significantly higher and more spread; 3rd class tightly clustered near zero |
| 3 | **Correlation Heatmap** | `partB_03_correlation_heatmap.png` | `fare` positively correlated with survival (r≈+0.26); `pclass` negatively correlated |
| 4 | **Bar Chart** | `partB_04_survival_by_sex.png` | Female survival ~74% vs male ~19% — "women and children first" policy evident |
| 5 | **Violin Plot** | `partB_05_violin_age_class_survival.png` | 1st class survivors skew older; 3rd class non-survivors concentrated in young working-age adults |

---

## Part C – Feature Engineering (20 Marks)

**File:** `partC_feature_engineering.py`

### Categorical Features Identified

| Column | Type | Unique Values |
|--------|------|---------------|
| `sex` | Binary nominal | male, female |
| `embarked` | Nominal | C, Q, S |
| `who` | Nominal | man, woman, child |
| `pclass` | Ordinal | 1, 2, 3 |
| `adult_male` | Binary | True, False |

### Encoding Applied

| Technique | Columns | Rationale |
|-----------|---------|-----------|
| **Label Encoding** | `sex`, `embarked` | Demonstrated for binary/low-cardinality features; compact representation |
| **One-Hot Encoding** | `embarked`, `age_group`, `fare_band` | Prevents false ordinal assumption for nominal categories; `drop_first=True` avoids multicollinearity |

### Derived Features Added

| Feature | Formula | Purpose |
|---------|---------|---------|
| `family_size` | `sibsp + parch + 1` | Measures group travel size |
| `is_alone` | `family_size == 1` | Binary solo-traveller flag |
| `age_group` | pd.cut (Child/Teen/Young Adult/Adult/Senior) | Enables survival analysis by life stage |
| `fare_band` | pd.qcut into 4 quartiles | Survival analysis by spending tier |

### Scaling Comparison

| Technique | Formula | Range | Best For |
|-----------|---------|-------|----------|
| **Min-Max Normalization** | `(x - min) / (max - min)` | [0, 1] | KNN, Neural Networks, image data |
| **Z-Score Standardization** | `(x - mean) / std` | (−∞, +∞), mean=0 | SVM, PCA, Logistic Regression |

> **Why scale?** Features with large ranges (e.g., `fare`: 0–512) would dominate algorithms that rely on distances or gradients, making scaling essential for fair model training.

---

## Part D – SQL Fundamentals with Python (25 Marks)

**File:** `partD_sql_fundamentals.py`

### Relational Database Schema

```
embarkation_ports          class_info
─────────────────          ──────────
port_code (PK)             pclass (PK)
port_name                  class_name
country                    description
      │                         │
      └──────────┐    ┌─────────┘
                 ▼    ▼
              passengers
              ────────────────────
              passenger_id (PK)
              survived
              pclass        → FK → class_info
              sex
              age
              sibsp / parch
              fare
              embarked      → FK → embarkation_ports
              who
              adult_male
              family_size
              is_alone
```

### SQL Queries Demonstrated

| # | Query Type | Description |
|---|-----------|-------------|
| Q1 | `SELECT … LIMIT` | First 10 passengers — basic column selection |
| Q2 | `SELECT … WHERE` | Female 1st-class survivors paying fare > 50 |
| Q3 | `GROUP BY` (single key) | Survival stats per passenger class |
| Q4 | `GROUP BY` (multi-key) | Survival rate by sex × class cross-tab |
| Q5 | `JOIN` | Passengers joined to `embarkation_ports` — survival by port |
| Q6 | `JOIN` | Passengers joined to `class_info` — enriched class statistics |
| Q7 | `GROUP BY` | Survival rate vs family size |
| Q8 | Subquery | Passengers who paid above-average fare |

### Key SQL Findings

| Finding | Value |
|---------|-------|
| Overall survival rate | **38.4%** |
| 1st class survival | **63.0%** |
| 3rd class survival | **24.2%** |
| Female survival | **74.2%** |
| Male survival | **18.9%** |
| Best embarkation port | **Cherbourg (55%+)** — concentration of 1st-class travellers |
| Optimal family size for survival | **2–4 members** — mid-size groups fared best |

---

## Author

**Muhammad John Raza**
PKCERT AI & Software Development Internship — Task 05
GitHub: [ai-internship-jkpkcert](https://github.com/MuhammadJohnRaza/ai-internship-jkpkcert)
