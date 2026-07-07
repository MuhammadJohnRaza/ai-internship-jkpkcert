# Task 04 – NumPy & Pandas for Data Analysis

> **PKCERT AI & Software Development Internship**
> **Total Marks: 100**

---

## 📁 Project Structure

```
Task04-NumPy-Pandas/
├── partA_numpy_fundamentals.py   # Part A – NumPy (40 marks)
├── partB_pandas_fundamentals.py  # Part B – Pandas (30 marks)
├── partC_titanic_eda.py          # Part C – Data Analysis Mini Project (30 marks)
├── plots/                        # Auto-generated EDA plots (Part C)
│   ├── 01_survival_count.png
│   ├── 02_survival_by_sex.png
│   ├── 03_survival_by_class.png
│   ├── 04_age_distribution.png
│   ├── 05_fare_distribution.png
│   ├── 06_survival_by_age_group.png
│   ├── 07_correlation_heatmap.png
│   ├── 08_family_size_survival.png
│   ├── 09_embarkation_analysis.png
│   └── 10_sex_class_heatmap.png
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install numpy pandas seaborn matplotlib scipy
```

> Python 3.8+ required.

---

## Part A – NumPy Fundamentals (40 Marks)

**File:** `partA_numpy_fundamentals.py`

### Topics Covered

| Section | Topic | Details |
|---------|-------|---------|
| 1 | Array Creation | `np.array`, `arange`, `linspace`, `zeros`, `ones`, `eye`, `random` — 1-D and multi-dimensional |
| 2 | Indexing / Slicing / Reshaping | Integer, boolean, fancy indexing; `.reshape()`, `.flatten()`, `.ravel()` |
| 2 | Math Operations | Element-wise arithmetic, `sqrt`, `log`, `cumsum`, `sum`, `mean`, `std` |
| 3 | Broadcasting | Scalar, 1-D over 2-D, column-vector, outer-product examples |
| 4 | Vectorized vs Loops | Timed comparison on 1M-element arrays (typically 50–200× speed-up) |
| 5 | Linear Algebra | Dot product, matrix multiplication (`@`), transpose, determinant, inverse, eigenvalues, linear system solve, vector norms |

### Run

```bash
python partA_numpy_fundamentals.py
```

---

## Part B – Pandas Fundamentals (30 Marks)

**File:** `partB_pandas_fundamentals.py`

### Topics Covered

| Section | Topic | Details |
|---------|-------|---------|
| 1 | Series & DataFrames | Creation from lists, dicts, NumPy arrays; dtype inspection; derived columns |
| 2 | Indexing & Selection | `.loc`, `.iloc`, boolean masking, `.query()`, `.isin()`, `.sort_values()` |
| 3 | GroupBy | Single/multi-key groupby, `.agg()`, `.transform()`, `.filter()`, `value_counts()` |
| 4 | Merge & Join | INNER, LEFT, RIGHT, OUTER joins; chained joins; `pd.concat`; `.join()` |

### Run

```bash
python partB_pandas_fundamentals.py
```

---

## Part C – Data Analysis Mini Project (30 Marks)

**File:** `partC_titanic_eda.py`

### Dataset

The **Titanic** dataset from the **seaborn** library (`sns.load_dataset("titanic")`).

| Attribute | Value |
|-----------|-------|
| Source | seaborn built-in (originally Kaggle / Encyclopedia Titanica) |
| Rows | 891 passengers |
| Columns | 15 (after cleaning: 14) |
| Target variable | `survived` (0 = No, 1 = Yes) |

**Key columns:**

| Column | Description |
|--------|-------------|
| `survived` | Survival indicator (0/1) |
| `pclass` | Passenger class (1st / 2nd / 3rd) |
| `sex` | Passenger sex |
| `age` | Age in years |
| `sibsp` | # siblings/spouses aboard |
| `parch` | # parents/children aboard |
| `fare` | Ticket fare (£) |
| `embarked` | Port of embarkation (C / Q / S) |

---

### Data Cleaning Process

| Issue | Action Taken | Reason |
|-------|-------------|--------|
| `age` — ~20% missing | Imputed with **grouped median** (by `pclass` × `sex`) | Preserves class/sex age patterns better than global median |
| `embarked` / `embark_town` — 2 missing | Filled with **mode** | Only 2 rows; safest imputation |
| `deck` — 77% missing | **Dropped** | Too sparse to impute reliably |
| `alive` — string duplicate of `survived` | **Dropped** | Redundant feature |

**Derived features added:**

| Feature | Formula | Purpose |
|---------|---------|---------|
| `family_size` | `sibsp + parch + 1` | Measures group size |
| `is_alone` | `family_size == 1` | Binary solo-traveller flag |
| `age_group` | pd.cut: Child/Teen/Adult/Middle-aged/Senior | Survival by life stage |
| `fare_band` | pd.qcut into 4 quartiles | Survival by fare tier |

---

### Analysis Performed

1. **Descriptive statistics** — mean, median, std, min/max for all numeric features
2. **Survival breakdown** — by sex, class, age group, family size, port of embarkation
3. **Distribution analysis** — age and fare histograms split by survival outcome
4. **Correlation analysis** — heatmap of all numeric features
5. **Cross-tabulation** — sex × class survival pivot heatmap
6. **Family size effect** — survival rate across every unique family size value

---

### Key Findings

| # | Finding |
|---|---------|
| 1 | **Overall survival rate: 38.4%** — fewer than 2-in-5 passengers survived |
| 2 | **Gender was the strongest predictor**: females 74.2% vs males 18.9% survival |
| 3 | **Class inequality**: 1st class 63.0% vs 3rd class 24.2% survival |
| 4 | **Children (<12)** had the highest survival among age groups (~58%) |
| 5 | **Mid-sized families** (2–4 members) survived at higher rates than solo or very large groups |
| 6 | **Cherbourg** (C) passengers had the highest survival rate (~55%) due to concentration of 1st-class travellers |
| 7 | **Fare** positively correlated with survival (r ≈ +0.26), reflecting class access to lifeboats |
| 8 | `deck` column had 77% missing values — a common real-world data quality challenge |

---

### Plots Generated (saved to `plots/`)

| File | Description |
|------|-------------|
| `01_survival_count.png` | Bar chart of survived vs not survived |
| `02_survival_by_sex.png` | Grouped bar: survival counts by sex |
| `03_survival_by_class.png` | Grouped bar: survival counts by class |
| `04_age_distribution.png` | Overlapping histograms of age by survival |
| `05_fare_distribution.png` | Fare distributions (log scale) by survival |
| `06_survival_by_age_group.png` | Survival rate % by age group |
| `07_correlation_heatmap.png` | Correlation matrix of all numeric features |
| `08_family_size_survival.png` | Line plot: survival rate vs family size |
| `09_embarkation_analysis.png` | Count + survival rate by boarding port |
| `10_sex_class_heatmap.png` | Pivot heatmap: sex × class survival rates |

### Run

```bash
python partC_titanic_eda.py
```

---

## Running All Parts

```bash
python partA_numpy_fundamentals.py
python partB_pandas_fundamentals.py
python partC_titanic_eda.py
```

---

## Author

**Muhammad John Raza**
PKCERT AI & Software Development Internship — Task 04
GitHub: [ai-internship-jkpkcert](https://github.com/MuhammadJohnRaza/ai-internship-jkpkcert)
