# Task 06 – Exploratory Data Analysis (EDA) Using Jupyter Notebook

> **PKCERT AI & Software Development Internship**
> **Total Marks: 100**

---

## 📁 Project Structure

```
Task06-Exploratory-Data-Analysis/
├── eda_stroke_prediction.ipynb       # Main Jupyter Notebook containing the full analysis and visualizations
├── healthcare-dataset-stroke-data.csv # Local copy of the Stroke Prediction dataset
└── README.md                          # Detailed documentation of findings and insights
```

---

## ⚙️ Requirements

```bash
pip install numpy pandas seaborn matplotlib ipykernel nbformat nbconvert
```

---

## 📊 Dataset Selection (Part A)

**Stroke Prediction Dataset** (originally from Kaggle, uploaded by user *fedesoriano*).

| Attribute | Value |
|-----------|-------|
| Source | Kaggle Stroke Prediction Dataset |
| Records | 5,110 patients |
| Features | 11 variables (demographics, clinical indicators, lifestyle) |
| Target Variable | `stroke` (0 = No, 1 = Yes) |

### Feature Reference Table

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `gender` | Categorical | "Male", "Female", or "Other" |
| `age` | Numerical | Age of the patient |
| `hypertension` | Binary | 0 if the patient doesn't have high blood pressure, 1 if they do |
| `heart_disease`| Binary | 0 if the patient doesn't have heart disease, 1 if they do |
| `ever_married` | Binary | "Yes" or "No" |
| `work_type` | Categorical | "children", "Govt_job", "Never_worked", "Private", or "Self-employed" |
| `Residence_type`| Categorical | "Rural" or "Urban" |
| `avg_glucose_level`| Numerical | Average glucose level in blood (mg/dL) |
| `bmi` | Numerical | Body Mass Index ($kg/m^2$) |
| `smoking_status`| Categorical | "formerly smoked", "never smoked", "smokes", or "Unknown" |

---

## 🧹 Data Preprocessing & Cleaning (Part B)

The following cleaning steps were executed programmatically within the notebook:
1. **Missing Value Imputation**: The feature `bmi` contained 201 missing values. Since `bmi` is right-skewed, these were imputed using the column's **median** (28.1) rather than the mean to prevent outlier bias.
2. **Handling Inconsistencies**: The `gender` column contained a single entry with the value `"Other"`. This record was removed to avoid noise and maintain binary representation in visualizations.
3. **Redundant Columns**: The `id` column (a unique random identifier) was dropped since it has no predictive or analytical value.
4. **Duplicate Verification**: The dataset was verified for duplicate rows (`df.duplicated().sum() == 0`), confirming no duplicate entries.

---

## 📈 Visualizations & Interpretations (Part B)

The Jupyter Notebook generates **six** distinct and meaningful visualizations using `Seaborn` and `Matplotlib`:

### 1. Class Distribution of Stroke Status
- **Plot Type**: Seaborn count plot.
- **Insight**: Highlights a severe class imbalance: ~4.9% of patients (249 records) had a stroke, while ~95.1% (4,860 records) did not. This informs future modeling choices (requires SMOTE, class weights, or down-sampling).

### 2. Age Distribution by Stroke Status
- **Plot Type**: Kernel Density Estimate (KDE) plot.
- **Insight**: Shows that stroke occurrence is closely linked to age. Risk is extremely low below age 50 and peaks between ages 70–80.

### 3. Glucose Level vs BMI Scatter Plot
- **Plot Type**: Scatter plot categorized by stroke status.
- **Insight**: Demonstrates that stroke occurrences are highly dense in patients with average glucose levels exceeding 150 mg/dL (indicating pre-diabetes/diabetes), regardless of BMI.

### 4. Correlation Matrix Heatmap
- **Plot Type**: Heatmap.
- **Insight**: Shows that `age` has the strongest linear correlation with `stroke` (+0.25), followed by `heart_disease` (+0.13) and `avg_glucose_level` (+0.13). Features have low multicollinearity, making them suitable for regression models.

### 5. Stroke Rate (%) by Co-morbidities
- **Plot Type**: Categorical bar plot.
- **Insight**: Patients with neither hypertension nor heart disease have a stroke rate of ~3%. The rate rises to over 8% for hypertension only, 11% for heart disease only, and spikes to **over 20%** for patients suffering from both conditions.

### 6. Stroke Rate (%) by Smoking Status
- **Plot Type**: Categorical bar plot.
- **Insight**: Former smokers display the highest stroke rate (~7.9%), followed by current smokers (~5.3%) and never-smokers (~4.7%). This indicates that long-term vascular damage from smoking persists even after cessation.

---

## 💡 Written Analysis & Insights (Part C)

### 1. Key Findings
- **Cardiovascular Compounding**: Co-morbidities do not simply add to stroke risk; they compound it. Having both hypertension and heart disease increases stroke likelihood by roughly 7 times compared to a healthy profile.
- **Age Dominance**: Age is the strongest indicator of stroke risk, meaning preventative cardiovascular healthcare should scale up aggressively starting from age 50.
- **Metabolic Indicators**: High blood glucose levels represent an independent risk factor for stroke, highlighting the need for managing diabetes to prevent cardiovascular events.

### 2. Research & Business Scope
- **Targeted Screening**: Clinics and hospitals can use these statistics to build automated patient-risk flags. A patient over 50 with high blood pressure and heart disease should trigger high-priority preventative clinical interventions.
- **Actuarial Risk Modeling**: Insurance providers can refine premium rates using the specific joint probabilities of stroke mapped in this analysis.

### 3. Dataset Limitations
- **Class Imbalance**: The low number of stroke positive cases (~4.9%) makes standard classification metrics like accuracy misleading.
- **High Percentage of Unknown Data**: The "Unknown" category in `smoking_status` accounts for ~30% of the dataset, which limits the reliability of smoking-related conclusions.
- **Omission of Key Features**: Crucial factors such as family history, physical activity levels, diet, alcohol usage, and blood pressure numbers (Systolic/Diastolic readings rather than binary flags) were not present.

---

## 🙋 Author

**Muhammad John Raza**  
PKCERT AI & Software Development Internship — Task 06  
GitHub Repository: [ai-internship-jkpkcert](https://github.com/MuhammadJohnRaza/ai-internship-jkpkcert)
