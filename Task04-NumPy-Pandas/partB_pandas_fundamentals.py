"""
========================================================
PKCERT AI & Software Development Internship
Task 04 - Part B: Pandas Fundamentals (30 Marks)
========================================================
Topics:
  1. Creating and Manipulating Series & DataFrames
  2. Indexing, Filtering, Sorting, and Selection
  3. GroupBy Operations
  4. Merging and Joining DataFrames
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 100)

# ─────────────────────────────────────────────────────
# SECTION 1: Series & DataFrames
# ─────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: Creating & Manipulating Series and DataFrames")
print("=" * 65)

# Series from list
scores = pd.Series([88, 92, 75, 60, 95], name="Exam_Score")
print("\n[Series from list]\n", scores)

# Series from dict
population = pd.Series(
    {"Lahore": 14_849_000, "Karachi": 16_093_000,
     "Islamabad": 1_014_825, "Peshawar": 4_269_079},
    name="Population"
)
print("\n[Series from dict - City Populations]\n", population)
print(f"  mean : {population.mean():,.0f}")

# DataFrame from dict
students = pd.DataFrame({
    "Name"   : ["Ali", "Sara", "Ahmed", "Zara", "Bilal", "Nadia"],
    "Age"    : [21, 22, 20, 23, 21, 22],
    "Major"  : ["CS", "EE", "CS", "Math", "EE", "CS"],
    "GPA"    : [3.5, 3.8, 2.9, 3.6, 3.1, 3.9],
    "Passed" : [True, True, False, True, True, True]
})
print("\n[DataFrame - Student Records]\n", students)
print(f"\n  Shape: {students.shape}  |  dtypes:\n{students.dtypes}")

# Add derived columns
students["Grade"] = students["GPA"].apply(
    lambda g: "A" if g >= 3.7 else ("B" if g >= 3.3 else "C"))
students["Scholarship"] = students["GPA"] >= 3.8
print("\n[After adding Grade & Scholarship columns]\n", students)

# DataFrame from NumPy array
np_data = np.random.randint(50, 100, size=(5, 3))
df_np   = pd.DataFrame(np_data, columns=["Math", "Physics", "Chemistry"])
df_np.index = [f"Student_{i+1}" for i in range(5)]
print("\n[DataFrame from NumPy array]\n", df_np)

# ─────────────────────────────────────────────────────
# SECTION 2: Indexing, Filtering, Sorting & Selection
# ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: Indexing, Filtering, Sorting & Selection")
print("=" * 65)

df = students.copy()

# .loc
print("\n[.loc - rows 1-3, Name+GPA]\n",
      df.loc[1:3, ["Name", "GPA"]])

# .iloc
print("\n[.iloc - last 2 rows, first 2 cols]\n",
      df.iloc[-2:, :2])

# Boolean filter
print("\n[GPA > 3.5]\n",
      df[df["GPA"] > 3.5][["Name", "Major", "GPA"]])

print("\n[CS students with GPA >= 3.5]\n",
      df[(df["Major"] == "CS") & (df["GPA"] >= 3.5)][["Name", "Major", "GPA"]])

# .query()
print("\n[.query('Age < 22 and GPA > 3.0')]\n",
      df.query("Age < 22 and GPA > 3.0")[["Name", "Age", "GPA"]])

# isin()
print("\n[.isin(['EE','Math'])]\n",
      df[df["Major"].isin(["EE", "Math"])][["Name", "Major", "GPA"]])

# Sorting
print("\n[Sort by GPA descending]\n",
      df.sort_values("GPA", ascending=False)[["Name", "Major", "GPA"]])

print("\n[Sort by Major then GPA]\n",
      df.sort_values(["Major", "GPA"], ascending=[True, False])[["Name", "Major", "GPA"]])

# describe
print("\n[describe()]\n", df.describe())

# ─────────────────────────────────────────────────────
# SECTION 3: GroupBy Operations
# ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: GroupBy Operations")
print("=" * 65)

records = {
    "Name"       : ["Ali","Sara","Ahmed","Zara","Bilal","Nadia","Omar","Hina"],
    "Department" : ["CS","EE","CS","Math","EE","CS","Math","EE"],
    "Year"       : [2, 3, 2, 3, 2, 3, 2, 3],
    "GPA"        : [3.5, 3.8, 2.9, 3.6, 3.1, 3.9, 3.4, 3.7],
    "Projects"   : [2, 4, 1, 3, 2, 5, 2, 4],
}
df2 = pd.DataFrame(records)

print("\n[Mean GPA by Department]\n",
      df2.groupby("Department")["GPA"].mean().round(2))

dept_stats = df2.groupby("Department").agg(
    Avg_GPA=("GPA","mean"),
    Max_GPA=("GPA","max"),
    Min_GPA=("GPA","min"),
    Total_Projects=("Projects","sum"),
    Count=("Name","count")
).round(2)
print("\n[Department Statistics]\n", dept_stats)

print("\n[GPA by Year x Department]\n",
      df2.groupby(["Year","Department"])["GPA"].mean().round(2))

df2["Dept_Avg"] = df2.groupby("Department")["GPA"].transform("mean").round(3)
df2["Above_Avg"] = df2["GPA"] > df2["Dept_Avg"]
print("\n[With Dept Avg & Above_Avg flag]\n", df2)

big_depts = df2.groupby("Department").filter(lambda g: len(g) >= 3)
print("\n[Departments with >=3 members]\n",
      big_depts[["Name","Department","GPA"]])

print("\n[Student count per Department]\n", df2["Department"].value_counts())

# ─────────────────────────────────────────────────────
# SECTION 4: Merging and Joining
# ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: Merging and Joining DataFrames")
print("=" * 65)

students_df = pd.DataFrame({
    "StudentID": [101, 102, 103, 104, 105],
    "Name"     : ["Ali","Sara","Ahmed","Zara","Bilal"],
    "DeptID"   : [1, 2, 1, 3, 2],
})
departments_df = pd.DataFrame({
    "DeptID"  : [1, 2, 3, 4],
    "DeptName": ["Computer Science","Electrical Engineering",
                 "Mathematics","Physics"],
})
grades_df = pd.DataFrame({
    "StudentID": [101, 102, 103, 104, 106],
    "GPA"      : [3.5, 3.8, 2.9, 3.6, 3.2],
})

print("\n[students_df]\n", students_df)
print("\n[departments_df]\n", departments_df)
print("\n[grades_df]\n", grades_df)

# INNER JOIN
inner = pd.merge(students_df, grades_df, on="StudentID", how="inner")
print("\n[INNER JOIN - only matching StudentIDs]\n", inner)
print("  -> StudentID 105 (no grade) and 106 (no student) excluded.")

# LEFT JOIN
left = pd.merge(students_df, grades_df, on="StudentID", how="left")
print("\n[LEFT JOIN - all students, GPA=NaN if missing]\n", left)

# RIGHT JOIN
right = pd.merge(students_df, grades_df, on="StudentID", how="right")
print("\n[RIGHT JOIN - all grade rows, Name=NaN if not found]\n", right)

# OUTER JOIN
outer = pd.merge(students_df, grades_df, on="StudentID", how="outer")
print("\n[OUTER JOIN - union of both, NaN where no match]\n", outer)

# Chain join (students + departments + grades)
full = (students_df
        .merge(departments_df, on="DeptID", how="left")
        .merge(grades_df, on="StudentID", how="left"))
print("\n[CHAIN JOIN - students + departments + grades]\n",
      full[["StudentID","Name","DeptName","GPA"]])

# pd.concat (stacking rows)
batch_a = students_df[students_df["StudentID"] <= 102]
batch_b = students_df[students_df["StudentID"] >= 103]
stacked = pd.concat([batch_a, batch_b], ignore_index=True)
print("\n[pd.concat rows - stacking two batches]\n", stacked)

# .join() (index-based)
df_a = students_df.set_index("StudentID")[["Name"]]
df_b = grades_df.set_index("StudentID")[["GPA"]]
joined = df_a.join(df_b, how="left")
print("\n[DataFrame.join() - index-based left join]\n", joined)

print("\n" + "=" * 65)
print("Part B - Pandas Fundamentals: COMPLETE")
print("=" * 65)
