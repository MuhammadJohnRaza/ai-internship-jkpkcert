import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3
import os
import warnings

warnings.filterwarnings("ignore")

print("=" * 65)
print("  TASK 05 - Part D: SQL Fundamentals with Python")
print("=" * 65)

DB_PATH = "titanic_db.sqlite"

if os.path.exists("titanic_cleaned.csv"):
    df = pd.read_csv("titanic_cleaned.csv")
    print("\n[LOAD] Loaded titanic_cleaned.csv")
else:
    df = sns.load_dataset("titanic")
    print("\n[LOAD] titanic_cleaned.csv not found - using raw seaborn dataset")

df["family_size"]  = df["sibsp"] + df["parch"] + 1
df["is_alone"]     = (df["family_size"] == 1).astype(int)
df["passenger_id"] = range(1, len(df) + 1)

print("\n-- Step 2: Creating SQLite Relational Database --")

conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.executescript("""
DROP TABLE IF EXISTS passengers;
DROP TABLE IF EXISTS embarkation_ports;
DROP TABLE IF EXISTS class_info;

CREATE TABLE embarkation_ports (
    port_code   TEXT PRIMARY KEY,
    port_name   TEXT NOT NULL,
    country     TEXT NOT NULL
);

CREATE TABLE class_info (
    pclass      INTEGER PRIMARY KEY,
    class_name  TEXT NOT NULL,
    description TEXT
);

CREATE TABLE passengers (
    passenger_id  INTEGER PRIMARY KEY,
    survived      INTEGER,
    pclass        INTEGER,
    sex           TEXT,
    age           REAL,
    sibsp         INTEGER,
    parch         INTEGER,
    fare          REAL,
    embarked      TEXT,
    who           TEXT,
    adult_male    INTEGER,
    family_size   INTEGER,
    is_alone      INTEGER,
    FOREIGN KEY (embarked) REFERENCES embarkation_ports(port_code),
    FOREIGN KEY (pclass)   REFERENCES class_info(pclass)
);
""")

ports_data = [
    ("C", "Cherbourg",   "France"),
    ("Q", "Queenstown",  "Ireland"),
    ("S", "Southampton", "England"),
]
cursor.executemany("INSERT INTO embarkation_ports VALUES (?, ?, ?)", ports_data)

class_data = [
    (1, "First Class",  "Luxury cabins on upper decks"),
    (2, "Second Class", "Comfortable mid-ship cabins"),
    (3, "Third Class",  "Dormitory-style lower-deck cabins"),
]
cursor.executemany("INSERT INTO class_info VALUES (?, ?, ?)", class_data)

passengers_rows = []
for _, row in df.iterrows():
    passengers_rows.append((
        int(row["passenger_id"]),
        int(row["survived"])   if pd.notna(row.get("survived"))   else None,
        int(row["pclass"])     if pd.notna(row.get("pclass"))     else None,
        str(row["sex"])        if pd.notna(row.get("sex"))        else None,
        float(row["age"])      if pd.notna(row.get("age"))        else None,
        int(row["sibsp"])      if pd.notna(row.get("sibsp"))      else None,
        int(row["parch"])      if pd.notna(row.get("parch"))      else None,
        float(row["fare"])     if pd.notna(row.get("fare"))       else None,
        str(row["embarked"])   if pd.notna(row.get("embarked"))   else None,
        str(row["who"])        if pd.notna(row.get("who"))        else None,
        int(row["adult_male"]) if pd.notna(row.get("adult_male")) else None,
        int(row["family_size"]),
        int(row["is_alone"]),
    ))

cursor.executemany(
    "INSERT INTO passengers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
    passengers_rows
)
conn.commit()
print(f"  Database created: {DB_PATH}")
print(f"  Tables: passengers ({len(passengers_rows)} rows), embarkation_ports (3 rows), class_info (3 rows)")

def run_query(title, sql, conn):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"  SQL:\n{sql}")
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))
    return result

run_query(
    "Q1 - SELECT: First 10 passengers",
    """
SELECT passenger_id, sex, ROUND(age,1) AS age, pclass, survived
FROM   passengers
LIMIT  10;
    """,
    conn
)

run_query(
    "Q2 - WHERE: Female survivors in 1st class, fare > 50",
    """
SELECT passenger_id, sex, ROUND(age,1) AS age, pclass,
       ROUND(fare,2) AS fare, survived
FROM   passengers
WHERE  sex = 'female'
  AND  survived = 1
  AND  pclass   = 1
  AND  fare     > 50
ORDER BY fare DESC
LIMIT  15;
    """,
    conn
)

run_query(
    "Q3 - GROUP BY: Survival stats per passenger class",
    """
SELECT pclass,
       COUNT(*)                         AS total,
       SUM(survived)                    AS survivors,
       ROUND(AVG(survived) * 100, 1)   AS survival_pct,
       ROUND(AVG(age), 1)              AS avg_age,
       ROUND(AVG(fare), 2)             AS avg_fare
FROM   passengers
GROUP BY pclass
ORDER BY pclass;
    """,
    conn
)

run_query(
    "Q4 - GROUP BY (multi-key): Survival rate by sex x class",
    """
SELECT sex, pclass,
       COUNT(*)                        AS total,
       SUM(survived)                   AS survivors,
       ROUND(AVG(survived)*100, 1)    AS survival_pct
FROM   passengers
GROUP BY sex, pclass
ORDER BY sex, pclass;
    """,
    conn
)

run_query(
    "Q5 - JOIN: Survival rate by embarkation port (with port name & country)",
    """
SELECT ep.port_name,
       ep.country,
       COUNT(p.passenger_id)           AS total_passengers,
       SUM(p.survived)                 AS survivors,
       ROUND(AVG(p.survived)*100, 1)  AS survival_pct,
       ROUND(AVG(p.fare), 2)          AS avg_fare
FROM   passengers p
JOIN   embarkation_ports ep ON p.embarked = ep.port_code
GROUP BY ep.port_name
ORDER BY survival_pct DESC;
    """,
    conn
)

run_query(
    "Q6 - JOIN: Class info enriched with passenger stats",
    """
SELECT ci.class_name,
       ci.description,
       COUNT(p.passenger_id)           AS passengers,
       ROUND(AVG(p.age), 1)           AS avg_age,
       ROUND(AVG(p.fare), 2)          AS avg_fare,
       ROUND(AVG(p.survived)*100, 1)  AS survival_pct
FROM   passengers p
JOIN   class_info ci ON p.pclass = ci.pclass
GROUP BY ci.class_name
ORDER BY p.pclass;
    """,
    conn
)

run_query(
    "Q7 - GROUP BY: Survival rate by family size",
    """
SELECT family_size,
       COUNT(*)                        AS total,
       SUM(survived)                   AS survivors,
       ROUND(AVG(survived)*100, 1)    AS survival_pct
FROM   passengers
GROUP BY family_size
ORDER BY family_size;
    """,
    conn
)

run_query(
    "Q8 - Subquery: Passengers who paid above-average fare",
    """
SELECT passenger_id, sex, pclass,
       ROUND(age, 1)  AS age,
       ROUND(fare, 2) AS fare,
       survived
FROM   passengers
WHERE  fare > (SELECT AVG(fare) FROM passengers)
ORDER BY fare DESC
LIMIT 10;
    """,
    conn
)

conn.close()
print(f"\n  Database connection closed.")
print(f"  Database file saved: {DB_PATH}")
print("\n[DONE] Part D complete.\n")
