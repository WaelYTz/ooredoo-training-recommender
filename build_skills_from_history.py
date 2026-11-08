"""
Fill employees_final.csv's Skills column from real training history.

An employee's skills = every skill tagged (Skills_EN) on the LinkedIn
Learning courses they completed (Status == "completed", i.e. >=75%).
Employees with no completed course get blank Skills and
Skills_Source = "none" -- no skills are generated or guessed.

Run from anywhere, after the real files are in data/processed/:
    python build_skills_from_history.py
Overwrites the Skills and Skills_Source columns of employees_final.csv.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

emp = pd.read_csv(PROCESSED_DATA / "employees_final.csv").fillna("")
hist = pd.read_csv(PROCESSED_DATA / "history_cleaned.csv").fillna("")
cat = pd.read_csv(PROCESSED_DATA / "catalog_full_en_fr_translated.csv", low_memory=False).fillna("")

skills_by_course = dict(zip(cat["Course ID"].astype(str), cat["Skills_EN"]))
completed = hist[hist["Status"].str.strip().str.lower() == "completed"]

real_skills = {}
for email, group in completed.groupby("email_n"):
    skills = set()
    for course_id in group["ID de contenu"].astype(str):
        skills.update(s.strip() for s in str(skills_by_course.get(course_id, "")).split(",") if s.strip())
    if skills:
        real_skills[email] = ", ".join(sorted(skills))

emp["Skills"] = emp["Email"].map(real_skills).fillna("")
emp["Skills_Source"] = emp["Skills"].map(lambda s: "real" if s else "none")
emp.to_csv(PROCESSED_DATA / "employees_final.csv", index=False)

print(f"Employees with skills from history: {(emp['Skills_Source'] == 'real').sum()} / {len(emp)}")
