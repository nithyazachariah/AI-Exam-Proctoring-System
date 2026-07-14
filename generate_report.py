import pandas as pd
import os

# Create reports folder if missing
os.makedirs("reports", exist_ok=True)

# Read log file
df = pd.read_csv("logs/events.csv")

# Count events
counts = df["Event"].value_counts()

# Calculate risk score
risk_score = 0

for event in df["Event"]:

    if event == "Looking Away":
        risk_score += 20

    elif event == "Student Absent":
        risk_score += 30

    elif event == "Mobile Phone Detected":
        risk_score += 40

    elif event == "Multiple Faces Detected":
        risk_score += 50

# Risk Level
if risk_score < 100:
    level = "LOW"

elif risk_score < 300:
    level = "MEDIUM"

else:
    level = "HIGH"

# Report Content
report = f"""
AI EXAM PROCTORING REPORT

==========================

Total Violations: {len(df)}

Risk Score: {risk_score}

Risk Level: {level}

Violation Breakdown

{counts.to_string()}

==========================
"""

# Save Report
with open(
    "reports/exam_summary.txt",
    "w"
) as file:

    file.write(report)

print(report)
print("Report Generated!")