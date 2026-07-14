import pandas as pd
import matplotlib.pyplot as plt
import os

# Create reports folder if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Read event logs
df = pd.read_csv("logs/events.csv")

# Count violations
counts = df["Event"].value_counts()

# Display summary in terminal
print("\nViolation Summary:\n")
print(counts)

# Save summary report CSV
counts.to_csv(
    "reports/summary_report.csv",
    header=["Count"]
)

# Create bar chart
plt.figure(figsize=(10, 6))

counts.plot(
    kind="bar"
)

plt.title("AI Exam Proctoring Violations")
plt.xlabel("Violation Type")
plt.ylabel("Count")

plt.xticks(rotation=20)

plt.tight_layout()

# Save graph
plt.savefig(
    "reports/violation_graph.png"
)

# Show graph
plt.show()

print("\nDashboard Generated Successfully!")
print("Files Created:")
print("1. reports/summary_report.csv")
print("2. reports/violation_graph.png")