
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load CSV file
df = pd.read_csv("students.csv")

# 2. Display the data
print("----- STUDENT DATA -----")
print(df)

# 3. Basic analysis
print("\n----- DATASET INFORMATION -----")
print("Total Students:", len(df))
print("Average Study Hours:", df["Study_Hours"].mean())
print("Average Attendance:", df["Attendance"].mean())
print("Average Marks:", df["Marks"].mean())

print("\n----- STATISTICAL SUMMARY -----")
print(df.describe())

# 4. Bar chart: Student marks
plt.figure(figsize=(9, 5))
plt.bar(df["Student"], df["Marks"])
plt.title("Student Marks Comparison")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()

# 5. Scatter plot: Study hours vs marks
plt.figure(figsize=(8, 5))
plt.scatter(df["Study_Hours"], df["Marks"])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# 6. Heatmap: Correlation analysis
correlation = df[
    ["Study_Hours", "Attendance", "Marks"]
].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()

print("\nAnalysis completed successfully!")
