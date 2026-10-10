"""Student academic performance analysis using Pandas, Matplotlib and Seaborn.

Run: python question_10_student_analysis.py
For a real dataset, replace the sample-data section with pd.read_csv('students.csv').
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 1. Create a sample dataset. Replace this with pd.read_csv("students.csv")
    data = {
        "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
        "Department": ["IT", "CSE", "IT", "CSE", "DS", "DS", "IT", "CSE", "DS", "IT"],
        "Attendance": [90, 85, 70, 95, 80, 88, 60, 92, 75, 85],
        "Internal_Marks": [85, 90, 65, 92, 75, 88, 55, 95, 70, 80],
        "Assignment_Marks": [90, 95, 70, 90, 80, 85, 60, 98, 75, 82],
        "Study_Hours": [5, 6, 3, 7, 4, 6, 2, 8, 4, 5],
        "Previous_GPA": [8.5, 9.0, 7.0, 9.2, 7.5, 8.8, 6.0, 9.5, 7.2, 8.0],
        "Final_Marks": [88, 94, 68, 96, 78, 89, 58, 98, 73, 83],
    }
    df = pd.DataFrame(data)

    # Demonstrate missing-value handling and duplicate removal.
    df.loc[2, "Study_Hours"] = None
    df.loc[6, "Attendance"] = None
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # deliberate duplicate

    # 2. Explore the dataset.
    print("\n--- First five rows ---\n", df.head())
    print("\nShape (rows, columns):", df.shape)
    print("\n--- Dataset information ---")
    df.info()
    print("\n--- Descriptive statistics ---\n", df.describe())
    print("\n--- Missing values by column ---\n", df.isna().sum())

    # 3. Fill missing numeric values with each column's median.
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill missing categorical values with the most frequent value, if any.
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # 4. Remove exact duplicate rows and repeated student IDs.
    print("\nDuplicate rows before removal:", df.duplicated().sum())
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="Student_ID", keep="first")
    print("Duplicate rows after removal:", df.duplicated().sum())
    print("\nMissing values after cleaning:\n", df.isna().sum())

    # 5. Filter students with final marks >= 85 and attendance >= 80%.
    high_performers = df.loc[(df["Final_Marks"] >= 85) & (df["Attendance"] >= 80)]
    print("\n--- High-performing students ---\n", high_performers)

    # 6. Department-wise averages.
    department_avg = df.groupby("Department")[["Internal_Marks", "Assignment_Marks", "Final_Marks"]].mean()
    print("\n--- Department-wise average marks ---\n", department_avg.round(2))

    # 7. Correlation between academic factors and final marks.
    features = ["Study_Hours", "Attendance", "Internal_Marks", "Assignment_Marks", "Previous_GPA", "Final_Marks"]
    correlation = df[features].corr(numeric_only=True)
    print("\n--- Correlation matrix ---\n", correlation.round(2))
    final_corr = correlation["Final_Marks"].drop("Final_Marks")
    print("\n--- Factors ranked by absolute correlation with final marks ---\n", final_corr.reindex(final_corr.abs().sort_values(ascending=False).index).round(2))

    # 8. Visualisation: department-wise average final marks.
    department_final = df.groupby("Department")["Final_Marks"].mean()
    department_final.plot(kind="bar", title="Average Final Marks by Department", xlabel="Department", ylabel="Average Final Marks", rot=0)
    plt.tight_layout()
    plt.show()

    # 9. Visualisation: correlation heatmap.
    plt.figure(figsize=(9, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Between Academic Factors")
    plt.tight_layout()
    plt.show()

    print("\nInterpretation: correlation shows association, not proof of causation.")


if __name__ == "__main__":
    main()
