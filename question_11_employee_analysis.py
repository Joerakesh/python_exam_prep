"""Employee dataset exploration and department-wise analysis.

Run: python question_11_employee_analysis.py
For a real dataset, replace the sample-data section with pd.read_csv('employees.csv').
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 1. Create sample data. Replace this with pd.read_csv("employees.csv")
    data = {
        "Employee_ID": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
        "Department": ["IT", "HR", "IT", "Sales", "HR", "Sales", "IT", "HR", "Sales", "IT"],
        "Age": [25, 30, 28, 35, 32, 27, 40, 29, 31, 26],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
        "Experience": [2, 5, 4, 10, 7, 3, 15, 4, 6, 2],
        "Salary": [30000, 45000, 40000, 70000, 55000, 35000, 90000, 42000, 60000, 32000],
        "Performance_Score": [75, 85, 80, 90, 82, 78, 95, 88, 84, 76],
        "Working_Hours": [40, 38, 42, 45, 40, 39, 44, 41, 42, 40],
        "Attrition": ["No", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No"],
    }
    df = pd.DataFrame(data)

    # Demonstrate missing-value handling.
    df.loc[2, "Salary"] = None
    df.loc[5, "Experience"] = None
    df.loc[7, "Department"] = None

    # 2. Explore the dataset.
    print("\n--- First five rows ---\n", df.head())
    print("\nShape (rows, columns):", df.shape)
    print("\n--- Dataset information ---")
    df.info()
    print("\n--- Descriptive statistics ---\n", df.describe())
    print("\nColumns:\n", df.columns.tolist())

    # 3. Identify numerical and categorical columns.
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    print("\nNumerical variables:\n", numerical_cols)
    print("\nCategorical variables:\n", categorical_cols)

    # 4. Detect and handle missing values.
    print("\nMissing values before cleaning:\n", df.isna().sum())
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    print("\nMissing values after cleaning:\n", df.isna().sum())

    # 5. Detect and remove duplicate rows and duplicate employee IDs.
    print("\nDuplicate rows before removal:", df.duplicated().sum())
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="Employee_ID", keep="first")
    print("Duplicate rows after removal:", df.duplicated().sum())

    # 6. Basic preprocessing and type checks.
    df["Employee_ID"] = df["Employee_ID"].astype(int)
    df["Age"] = df["Age"].astype(int)
    df["Experience"] = df["Experience"].astype(int)
    df["Salary"] = df["Salary"].astype(int)
    print("\n--- Cleaned dataset ---\n", df.head())
    print("\n--- Data types ---\n", df.dtypes)

    # 7. Descriptive statistics for numerical variables.
    print("\n--- Numerical descriptive statistics ---\n", df[numerical_cols].describe())
    print("\nMedian salary:", df["Salary"].median())
    print("Average experience:", round(df["Experience"].mean(), 2))
    print("Average performance score:", round(df["Performance_Score"].mean(), 2))

    # 8. Department-wise salary and performance aggregation.
    department_summary = df.groupby("Department").agg(
        Employee_Count=("Employee_ID", "count"),
        Average_Salary=("Salary", "mean"),
        Maximum_Salary=("Salary", "max"),
        Average_Performance=("Performance_Score", "mean"),
        Average_Experience=("Experience", "mean"),
    )
    print("\n--- Department-wise aggregation ---\n", department_summary.round(2))

    # 9. Visualisation: average salary by department.
    department_summary["Average_Salary"].plot(kind="bar", title="Average Salary by Department", xlabel="Department", ylabel="Average Salary", rot=0)
    plt.tight_layout()
    plt.show()

    # 10. Visualisation: performance score distribution.
    sns.histplot(data=df, x="Performance_Score", bins=5, kde=True)
    plt.title("Employee Performance Score Distribution")
    plt.xlabel("Performance Score")
    plt.ylabel("Number of Employees")
    plt.tight_layout()
    plt.show()

    # 11. Optional: compare attrition counts by department.
    print("\n--- Attrition by department ---\n", pd.crosstab(df["Department"], df["Attrition"]))


if __name__ == "__main__":
    main()
