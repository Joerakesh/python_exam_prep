# Python Data Science Practical Exercises

This repository contains runnable examples for two practical data-analysis questions.

## Files

- `question_10_student_analysis.py` — student academic performance analysis: dataset exploration, missing-value handling, duplicate removal, filtering, department aggregation, correlation and visualisations.
- `question_11_employee_analysis.py` — employee dataset exploration, variable identification, data cleaning, descriptive statistics, department-wise aggregation and visualisations.
- `requirements.txt` — Python package dependencies.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows Command Prompt**

```bat
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run either script:

```bash
python question_10_student_analysis.py
python question_11_employee_analysis.py
```

Both scripts use small sample datasets so they can run immediately. To analyse a real CSV, replace the sample-data section with `pd.read_csv("students.csv")` or `pd.read_csv("employees.csv")` and ensure the column names match the code.

## Libraries

- **Pandas:** tabular data loading, cleaning, filtering, grouping and statistics.
- **Matplotlib:** chart creation and formatting.
- **Seaborn:** statistical visualisations such as heatmaps and histograms.
- **NumPy:** numerical computing dependency commonly used with Pandas and the scientific Python ecosystem.

Correlation measures association and does not establish causation. Missing-value strategies should be selected based on the real dataset and why values are missing.
