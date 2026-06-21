import pandas as pd

output_path = r"D:\Internship Projects Thirex\Task4_Data_Cleaning"

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

print("Original Dataset Shape:")
print(df.shape)

duplicate_rows = df.duplicated().sum()

print("\nDuplicate Rows Found:")
print(duplicate_rows)

df_cleaned = df.drop_duplicates()

missing_values = df_cleaned.isnull().sum()

print("\nMissing Values:")
print(missing_values)

cleaned_file = f"{output_path}\\cleaned_superstore.csv"
df_cleaned.to_csv(cleaned_file, index=False)

summary_report = df_cleaned.describe(include="all")
summary_report_file = f"{output_path}\\data_summary_report.csv"
summary_report.to_csv(summary_report_file)

report = pd.DataFrame({
    "Column Name": df_cleaned.columns,
    "Missing Values": df_cleaned.isnull().sum().values
})

report_file = f"{output_path}\\data_cleaning_report.csv"
report.to_csv(report_file, index=False)

print("\nFiles Generated Successfully:")
print("1. cleaned_superstore.csv")
print("2. data_cleaning_report.csv")
print("3. data_summary_report.csv")