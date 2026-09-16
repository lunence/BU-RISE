import glob
import pandas as pd
import numpy as np
import os


folder = "/Users/mooncyli/Desktop/BU_RISE/BU-RISE/results/model"

# Folder containing the CSV files
csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

errors = []

for file in csv_files:
    df = pd.read_csv(file)

    file_errors = df.iloc[:, -1].tolist()

    print(f"{file}: {file_errors}")

    errors.extend(file_errors)

errors = np.array(errors)

stats = {
    "Count": len(errors),
    "Mean": np.mean(errors),
    "Median": np.median(errors),
    "Standard deviation": np.std(errors, ddof=1),
    "Variance": np.var(errors, ddof=1),
    "Minimum": np.min(errors),
    "Maximum": np.max(errors),
    "Range": np.max(errors) - np.min(errors),
    "25th percentile (Q1)": np.percentile(errors, 25),
    "75th percentile (Q3)": np.percentile(errors, 75),
    "Interquartile range (IQR)": np.percentile(errors, 75)
                                - np.percentile(errors, 25),
}

print("\nNavigation Error Statistics")
print("-" * 35)
for key, value in stats.items():
    if key == "Count":
        print(f"{key:25s}: {value}")
    else:
        print(f"{key:25s}: {value:.4f}")

