import pandas as pd
import glob
import os

folder = "/Users/mooncyli/Desktop/BU_RISE/BU-RISE/results/model"

# Folder containing the CSV files
csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

all_errors = []

for file in csv_files:
    df = pd.read_csv(file)

    # Get every value from the rightmost column
    errors = df.iloc[:, -1].tolist()

    print(f"{file}: {errors}")

    all_errors.extend(errors)

average_error = sum(all_errors) / len(all_errors)

print("\n-------------------------")
print(f"Number of values: {len(all_errors)}")
print(f"Average error: {average_error:.6f} m")