import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

folder = "/Users/mooncyli/Desktop/BU_RISE/BU-RISE/results/model"

# Folder containing the CSV files
csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

all_errors = []

for file in csv_files:
    df = pd.read_csv(file)

    # Get every value from the rightmost column
    all_errors.extend(df.iloc[:, -1].tolist())

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(all_errors, bins=8)   # Adjust bins as desired
plt.xlabel("Final Position Error (m)")
plt.ylabel("Frequency")
plt.title("Distribution of Final Position Errors")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig("position_error_histogram.png", dpi=300)
plt.show()

print(f"Number of samples: {len(all_errors)}")
print(f"Mean error: {sum(all_errors)/len(all_errors):.4f} m")