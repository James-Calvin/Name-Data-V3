import pandas as pd
import numpy as np
import os

# Load and transform first name race data
print("Loading first name race data...")
race_data = pd.read_csv("source_data/firstnames.csv")

# Convert observations and percentages to per-field counts
for col in race_data.columns[2:]:
  race_data[col] = round(race_data[col] * race_data["obs"] / 100).astype(int)
race_data = race_data.drop(columns=["obs", "pct2prace"])
race_data.columns = race_data.columns.str.replace("pct", "", regex=False)
print(race_data.head())

# Load gender counts
print()
print("Loading first name gender data...")
gender_data = pd.read_csv("source_data/gender_data_national(1984-2023).csv")
gender_data["name"] = gender_data["name"].str.upper()
print(gender_data.head())

# Merge
print()
print("Merging datasets...")
merged_data = pd.merge(
    gender_data, 
    race_data, 
    left_on='name', 
    right_on='firstname', 
    how='inner'
)
merged_data = merged_data.drop(columns=["firstname"])
def geometric_mean(a, b):
    return np.sqrt(a * b)
gender_cols = ['f', 'm']
race_cols = ['hispanic', 'white', 'black', 'api', 'aian']
results = []
for gender in gender_cols:
  for race in race_cols:
    col_name = f"{gender}{race}"
    merged_data[col_name] = merged_data.apply(
      lambda row: geometric_mean(row[gender], row[race]), axis=1
    )
    results.append(col_name)

# Filter only the names we want
final_columns = ['name'] + results
final_data = merged_data[final_columns]
final_data["name"] = final_data["name"].str.capitalize()

# Saving an approximate, integer count of the names
counts_data = final_data.copy()
for col in counts_data.columns:
  if col == "name": continue
  counts_data[col] = round(counts_data[col]).astype(int)

# Instantiate the intermediate dataset directory
intermediate_dir = "intermediate"
os.makedirs(intermediate_dir, exist_ok=True)

# Save the counts
counts_path = os.path.join(intermediate_dir,"first_name_counts.csv")
counts_data.to_csv(counts_path,index=False)

print(final_data.head())

print()
print("Converting to probabilities...")

# Column-wise normalization
names = final_data["name"]
convert_data = final_data.drop(columns=["name"])
convert_data = convert_data.div(convert_data.sum(axis=0), axis=1)
print(convert_data.head())

# If we filter results by top *n* values here, we're looking at the portion of
#   that name inside its category. This removes biases from larger categories.
#   For example, male & white may just have more data samples in our dataset so 
#   the top names would favor male & white names.
convert_data["score"] = convert_data.apply(
  lambda row: row[row > 0].mean(), 
  axis=1
)
count = 4000
convert_data = convert_data.sort_values(by="score", ascending=False).head(count)
convert_data = convert_data.drop(columns=["score"])
final_names = names.loc[convert_data.index].reset_index(drop=True)
probability_save = pd.concat(
  [final_names, convert_data.reset_index(drop=True)], 
  axis=1
)

# Re-normalize columns since we drop names since the last normalization
names = probability_save["name"]
probability_save = probability_save.drop(columns=["name"])
probability_save = probability_save.div(probability_save.sum(axis=0),axis=1)
probability_save = pd.concat([names,probability_save],axis=1)

print(probability_save.head())

# probability_save = pd.concat([names,convert_data],axis=1)
prob_path = os.path.join(intermediate_dir, "first_name_probabilities.csv")
probability_save.to_csv(prob_path, index=False)


print()
print("Converting to cumulative sum...")
names = probability_save["name"]
cumulative_data = probability_save.drop(columns=["name"])
cumulative_data = cumulative_data.cumsum(axis=0)
# Normalize each column to ensure the last value equals 1.0
for col in cumulative_data.columns:
    last_value = cumulative_data[col].iloc[-1]
    cumulative_data[col] = cumulative_data[col] / last_value
cumulative_data = pd.concat([names,cumulative_data],axis=1)

cum_path = os.path.join(intermediate_dir, "first_name_cumulative.csv")
cumulative_data.to_csv(cum_path, index=False)
print(cumulative_data.head())

# Define thresholds for reporting: 5%, 10%, ..., 95%
thresholds = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
current_threshold = thresholds.pop(0)
print("Finding names at cumulative thresholds...")
for idx, row in cumulative_data.iterrows():
    name = row["name"]
    avg_cumsum = row.iloc[1:].mean()  # Average cumulative sum, excluding 'name' column
    
    # Check if we crossed the current threshold
    if avg_cumsum >= current_threshold:
        print(f"Threshold {current_threshold:.3f} crossed at index {idx} with name '{name}' (avg_cumsum={avg_cumsum:.5f})")
        
        # Move to the next threshold
        if thresholds:
            current_threshold = thresholds.pop(0)
        else:
            break  # Stop when all thresholds are crossed

# Save the names as a text file (lookup table)
out_dir = os.path.join("output", "first_names")
os.makedirs(out_dir,exist_ok=True)

name_file = os.path.join(out_dir,"names.txt")
with open(name_file, "w") as f:
    for name in cumulative_data["name"]:
        f.write(name + "\n")
print(f"Saved names to {name_file}")

# Save each column as a binary file
for col in cumulative_data.columns[1:]:  # Exclude 'name' column
    binary_file = os.path.join(out_dir,f"{col}.bin")
    
    # Convert column to float32 and save as raw binary
    data = cumulative_data[col].astype(np.float32)
    data.to_numpy().tofile(binary_file)
    
    print(f"Saved {col} data to {binary_file}")