import pandas as pd
import numpy as np
import os
# from nameparser import HumanName

print("Loading last name race data...")
race_data = pd.read_csv("source_data/Names_2010Census.csv", dtype={"name": str})  # Force 'name' column to string
race_data["name"] = race_data["name"].fillna("").str.strip()  # Remove leading/trailing whitespace and fill NaNs with empty strings

# Check for and drop rows with invalid names (e.g., empty after processing)
# Note: This removed the name "NULL". Given the word's ubiquity among languages
#   as a reserved word and the issues it could have, am decided to agree with
#   the result and maintain the removal of the name "NULL"
invalid_names = race_data[race_data["name"] == ""]
if not invalid_names.empty:
    print(f"Found invalid names. Removing {len(invalid_names)} rows:")
    print(invalid_names)
    race_data = race_data[race_data["name"] != ""]

# Census data as some suppressed fields, we are filtering to only use the names
#   which have all race data. Suppressed fields are marked with "(S)"
print()
print("Removing incomplete data...")
race_data = race_data[~race_data.apply(lambda row: row.astype(str).str.contains(r"\(S\)").any(), axis=1)]
print(race_data.head())

# Find the counts of each race for each name
for col in race_data.columns[5:]:
  race_data[col] = round(race_data[col].astype(float) * race_data["count"] / 100).astype(int)

# remove unneeded fields
race_data = race_data.drop(columns=[
  "rank", "count", "prop100k", "cum_prop100k", "pct2prace"
  ])

# rename fields
race_data.columns = race_data.columns.str.replace("pct", "", regex=False)
print()
print(race_data.head())

# Instantiate the intermediate dataset directory
intermediate_dir = "intermediate"
os.makedirs(intermediate_dir, exist_ok=True)

# Save the counts
counts_path = os.path.join(intermediate_dir,"last_name_counts.csv")
race_data.to_csv(counts_path,index=False)

# Column-wise normalization
print()
print("Converting to probabilities...")
names = race_data["name"]
convert_data = race_data.drop(columns=["name"])
convert_data = convert_data.div(convert_data.sum(axis=0), axis=1)
print(convert_data.head())

# Calculate top names
convert_data["score"] = convert_data.apply(
  lambda row: row[row > 0].mean(), 
  axis=1
)
count = 10000
convert_data = convert_data.sort_values(by="score", ascending=False).head(count)
convert_data = convert_data.drop(columns=["score"])
final_names = names.loc[convert_data.index].reset_index(drop=True)
probability_save = pd.concat(
  [final_names, convert_data.reset_index(drop=True)], 
  axis=1
)

formatted_names = pd.read_csv("intermediate/filtered_names_map.csv")
name_map = dict(zip(formatted_names["standard"], formatted_names["original"]))

used_dict_names = []

def format_name(name: str) -> str:
  if name in name_map:
    used_dict_names.append(name_map[name])
    return name_map[name]
  return name.capitalize()

names = probability_save["name"].apply(format_name)

pd.DataFrame(used_dict_names).to_csv(
  "special_names.csv", 
  index=False, 
  header=False
)

print()
print(f"Names that use name dictionary: {len(used_dict_names)}")
print()

# Re-normalize the data
probability_save = probability_save.drop(columns=["name"])
probability_save = probability_save.div(probability_save.sum(axis=0),axis=1)
probability_save = pd.concat([names,probability_save],axis=1)

print(probability_save.head())
# probability_save = pd.concat([names,convert_data],axis=1)
prob_path = os.path.join(intermediate_dir, "last_name_probabilities.csv")
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

cum_path = os.path.join(intermediate_dir, "last_name_cumulative.csv")
cumulative_data.to_csv(cum_path, index=False)
print(cumulative_data.head())

# Define thresholds for reporting: 5%, 10%, ..., 95%
thresholds = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
current_threshold = thresholds.pop(0)
print("Finding names at cumulative thresholds...")
for idx, row in cumulative_data.iterrows():
  name = row["name"]
  avg_cumsum = row.iloc[1:].mean()  # Average cumulative sum
  
  # Check if we crossed the current threshold
  if avg_cumsum >= current_threshold:
    print(f"Threshold {current_threshold:.3f} crossed at index {idx} with "
          +f"name '{name}' (avg_cumsum={avg_cumsum:.5f})")
    
    # Move to the next threshold
    if thresholds:
      current_threshold = thresholds.pop(0)
    else:
      break  # Stop when all thresholds are crossed

# Save the names as a text file (lookup table)
out_dir = os.path.join("output", "last_names")
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