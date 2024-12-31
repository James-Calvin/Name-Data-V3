import pandas as pd
import numpy as np
import os

race_data = pd.read_csv("source_data/middle_nameRaceProbs.csv")
print(race_data.head())

# Note: middle_nameRaceProb.csv only contains white, black, hispanic, asian,
#   and other categories. Although in accurate, we will be mapping the "other"
#   category to aian (American Indian and Alaska Native). This will allow the
#   constructed dataset to interface with the same code used to work with the
#   first name data. Ideally, we will find a dataset with all the same 
#   categories as the first name data and with raw counts.

gender_data = pd.read_csv("source_data/gender_data_national(1984-2023).csv")
gender_data["name"] = gender_data["name"].str.upper()
print(gender_data.head())

print()
merged_data = pd.merge(
  gender_data,
  race_data,
  left_on="name",
  right_on="name",
  how="inner"
)
merged_data = merged_data.rename(columns={
  "whi": "white",
  "bla": "black",
  "his": "hispanic",
  "asi": "api",
  "oth": "aian"
  }
)
print(merged_data.head())
gender_cols = ['f', 'm']
race_cols = ['hispanic', 'white', 'black', 'api', 'aian']
results = []

# middle_nameRaceProbs.csv is normalized, so the scale of samples is exclusively
#   obtained from the scale in the gender data
for gender in gender_cols:
  for race in race_cols:
    col_name = f"{gender}{race}"
    merged_data[col_name] = merged_data.apply(
      lambda row: round(row[gender] * row[race]), axis=1
    )
    results.append(col_name)

# Filter only the names we want
final_columns = ['name'] + results
final_data = merged_data[final_columns]
final_data["name"] = final_data["name"].str.capitalize()

# Instantiate the intermediate dataset directory
intermediate_dir = "intermediate"
os.makedirs(intermediate_dir, exist_ok=True)

counts_path = os.path.join(intermediate_dir,"middle_name_counts.csv")
final_data.to_csv(counts_path,index=False)
print(final_data.head())

# Column-wise normalization
print()
print("Converting to probabilities...")
names = final_data["name"]
convert_data = final_data.drop(columns=["name"])
convert_data = convert_data.div(convert_data.sum(axis=0), axis=1)
print(convert_data.head())

# Calculate top names
convert_data["score"] = convert_data.apply(
  lambda row: row[row > 0].mean(), 
  axis=1
)
count = 2520
convert_data = convert_data.sort_values(by="score", ascending=False).head(count)
convert_data = convert_data.drop(columns=["score"])
final_names = names.loc[convert_data.index].reset_index(drop=True)
probability_save = pd.concat(
  [final_names, convert_data.reset_index(drop=True)], 
  axis=1
)

# Re-normalize the data
names = probability_save["name"]
probability_save = probability_save.drop(columns=["name"])
probability_save = probability_save.div(probability_save.sum(axis=0),axis=1)
probability_save = pd.concat([names,probability_save],axis=1)

print(probability_save.head())
# probability_save = pd.concat([names,convert_data],axis=1)
prob_path = os.path.join(intermediate_dir, "middle_name_probabilities.csv")
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

cum_path = os.path.join(intermediate_dir, "middle_name_cumulative.csv")
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
out_dir = os.path.join("output", "middle_names")
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