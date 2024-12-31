import unicodedata
import csv
from collections import defaultdict

input_file = "source_data/Surnames.txt" 
output_file = "intermediate/name_format_map.csv"

def standardize_name(name):
    name = unicodedata.normalize("NFKD", name)
    # Keep only alphabetic characters
    name = ''.join(char for char in name if char.isalpha())  
    return name.upper()

# Collect names by standard form
name_groups = defaultdict(list)
with open(input_file, "r", encoding="utf-8") as file:
  for line in file:
    original_name = line.strip()
    simplified_name = ''.join(char for char in original_name if char.isalpha())
    simplified_name = simplified_name.capitalize()
    if simplified_name == original_name:
      continue
    standardized_name = standardize_name(original_name)
    name_groups[standardized_name].append(original_name)

resolved_names = []
for standard, originals in name_groups.items():
  if len(originals) > 1:
    print(f"Standard form '{standard}' has multiple options:")
    for idx, name in enumerate(originals, 1):
      name = str(name)
      print(f"{idx}. '{name}'")
    while True:
      try:
        selection = int(input(f"Select the number to keep for '{standard}': "))
        if 1 <= selection <= len(originals):
          resolved_names.append((originals[selection - 1], standard))
          break
        else:
          print("Invalid choice. Please select a valid number.")
      except ValueError:
        print("Invalid input. Please enter a number.")
  else:
    resolved_names.append((originals[0], standard))

# Write the resolved results to a CSV file
with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["original", "standard"])
    writer.writerows(resolved_names)

print(f"Filtered and resolved names saved to {output_file}")
