import pandas as pd
import json
import sys

# Function to read the CSV and convert it to a list of dictionaries
def csv_to_self_instruct_json(csv_file_path):
    df = pd.read_csv(csv_file_path)
    data = []
    for _, row in df.iterrows():
        entry = {
            'prompt': row['input'],
            'completion': row['output']
        }
        data.append(entry)
    return data

# Path to your CSV file
csv_file_path = sys.argv[1]
data = csv_to_self_instruct_json(csv_file_path)

# Function to save the data to a JSON file
def save_as_json(data, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Path where you want to save the JSON file
json_file_path = sys.argv[2]
save_as_json(data, json_file_path)
