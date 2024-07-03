import csv
import json


def csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        with open(jsonl_file, 'w') as jsonl:
            for row in reader:
                template = {
                    "prompt": row['Observation'],
                    "completion": row['Conclusion']
                }
                jsonl.write(json.dumps(template) + '\n')


def csv_to_jsonl_2(csv_file, jsonl_file):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        with open(jsonl_file, 'w') as jsonl:
            for row in reader:
                template = {
                    "text": f"###prompt: {row['Observation']}, ###completion: {row['Conclusion']}"
                }
                jsonl.write(json.dumps(template) + '\n')



def make_csv(file_path, new_file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        with open(new_file_path, 'w', newline='') as new_file:
            writer = csv.DictWriter(new_file, fieldnames=['text'])
            writer.writeheader()
            for row in reader:
                text = f"###prompt: {row['Observation']}, ###completion: {row['Conclusion']}"
                writer.writerow({'text': text})


# Define the file paths
old_file_path = 'training.jsonl'
new_file_path = 'training_220.jsonl'

def convert_jsonl(old_file_path, new_file_path):
    instructions = "Write a conclusion based on the provided MRI scan observations. The conclusion should summarize the key findings, interpret their significance, provide recommendations for further evaluation, and give an overall impression of the patient's condition."

    with open(old_file_path, 'r') as old_file, open(new_file_path, 'w') as new_file:
        for line in old_file:
            # Parse the JSON line
            old_data = json.loads(line)

            # Create the new data structure
            new_data = {
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": old_data["prompt"]},
                    {"role": "assistant", "content": old_data["completion"]}
                ]
            }

            # Write the new data to the new file
            new_file.write(json.dumps(new_data) + '\n')

    print(f'Converted data written to {new_file_path}')



#==== Usage ====#


# convert_jsonl(old_file_path, new_file_path)


# path_to_csv = 'BrAIn MRI Data 2023 Full.csv'
# path_to_jsonl = 'training_v1.jsonl'

# csv_to_jsonl(path_to_csv, path_to_jsonl)
# csv_to_jsonl_2(path_to_csv, path_to_jsonl)


# make_csv(path_to_csv, 'new.csv')



def csv_to_json(file_path):
    result = {}
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            text = row["Conclusion"]
            num_words = len(text.split())
            result[idx] = {"text": text, "number of words": num_words}
    
    return json.dumps(result, indent=4)

# get highest number of words from json
def get_highest_number_of_words(json_data):
    data = json.loads(json_data)
    max_words = 0
    for key in data:
        num_words = data[key]["number of words"]
        if num_words > max_words:
            max_words = num_words
    return max_words

# Example usage:
file_path = 'BrAIn MRI Data 2023 Full.csv'
json_result = csv_to_json(file_path)
print(json_result)
max_words = get_highest_number_of_words(json_result)
print(max_words)
