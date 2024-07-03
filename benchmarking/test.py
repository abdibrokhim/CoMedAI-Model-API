import os
from dotenv import load_dotenv
import re
import json
from openai import OpenAI
import time
import gpt4, gpt4o, fine_tune

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_model(content):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You will be given two Medical Conclusions. Your task is to evaluate the quality of the generated conclusion versus the real conclusion from 0 to 100. You MUST return the evaluation in the following list format: [int, int, int, ...]. NEVER return explanation or comments. Return ONLY list of integers. You will be given real conclusion and generated conclusion in the following pattern: \n[Test <number>]\n[REAL] <Real conclusion goes here>. \n\n[GENERATED] <Generated conclusion goes here>."
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=1,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response_text = response.choices[0].message.content
    print('\n=====================\n')
    print("OpenAI response_text (grade_list): ", response_text)
    print('\n=====================\n')

    if not response_text:
        raise ValueError("Received empty response from the API.")
    try:
        grade_list = json.loads(response_text)
    except json.JSONDecodeError as e:
        print("JSONDecodeError: ", e)
        raise ValueError("Failed to decode JSON from the response text.")
    return grade_list

def benchmark_json(json_file_path, observation_list, real_result_list, generated_conclusion_list, grade_list):
    data = []
    for i in range(len(observation_list)):
        data.append({
            i+1: {
                "observation": observation_list[i],
                "real_conclusion": real_result_list[i],
                "generated_conclusion": generated_conclusion_list[i],
                "evaluation": [{"grade": grade_list[i], "from": 100}]
            }
        })
    average_grade = sum(grade_list) / len(grade_list)
    data.append({"average_grade": average_grade})
    
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_observation_conclusion(file):
    observation_list = []
    conclusion_list = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            for message in data["messages"]:
                if message["role"] == "user":
                    observation_list.append(message["content"])
                elif message["role"] == "assistant":
                    conclusion_list.append(message["content"])
    return observation_list, conclusion_list

def backup_it(content_str, file_name):
    file_path = f"{file_name}.txt"
    with open(f"{file_name}.txt", "w", encoding="utf-8") as f:
        f.write(content_str)
    return file_path

def processing(model_name, jsonl_file_path):
    json_file_path = model_name + '.json'
    content_str = ""
    observation_list, real_result_list = get_observation_conclusion(jsonl_file_path)
    generated_conclusion_list = []

    total = len(observation_list)
    
    for i in range(total):
        print(f"\nProcessing test {i+1}...")
        if model_name == "gpt4-turbo":
          gen_con = gpt4.conclude(observation_list[i])
          generated_conclusion_list.append(gen_con)
        if model_name == "gpt4o":
          gen_con = gpt4o.conclude(observation_list[i])
          generated_conclusion_list.append(gen_con)
        if model_name == "gpt3.5-fine-tuned":
          gen_con = fine_tune.conclude(observation_list[i])
          generated_conclusion_list.append(gen_con)
               
    for i in range(total):
        content_str += f"""
[Test {i+1}]
[REAL] 
{real_result_list[i]}.
[GENERATED]
{generated_conclusion_list[i]}
\n"""
    
    print("\nTesting model...")
    print("Content: ", content_str)
    backup_content_str = backup_it(content_str, model_name)
    print("Backup content: ", backup_content_str)
    
    success = False
    while not success:
        try:
            grade_list = test_model(content_str)
            print("Testing model completed!")
            success = True
        except Exception as e:
            print("An error occurred on Testing model: ", e)
            print("Retrying with backup content...")
            content_str = open(backup_content_str, "r", encoding="utf-8").read()

    print("Grade list: ", grade_list)
    print("Benchmarking...")
    try:
      benchmark_json(json_file_path, observation_list, real_result_list, generated_conclusion_list, grade_list)
      print("Benchmarking completed!")
    except Exception as e:
      print("An error occurred on Benchmarking: ", e)

def best_model(model_list):
    best_model_name = ""
    highest_average = 0
    for model in model_list:
        json_file_path = model + '.json'
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            average_grade = data[-1]["average_grade"]
            if average_grade > highest_average:
                highest_average = average_grade
                best_model_name = model
    return best_model_name

if __name__ == "__main__":
    jsonl_file_path = "validation_200.jsonl"
    delay_for = 60  # seconds
    model_list = ["gpt4-turbo", "gpt4o", "gpt3.5-fine-tuned"]
    try:
        for m in model_list:
            print(f"\nProcessing {m} model...")
            processing(m, jsonl_file_path)
            print(f"{m} model completed!")
            print(f"Waiting for {delay_for} seconds before processing the next model...")
            time.sleep(delay_for)
        print("\nAll tests completed!")
        print("The best model is: ", best_model(model_list))
    except Exception as e:
        print("An error occurred on Processing: ", e)
