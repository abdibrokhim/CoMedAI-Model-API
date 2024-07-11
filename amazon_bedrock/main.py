import json

def convert_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            original = json.loads(line)
            system_message = original['messages'][0]['content']
            user_message = original['messages'][1]
            assistant_message = original['messages'][2]
            
            new_format = {
                "system": system_message,
                "messages": [
                    {
                        "role": user_message['role'],
                        "content": user_message['content']
                    },
                    {
                        "role": assistant_message['role'],
                        "content": assistant_message['content']
                    }
                ]
            }
            outfile.write(json.dumps(new_format) + '\n')

# Specify the input and output file paths
input_file = 'validation_200.jsonl'
output_file = 'new_validation_200.jsonl'

# Convert the JSONL file
convert_jsonl(input_file, output_file)
