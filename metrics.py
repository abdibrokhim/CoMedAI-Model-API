import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()

def pretty_print_json(data):
    print(json.dumps(data, indent=4))


def get_metrics(fine_tuning_job_id):
    url = f"https://api.openai.com/v1/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints"
    
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    response = requests.get(url, headers=headers)
    
    return response.json()

# Usage
# fine_tuning_job_id = "ftjob-EPh6MssqzIvPP0P95VtPaBkJ"
# r = get_metrics(fine_tuning_job_id)
# pretty_print_json(r)