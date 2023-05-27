import argparse
import os
import json
from query import generate_messages, query, handle_query
import csv
from utils import join_lists, parse_sox_instruction


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_csv", type=str, default="14effects_1instruction.csv")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--result_root", type=str, default="./exps/v1/", help="folder to store the experiment")
    
    args = parser.parse_args()
    return args


def get_instructions_from_response(response):
    choices = response["choices"]
    return join_lists([parse_sox_instruction(choice["message"]["content"]) for choice in choices])


def main(args):
    prompt_csv = args.prompt_csv
    model = args.model
    n = args.n
    temp = args.temp
    result_json_path = os.path.join(args.result_root, 'result.json')
    response_root = os.path.join(args.result_root, 'responses')
    os.makedirs(response_root, exist_ok=True)
    
    if os.path.isfile(result_json_path):
        with open(result_json_path, 'r') as f:
            result_json = json.load(f)
    else:
        result_json = {}        
    
    
    template = "Give me 10 diverse sentences with the same meaning as <instruction>."
    system_prompt = "You are a helpful assistant for generating instructions."
    
    with open(prompt_csv, 'r') as f:
        reader = csv.DictReader(f)
        try:
            for row in reader:
                print(row)
                name = row['name'].strip()
                if name in result_json:
                    continue
                instruction = row['Instruction'].strip()
                instruction = template.replace('<instruction>', f'"{instruction}"')
                
                messages = generate_messages(system_prompt, instruction)
                response_path = os.path.join(response_root, f'{name}.json')
                response = handle_query(messages, model, n, temp, response_path)
                result_json[name] = get_instructions_from_response(response)
        except:
            with open(result_json_path, 'w+') as f:
                json.dump(result_json, f, indent=4)
            print(f"Save to {result_json_path}")
            exit(0)
    
    with open(result_json_path, 'w+') as f:
        json.dump(result_json, f, indent=4)
    

if __name__ == '__main__':
    args = get_args()
    main(args)