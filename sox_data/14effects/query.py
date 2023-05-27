import openai
import json
import argparse
from tqdm import tqdm
import os


def generate_messages(system_message, single_prompt):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": single_prompt}
    ]


def get_all_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        return [
            line.strip() for line in f.readlines()
        ]


def get_all_messages(prompt_file):
    system_message = "You are a helpful assistant for generating instructions."
    prompts = get_all_prompts(prompt_file)
    return [
        generate_messages(system_message, prompt)
        for prompt in prompts
    ]


def query(messages, model="gpt-3.5-turbo-0301", n=5, temp=1):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temp,
    )
    return completion


def handle_query(messages, model, n, temp, result_path):
    response = query(messages, model, n, temp)
    parameters = {"model": model, "n": n, "temp": temp}
    result = {
        "request": messages,
        "response": response,
        "params": parameters,
    }
    with open(result_path, 'w+') as f:
        json.dump(result, f, indent=4)
    return response


def main(args):
    prompt_file = args.prompt_file
    model = args.model
    n = args.n
    temp = args.temp
    reponse_root = args.response_root
    os.makedirs(reponse_root, exist_ok=True)
    
    all_messages = get_all_messages(prompt_file)
    for idx, messages in tqdm(enumerate(all_messages)):
        result_path = os.path.join(reponse_root, f'{idx}.json')
        if os.path.isfile(result_path):
            continue
        _ = handle_query(messages, model, n, temp, result_path)


if __name__ == '__main__':
    args = None
    main(args)