from openai import OpenAI
import json
import time
from openai import OpenAIError
from tqdm import tqdm
import argparse
SYSTEM_PROMPT = "Give Explanation and reasoning for your answer. Answer in detail, and be specific. Do not random guess. If you don't know say 'I don't know'."
# Set your OpenAI API key
api_key = "" # ! change to your api key
client = OpenAI(api_key=api_key)

correct_count = 0
wrong_count = 0
invalid_count = 0
total_cost_so_far = 0

def get_response(prompt, retries=3, delay=2):
    global correct_count, wrong_count, invalid_count, total_cost_so_far
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            elapsed_time = time.time() - start_time

            choice = response.choices[0].message.content.strip().lower()
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Cost calculation (GPT-4o)
            input_cost = prompt_tokens * 0.0006
            output_cost = completion_tokens * 0.0024
            total_cost = (input_cost + output_cost) / 1000  # Convert to dollars
            total_cost_so_far += total_cost

            print(f"Time: {elapsed_time:.2f}s | Tokens: input={prompt_tokens}, output={completion_tokens}, total={total_tokens} | Cost: ${total_cost:.6f} | Total Cost So Far: ${total_cost_so_far:.6f}")

            if "correct" in choice or "wrong" in choice:
                if "correct" in choice:
                    correct_count += 1
                else:
                    wrong_count += 1
                return choice
            else:
                print(f"Attempt {attempt+1}: Unexpected reply '{choice}', retrying...")
                time.sleep(delay)
        except Exception as e:
            print(f"Attempt {attempt+1}: Error - {e}, retrying...")
            time.sleep(delay)
    invalid_count += 1
    return "invalid_response"

def build_prompt(pred, gt, query):
    return f"""
You are a helpful and fair evaluator. Your task is to determine whether the predicted answer correctly follows the ground truth answer for a relative distance query. This task involves comparing the person’s distance to a specific object during two different actions, based on the scene.

Predicted Result: {pred}
Ground Truth Result: {gt}
Query: {query}

Please answer only with "Correct" or "Wrong", based on the following criteria:

- Mark as "Correct" if the predicted answer accurately conveys the relative distance relationship described in the ground truth, even if expressed with different wording.

- The prediction must clearly indicate which action places the person closer (or if the distances are about the same).

- Minor wording variations or additional clarifications are acceptable as long as the core spatial relationship is preserved.

- If the prediction contradicts or misses the comparison stated in the ground truth, mark it as "Wrong".

- Move closer and move further sometimes can be similar to remain the same distance, based on the context of the prediction, give resenable judge. 

Only reply with one word: "correct" or "wrong" — no explanation or extra text. If the prediction matches the ground truth, reply "correct". Otherwise, reply "wrong".
"""

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="ChatGPT Inference Script")
    argparse.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file")
    argparse.add_argument("--remove_special_tokens", action='store_false', help="Remove special tokens from the ground truth qa query")
    args = argparse.parse_args()
    result_json_path = args.input_json
    new_path = result_json_path.replace(".json", "_afteropenai_relativedistance4onew.json")


    # Load the input JSON
    with open(result_json_path, "r") as f:
        data = json.load(f)
    
    # ground truth qa data
    with open("../REA_dataset/qa_val_1757_v20.json", "r") as f:
        qa_data = json.load(f)

    if args.remove_special_tokens:
        for q in qa_data:
            q['conversations'][0]['value'] = q['conversations'][0]['value'].replace("<pointcloud>", "").replace("<image>", "")

    # Try loading progress from partially completed output file
    try:
        with open(new_path, "r") as f:
            res = json.load(f)
        start_idx = len(res)
        print(f"Resuming from index {start_idx}...")
    except FileNotFoundError:
        res = []
        start_idx = 0

    # Main loop
    for idx in range(start_idx, len(data)):
        qa_item = qa_data[idx]
        item = data[idx]
        pred = item["pred"]
        gt = item["gt"]
        query = item["query"]
        query = query.replace(SYSTEM_PROMPT, "")
        if query != qa_item['conversations'][0]['value']:
            print(f"Query mismatch: {query} != {qa_item['conversations'][0]['value']}")
            found = False
            for q in qa_data:
                if query == q['conversations'][0]['value']:
                    found = True
                    qa_item = q
                    break
            if not found:
                print(f"Query not found in qa_data: {query}")
                continue

        if qa_item['metadata']['question_type'] != 'relative_distance':
            print(f"Question type mismatch: {qa_item['metadata']['question_type']} != relative_distance")
            continue

        gt = qa_item['conversations'][1]['value']
        prompt = build_prompt(pred, gt, query)
        output = get_response(prompt)

        res.append({
            "pred": pred,
            "gt": gt,
            "query": query,
            "output": output
        })

        # Save progress immediately
        with open(new_path, "w") as f:
            json.dump(res, f, indent=4)

        print(f"[{idx+1}/{len(data)}] - Output: {output}")
        
        time.sleep(1)

    print("Finished processing all entries.")
    print(f"Correct: {correct_count}, Wrong: {wrong_count}, Invalid: {invalid_count}")
    print(f"The sum of should be {correct_count + wrong_count + invalid_count} and the length of the result is {len(res)}")
    print(f"Correct rate: {correct_count / (correct_count + wrong_count + invalid_count) * 100:.2f}%")
