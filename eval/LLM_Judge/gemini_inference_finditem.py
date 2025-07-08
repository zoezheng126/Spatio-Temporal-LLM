import json
import time
import argparse
from google import genai
from google.genai import types

SYSTEM_PROMPT = "Give Explanation and reasoning for your answer. Answer in detail, and be specific. Do not random guess. If you don't know say 'I don't know'."
# Set your OpenAI API key
api_key = "" # ! change to your api key
client = genai.Client(api_key=api_key)

correct_count = 0
wrong_count = 0
invalid_count = 0
total_cost_so_far = 0

# Define the Gemini model to use
model_name = "gemini-2.0-flash" # Or "gemini-2-pro-vision" if your queries involve images

def get_response(prompt, retries=3, delay=2):
    global correct_count, wrong_count, invalid_count, total_cost_so_far
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=500,
                    temperature=0
                )
            )
            elapsed_time = time.time() - start_time

            if hasattr(response, "text") and response.text:
                choice = response.text.strip().lower()
                # Gemini doesn't directly provide token counts or cost in the same way as OpenAI
                # Estimating cost is more complex and depends on the model and input/output length.
                # For simplicity, we'll log the time taken as a proxy for potential cost factors.
                print(f"Time: {elapsed_time:.2f}s | Output: '{choice}'")

                if "correct" in choice or "wrong" in choice:
                    if "correct" in choice:
                        correct_count += 1
                    else:
                        wrong_count += 1
                    return choice
                else:
                    print(f"Attempt {attempt+1}: Unexpected reply '{choice}', retrying...")
                    time.sleep(delay)
            else:
                print(f"Attempt {attempt+1}: No text in response, retrying...")
                time.sleep(delay)

        except Exception as e:
            print(f"Attempt {attempt+1}: Error - {e}, retrying...")
            time.sleep(delay)
    invalid_count += 1
    return "invalid_response"

def build_prompt(pred, gt, query):
    return f"""
You are a helpful and fair evaluator. Your task is to determine whether the predicted answer correctly follows the ground truth answer for a 'Find My Item' query. This task requires identifying the location of a target object and describing how the person can get to it, based on the scene.

Predicted Result: {pred}
Ground Truth Result: {gt}
Query: {query}

Please answer only with "Correct" or "Wrong", based on the following criteria:

- Mark as "Correct" if the predicted answer matches the essential intent and meaning of the ground truth, even if phrased differently.

- The answer must correctly identify the item's location and provide a reasonable description of how to reach it.

- Minor differences in language, additional helpful navigation details, or alternative phrasing are acceptable if the overall meaning is consistent with the ground truth.

- If the predicted answer omits key information, misidentifies the item's location, or gives an implausible or unrelated navigation instruction, mark it as "Wrong".

Only reply with one word: "correct" or "wrong" — no explanation or extra text. If the prediction matches the ground truth, reply "correct". Otherwise, reply "wrong".
"""

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Gemini Inference Script")
    argparse.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file")
    argparse.add_argument("--remove_special_tokens", action='store_false', help="Remove special tokens from the ground truth qa query")

    args = argparse.parse_args()
    result_json_path = args.input_json
    new_path = result_json_path.replace(".json", "_aftergemini_findmyitemflash2.json")


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
        # Recalculate correct_count, wrong_count, and invalid_count from loaded data
        for item in res:
            if item.get("output") == "correct":
                correct_count += 1
            elif item.get("output") == "wrong":
                wrong_count += 1
            else:
                invalid_count += 1
        print(f"Resuming from index {start_idx} with counts: Correct: {correct_count}, Wrong: {wrong_count}, Invalid: {invalid_count}.")
    except FileNotFoundError:
        res = []
        start_idx = 0
        print("Starting new processing.")

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

        if qa_item['metadata']['question_type'] != 'find_my_item':
            print(f"Question type mismatch: {qa_item['metadata']['question_type']} != find_my_item")
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
        
        time.sleep(0.1)

    print("Finished processing all entries.")
    print(f"Correct: {correct_count}, Wrong: {wrong_count}, Invalid: {invalid_count}")
    print(f"The sum of should be {correct_count + wrong_count + invalid_count} and the length of the result is {len(res)}")
    # Avoid division by zero if no calls were made
    total_evaluated = correct_count + wrong_count + invalid_count
    if total_evaluated > 0:
        print(f"Correct rate: {correct_count / total_evaluated * 100:.2f}%")
    else:
        print("No API calls were successfully made.")
    # Note: Cost calculation is not directly available with the Gemini API in the same way as OpenAI.
    print("Cost information is not directly available with the Gemini API.")