import gradio as gr
import json
import cv2
import os
import uuid
import shutil
import requests


GLB_DIR="./glbs"
RGB_DIR="EPIC-KITCHENS/frames_rgb_flow/rgb" # ! NEED TO BE UPDATED
# Load your JSON file
with open("json/qa_val_50.json") as f:
    all_entries = json.load(f)

for entry in all_entries:
    entry["video"] = entry["video"].replace("EPIC-KITCHENS/rgb", RGB_DIR)
    # entry["pointcloud"] = entry["pointcloud"].replace("EPIC-KITCHENS", "/data/haozhen/4dllm/data_preprocess/Epic-Field/pc_reconstruction/vggt")
    entry["pointcloud"] = os.path.join(GLB_DIR, entry["scene_id"].split("_")[0] + ".glb")

def get_video_clip(scene_id, start_frame, end_frame):
    # Generate a unique directory name using UUID
    unique_dir = str(uuid.uuid4())
    unique_dir = os.path.join('/data/haozhen/4dllm/4D-LLM/epic-fields/scripts/big_object_navigation/visualizer','video', unique_dir)
    os.makedirs(unique_dir, exist_ok=True)


    # Prepare the request payload
    payload = {
        "video_id": scene_id,
        "start_frame": str(start_frame),
        "stop_frame": str(end_frame),
        "save_path": unique_dir
    }

    # Send POST request to the video generation server
    try:
        response = requests.post("http://localhost:5000/generate-video", json=payload)
        if response.status_code == 200:
            # Save the received video to the unique directory
            video_path = os.path.join(unique_dir, response.json()['video_path'])
            cleanup_response = requests.post("http://localhost:5000/cleanup")
            if cleanup_response.status_code != 200:
                print("Cleanup failed:", cleanup_response.json().get('error', 'Unknown error'))
            print(f"Video saved to {video_path}")
            return video_path, unique_dir
        else:
            raise Exception(f"Server returned status code {response.status_code}")
    except Exception as e:
        print(f"Error generating video clip: {e}")
        return None, None

def show_entry(index, entries):
    if not entries or index >= len(entries):
        # Return default/empty values if no entries or index is out of bounds
        return None, None, "0/0", None, "", "", ""
    entry = entries[index]
    print(f'index: {index}, entry: {entry}')
    video_clip, video_dir = get_video_clip(entry["scene_id"], entry["metadata"]["query_start_frame"], entry["metadata"]["query_stop_frame"])
    # Get the conversation from the entry
    conversation = entry.get("conversations", [])
    question = next((conv["value"] for conv in conversation if conv["from"] == "human"), "")
    answer = next((conv["value"] for conv in conversation if conv["from"] == "gpt"), "")
    question_type = entry["metadata"].get("question_type", "")
    glb = entry["pointcloud"]
    return video_clip, glb, f"{index+1}/{len(entries)}", video_dir, question, answer, question_type



def next_entry(current_idx, current_dir, entries):
    # Delete the previous video directory if it exists
    if current_dir and os.path.exists(current_dir):
        print(f"Deleting previous video directory: {current_dir}")
        shutil.rmtree(current_dir)
    
    if not entries:
        return None, None, "0/0", 0, None, "", "", ""

    idx = (current_idx + 1) % len(entries)
    video, glb, counter, new_dir, question, answer, question_type = show_entry(idx, entries)
    return video, glb, counter, idx, new_dir, question, answer, question_type

def jump_to_entry(target_idx, current_dir, entries):
    # Delete the previous video directory if it exists
    if current_dir and os.path.exists(current_dir):
        print(f"Deleting previous video directory: {current_dir}")
        shutil.rmtree(current_dir)

    if not entries:
        return None, None, "0/0", 0, None, "", "", ""

    # Ensure the index is within bounds
    idx = max(0, min(target_idx, len(entries) - 1))
    video, glb, counter, new_dir, question, answer, question_type = show_entry(idx, entries)
    return video, glb, counter, idx, new_dir, question, answer, question_type

video_clip, glb_path, counter_val, initial_dir, initial_question, initial_answer, initial_question_type = show_entry(0, all_entries)

with gr.Blocks() as demo:
    gr.Markdown("# Dataset Visualizer")
    question_types = ["All", "relative_direction", "relative_distance", "furniture_affordance", "action_planning", "find_my_item"]
    with gr.Row():
        with gr.Column(scale=1):
            q_type_filter = gr.Radio(question_types, label="Filter by Question Type", value="All")

    with gr.Row():
        video = gr.Video(label="Video", interactive=False, value=video_clip, height=520)
        glb = gr.Model3D(label="PointCloud (GLB)", value=glb_path, height=520, zoom_speed=0.5, pan_speed=0.5)
    with gr.Row():
        with gr.Column():
            question_type = gr.Textbox(label="Question Type", value=initial_question_type, interactive=False)
            question = gr.Textbox(label="Question", value=initial_question, interactive=False)
            answer = gr.Textbox(label="Answer", value=initial_answer, interactive=False)
    with gr.Row():
        counter = gr.Textbox(label="Entry", interactive=False, value=counter_val)
        entry_number = gr.Number(label="Jump to Entry (0-" + str(len(all_entries)-1) + ")", value=0, interactive=True)
        jump_btn = gr.Button("Jump")
        next_btn = gr.Button("Next")
    idx_state = gr.State(0)
    dir_state = gr.State(value=initial_dir)
    entries_state = gr.State(value=all_entries)

    def on_next(idx, current_dir, entries):
        return next_entry(idx, current_dir, entries)

    def on_jump(target_idx, current_dir, entries):
        return jump_to_entry(target_idx, current_dir, entries)

    def on_filter_change(q_type, current_dir):
        if current_dir and os.path.exists(current_dir):
            print(f"Deleting previous video directory: {current_dir}")
            shutil.rmtree(current_dir)
        
        if q_type == "All":
            filtered_entries = all_entries
        else:
            filtered_entries = [e for e in all_entries if e.get("metadata", {}).get("question_type") == q_type]

        if not filtered_entries:
            jump_label_update = gr.update(label="Jump to Entry (0-0)", value=0, interactive=False)
            return None, None, "0/0", 0, None, "", "", "", [], jump_label_update
        
        video, glb, counter, new_dir, question, answer, question_type = show_entry(0, filtered_entries)
        
        new_max_idx = len(filtered_entries) - 1
        jump_label_update = gr.update(label=f"Jump to Entry (0-{new_max_idx})", value=0, interactive=True)
        
        return video, glb, counter, 0, new_dir, question, answer, question_type, filtered_entries, jump_label_update


    next_btn.click(on_next, inputs=[idx_state, dir_state, entries_state], outputs=[video, glb, counter, idx_state, dir_state, question, answer, question_type])
    jump_btn.click(on_jump, inputs=[entry_number, dir_state, entries_state], outputs=[video, glb, counter, idx_state, dir_state, question, answer, question_type])
    q_type_filter.change(on_filter_change, 
                         inputs=[q_type_filter, dir_state], 
                         outputs=[video, glb, counter, idx_state, dir_state, question, answer, question_type, entries_state, entry_number])

demo.launch(share=True, allowed_paths = ["/data/haozhen/4dllm/data_preprocess/Epic-Field/pc_reconstruction/vggt"])