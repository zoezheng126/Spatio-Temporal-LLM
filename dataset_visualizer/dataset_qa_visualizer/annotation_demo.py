import gradio as gr
import json
import cv2
import os
import uuid
import shutil
import requests
from datetime import datetime


GLB_DIR="./glbs"
RGB_DIR="EPIC-KITCHENS/frames_rgb_flow/rgb" # ! NEED TO BE UPDATED
# Load your JSON file
with open("json/qa_val_50_find_my_item.json") as f:
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
        return None, None, "0/0", None, "", "", "", "", "", ""
    entry = entries[index]
    print(f'index: {index}, entry: {entry}')
    video_clip, video_dir = get_video_clip(entry["scene_id"], entry["metadata"]["query_start_frame"], entry["metadata"]["query_stop_frame"])
    # Get the conversation from the entry
    conversation = entry.get("conversations", [])
    question = next((conv["value"] for conv in conversation if conv["from"] == "human"), "")
    answer = next((conv["value"] for conv in conversation if conv["from"] == "gpt"), "")
    question_type = entry["metadata"].get("question_type", "")
    glb = entry["pointcloud"]
    scene_id = entry.get("scene_id", "")
    entry_id = entry.get("id", "")
    
    # Check if this entry has been annotated by the current user
    annotated_answer = answer  # Default to original answer
    if "annotations" in entry:
        for annotation in entry["annotations"]:
            if annotation.get("username") == getattr(show_entry, 'current_user', ''):
                annotated_answer = annotation.get("annotated_answer", answer)
                break
    
    return video_clip, glb, f"{index+1}/{len(entries)}", video_dir, question, answer, question_type, scene_id, annotated_answer, entry_id

def auto_save_entry(username, annotated_answer, entry_id):
    """Auto-save the current entry without user interaction"""
    if not username.strip() or not annotated_answer.strip():
        return  # Don't save if username or answer is empty
    
    try:
        # Save to a single file named by username
        output_filename = f"json/qa_val_50_annotated_{username}.json"
        
        # Load existing annotated entries or start with original structure
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                existing_entries = json.load(f)
        else:
            # Start with a copy of the original structure
            existing_entries = json.loads(json.dumps(all_entries))
        
        # Find the entry in the existing entries and update it
        entry_updated = False
        for entry in existing_entries:
            if entry.get("id") == entry_id:
                # Get the original answer before updating
                original_answer = next((conv["value"] for conv in entry.get("conversations", []) if conv["from"] == "gpt"), "")
                
                # Update the originalanswer in the conversations
                for conv in entry.get("conversations", []):
                    if conv.get("from") == "gpt":
                        conv["value"] = annotated_answer
                        break
                
                # Add annotation metadata
                if "annotations" not in entry:
                    entry["annotations"] = []
                
                annotation = {
                    "username": username,
                    "annotated_answer": annotated_answer,
                    "timestamp": datetime.now().isoformat(),
                    "original_answer": original_answer
                }
                
                entry["annotations"].append(annotation)
                entry_updated = True
                break
        
        # Save the updated data
        with open(output_filename, 'w') as f:
            json.dump(existing_entries, f, indent=2)
        
    except Exception as e:
        print(f"Auto-save error: {str(e)}")

def next_entry(current_idx, current_dir, entries, username, annotated_answer, entry_id):
    # Auto-save current entry before navigating
    auto_save_entry(username, annotated_answer, entry_id)
    
    # Delete the previous video directory if it exists
    if current_dir and os.path.exists(current_dir):
        print(f"Deleting previous video directory: {current_dir}")
        shutil.rmtree(current_dir)
    
    if not entries:
        return None, None, "0/0", 0, None, "", "", "", "", "", ""

    idx = (current_idx + 1) % len(entries)
    video, glb, counter, new_dir, question, answer, question_type, scene_id, annotated_answer, entry_id = show_entry(idx, entries)
    return video, glb, counter, idx, new_dir, question, answer, question_type, scene_id, annotated_answer, entry_id

def jump_to_entry(target_idx, current_dir, entries, username, annotated_answer, entry_id):
    # Auto-save current entry before navigating
    auto_save_entry(username, annotated_answer, entry_id)
    
    # Delete the previous video directory if it exists
    if current_dir and os.path.exists(current_dir):
        print(f"Deleting previous video directory: {current_dir}")
        shutil.rmtree(current_dir)

    if not entries:
        return None, None, "0/0", 0, None, "", "", "", "", "", ""

    # Ensure the index is within bounds
    idx = max(0, min(target_idx, len(entries) - 1))
    video, glb, counter, new_dir, question, answer, question_type, scene_id, annotated_answer, entry_id = show_entry(idx, entries)
    return video, glb, counter, idx, new_dir, question, answer, question_type, scene_id, annotated_answer, entry_id

def save_annotation(username, annotated_answer, current_idx, entries, entry_id):
    """Save the annotated answer to the JSON file"""
    if not username.strip():
        return "Please enter a username before saving."
    
    if not annotated_answer.strip():
        return "Please enter an annotated answer before saving."
    
    if not entries or current_idx >= len(entries):
        return "No valid entry to save."
    
    try:
        # Save to a single file named by username
        output_filename = f"json/qa_val_50_annotated_{username}.json"
        
        # Load existing annotated entries or start with original structure
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                existing_entries = json.load(f)
        else:
            # Start with a copy of the original structure
            existing_entries = json.loads(json.dumps(all_entries))
        
        # Find the entry in the existing entries and update it
        entry_updated = False
        for entry in existing_entries:
            if entry.get("id") == entry_id:
                # Get the original answer before updating
                original_answer = next((conv["value"] for conv in entry.get("conversations", []) if conv["from"] == "gpt"), "")
                
                # Update the GPT answer in the conversations
                for conv in entry.get("conversations", []):
                    if conv.get("from") == "gpt":
                        conv["value"] = annotated_answer
                        break
                
                # Add annotation metadata
                if "annotations" not in entry:
                    entry["annotations"] = []
                
                annotation = {
                    "username": username,
                    "annotated_answer": annotated_answer,
                    "timestamp": datetime.now().isoformat(),
                    "original_answer": original_answer
                }
                
                entry["annotations"].append(annotation)
                entry_updated = True
                break
        
        # Save the updated data
        with open(output_filename, 'w') as f:
            json.dump(existing_entries, f, indent=2)
        
        return f"Entry saved successfully to {output_filename}"
    
    except Exception as e:
        return f"Error saving entry: {str(e)}"

def update_current_user(username):
    """Update the current user for the show_entry function"""
    show_entry.current_user = username
    return username

video_clip, glb_path, counter_val, initial_dir, initial_question, initial_answer, initial_question_type, initial_scene_id, initial_annotated_answer, initial_entry_id = show_entry(0, all_entries)

with gr.Blocks() as demo:
    gr.Markdown("# Dataset Annotator")
    
    # User information section
    with gr.Row():
        username = gr.Textbox(label="Username", placeholder="Enter your username", interactive=True)
    
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
            original_answer = gr.Textbox(label="Original Answer", value=initial_answer, interactive=False)
            annotated_answer = gr.Textbox(label="Your Annotated Answer", value=initial_annotated_answer, interactive=True, lines=3)
    
    with gr.Row():
        counter = gr.Textbox(label="Entry", interactive=False, value=counter_val)
        entry_number = gr.Number(label="Jump to Entry (0-" + str(len(all_entries)-1) + ")", value=0, interactive=True)
        jump_btn = gr.Button("Jump")
        next_btn = gr.Button("Next")
    
    with gr.Row():
        save_btn = gr.Button("Save Entry", variant="primary")
        save_status = gr.Textbox(label="Save Status", interactive=False, value="")
    
    # State variables
    idx_state = gr.State(0)
    dir_state = gr.State(value=initial_dir)
    entries_state = gr.State(value=all_entries)
    scene_id_state = gr.State(value=initial_scene_id)
    entry_id_state = gr.State(value=initial_entry_id)

    def on_next(idx, current_dir, entries, username, annotated_answer, entry_id):
        return next_entry(idx, current_dir, entries, username, annotated_answer, entry_id)

    def on_jump(target_idx, current_dir, entries, username, annotated_answer, entry_id):
        return jump_to_entry(target_idx, current_dir, entries, username, annotated_answer, entry_id)

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
            return None, None, "0/0", 0, None, "", "", "", "", "", "", jump_label_update
        
        video, glb, counter, new_dir, question, answer, question_type, scene_id, annotated_answer, entry_id = show_entry(0, filtered_entries)
        
        new_max_idx = len(filtered_entries) - 1
        jump_label_update = gr.update(label=f"Jump to Entry (0-{new_max_idx})", value=0, interactive=True)
        
        return video, glb, counter, 0, new_dir, question, answer, question_type, scene_id, annotated_answer, entry_id, filtered_entries, jump_label_update

    def on_save(username, annotated_answer, current_idx, entries, entry_id):
        return save_annotation(username, annotated_answer, current_idx, entries, entry_id)

    # Event handlers
    next_btn.click(on_next, inputs=[idx_state, dir_state, entries_state, username, annotated_answer, entry_id_state], 
                  outputs=[video, glb, counter, idx_state, dir_state, question, original_answer, question_type, scene_id_state, annotated_answer, entry_id_state])
    
    jump_btn.click(on_jump, inputs=[entry_number, dir_state, entries_state, username, annotated_answer, entry_id_state], 
                  outputs=[video, glb, counter, idx_state, dir_state, question, original_answer, question_type, scene_id_state, annotated_answer, entry_id_state])
    
    q_type_filter.change(on_filter_change, 
                         inputs=[q_type_filter, dir_state], 
                         outputs=[video, glb, counter, idx_state, dir_state, question, original_answer, question_type, scene_id_state, annotated_answer, entry_id_state, entries_state, entry_number])
    
    save_btn.click(on_save, inputs=[username, annotated_answer, idx_state, entries_state, entry_id_state], 
                  outputs=[save_status])

demo.launch(share=True) 