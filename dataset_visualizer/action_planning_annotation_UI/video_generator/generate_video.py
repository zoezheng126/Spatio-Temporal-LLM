import os
import cv2
import json
import subprocess
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import time
import shutil

BASE_PATH = 'EPIC-KITCHENS/frames_rgb_flow/rgb/test'  # ! NEED TO BE UPDATED

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def generate_video_from_entry(entry, base_path):
    """
    Generate a video for a single entry
    
    Args:
        entry: JSON entry containing video metadata
        base_path: Base path for frame images
        mask_directory: Directory containing mask files
    """
    video_id = entry["video_id"]
    start_frame = int(entry["start_frame"])
    stop_frame = int(entry["stop_frame"])
    save_path = entry["save_path"] if "save_path" in entry else "../static/generated_videos"

    print(f"video_id: {video_id}, start_frame: {start_frame}, stop_frame: {stop_frame}")

    video_path = os.path.join(base_path, video_id.split('_')[0], video_id)

    # Video output name
    output_video_name = f"{video_id}_{start_frame}_{stop_frame}.mp4"
    output_video_path = os.path.join(save_path, output_video_name)
    # Frame list
    frame_paths = [
        os.path.join(video_path, f"frame_{i:010d}.jpg")
        for i in range(start_frame, stop_frame + 1)
    ]

    # Check if frames exist
    valid_frames = [frame for frame in frame_paths if os.path.exists(frame)]

    if not valid_frames:
        raise ValueError(f"No valid frames found for {output_video_path}")
    # Process frames
    temp_frames_dir = "temp_frames"
    os.makedirs(temp_frames_dir, exist_ok=True)
    temp_frame_paths = []

    frame_index = 0
    for frame_path in valid_frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to read frame: {frame_path}")
            continue
        else:
            temp_frame_path = os.path.join(temp_frames_dir, f"frame_{frame_index:010d}.jpg")
            cv2.imwrite(temp_frame_path, frame)
            temp_frame_paths.append(temp_frame_path)
            frame_index += 1

    # Create frames list file
    with open("frames_list.txt", "w") as f:
        for temp_frame in temp_frame_paths:
            f.write(f"file '{temp_frame}'\n")

    # Generate video using ffmpeg
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-r", "60", "-f", "concat", "-safe", "0",
        "-i", "frames_list.txt", "-vcodec", "libx264", "-pix_fmt", "yuv420p", 
        output_video_path
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print("FFmpeg stderr:\n", stderr.decode())  # Decode bytes to string
        raise RuntimeError(f"FFmpeg failed with return code {process.returncode}")

    # Wait for the video file to exist
    timeout = 300  # Maximum time to wait in seconds
    start_time = time.time()
    while not os.path.exists(output_video_path):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Video generation timed out after {timeout} seconds")
        time.sleep(0.1)  # Sleep for a short time to avoid excessive CPU usage

    return output_video_name

@app.route('/generate-video', methods=['POST'])
def generate_video():
    data = request.json
    video_name = generate_video_from_entry(data, BASE_PATH)
    print(f'video_name: {video_name}')

    return jsonify({'success': True, 'video_path': video_name})

@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
        if os.path.exists("frames_list.txt"):
            os.remove("frames_list.txt")
        temp_frames_dir = "temp_frames"
        if os.path.exists(temp_frames_dir):
            for file in os.listdir(temp_frames_dir):
                file_path = os.path.join(temp_frames_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.rmdir(temp_frames_dir)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":

    app.run(port=5000) 

