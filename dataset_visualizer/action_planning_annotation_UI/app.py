from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import json
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this in production

DATA_DIR = '/data/haozhen/4dllm/4D-LLM/epic-fields/scripts/action_planning'  # Adjust as needed
USER_DATA_DIR = './user_data'  # Where user progress is saved
# GT_FILE = os.path.join(DATA_DIR, 'val_all_question_groundtruth_interval_10_600_sampled.json')
GT_FILE = "/data/haozhen/4dllm/4D-LLM/epic-fields/scripts/action_planning/val_all_question_groundtruth_interval_10_600_sampled.json"

# Ensure user data dir exists
os.makedirs(USER_DATA_DIR, exist_ok=True)

import shutil
def prepare_user_frames(username, prefix, video_id, start_frame, end_frame, ann_start_frame, ann_end_frame):
    src_dir = f'/data/beitong2/EPIC-KITCHENS-ORIGIN_1TB/frames_rgb_flow/rgb/test/{prefix}/{video_id}'
    dst_dir = os.path.join('./static', 'frames', username)
    # Clean up old frames
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    # Copy needed frames
    for i in range(int(start_frame.split('.')[0].split('_')[1]), int(end_frame.split('.')[0].split('_')[1]) + 1):
        fname = f'frame_{int(i):010d}.jpg'
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
    for i in range(int(ann_start_frame.split('.')[0].split('_')[1]), int(ann_end_frame.split('.')[0].split('_')[1]) + 1):
        fname = f'frame_{int(i):010d}.jpg'
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)

def load_groundtruth():
    with open(GT_FILE, 'r') as f:
        return json.load(f)

def get_user_file(username):
    return os.path.join(USER_DATA_DIR, f'{secure_filename(username)}.json')

def load_user_progress(username):
    user_file = get_user_file(username)
    if os.path.exists(user_file):
        with open(user_file, 'r') as f:
            return json.load(f)
    else:
        return None

def save_user_progress(username, data):
    user_file = get_user_file(username)
    with open(user_file, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/', methods=['GET', 'POST'])
def login():
    print(request.method)
    if request.method == 'POST':
        username = request.form['username']
        session['username'] = username
        return redirect(url_for('annotate'))
    return render_template('login.html')

@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    gt_data = load_groundtruth()
    user_data = load_user_progress(username)
    if not user_data:
        user_data = {'progress': 0, 'annotations': gt_data}
    idx = user_data['progress']
    if request.method == 'POST':
        # Save annotation
        movement = request.form['movement']
        user_data['annotations'][idx]['motion']['movement'] = movement
        user_data['progress'] = min(idx + 1, len(gt_data) - 1)
        save_user_progress(username, user_data)
        return redirect(url_for('annotate'))
    entry = user_data['annotations'][idx]

    # --- Add this call before rendering the template ---
    video_id = entry['video_id']
    prefix = video_id.split('_')[0]
    start_frame = entry['start_frame']
    end_frame = entry['end_frame']
    ann_start_frame = entry['img_input'][0]
    ann_end_frame = entry['img_input'][1]

    current_entry = {
        "video_id": video_id,
        "start_frame": start_frame.split('.')[0].split('_')[1],
        "stop_frame": end_frame.split('.')[0].split('_')[1],
    }
    video_name_finished_actions = requests.post("http://localhost:5000/generate-video", json=current_entry)
    entry['video_1'] = os.path.join('/static/generated_videos', video_name_finished_actions.json()['video_path'])

    cleanup_response = requests.post("http://localhost:5000/cleanup")
    if cleanup_response.status_code != 200:
        print("Cleanup failed:", cleanup_response.json().get('error', 'Unknown error'))
    current_entry = {
        "video_id": video_id,
        "start_frame": ann_start_frame.split('.')[0].split('_')[1],
        "stop_frame": ann_end_frame.split('.')[0].split('_')[1],
    }
    video_name_transition = requests.post("http://localhost:5000/generate-video", json=current_entry)

    entry['video_2'] = os.path.join('/static/generated_videos', video_name_transition.json()['video_path'])

    cleanup_response = requests.post("http://localhost:5000/cleanup")
    if cleanup_response.status_code != 200:
        print("Cleanup failed:", cleanup_response.json().get('error', 'Unknown error'))
    # prepare_user_frames(username, prefix, video_id, start_frame, end_frame,ann_start_frame, ann_end_frame)
    # ---------------------------------------------------
    video_1_filename = entry['video_1'].replace('/static/', '')
    video_2_filename = entry['video_2'].replace('/static/', '')
    return render_template('annotate.html', 
                           entry=entry, 
                           main_video_path=url_for('static', filename=video_1_filename),
                           input_video_path=url_for('static', filename=video_2_filename),
                           idx=idx, 
                           total=len(gt_data))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 