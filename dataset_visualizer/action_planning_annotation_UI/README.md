# Annotation Flask App

## Setup

1. Install dependencies:
   ```bash
   pip install flask werkzeug
   ```

2. Place your groundtruth JSON file at:
   `../datasets/WideRefer/data/val_all_question_groundtruth_updated_opencv.json`

3. Place your frames in:
   `annotation_flask_app/static/frames/<video_id>/frame_XXXXXXXXXX.jpg`
   (e.g., `annotation_flask_app/static/frames/P01_11/frame_0000000001.jpg`)

4. Run the app:
   ```bash
   python app.py
   ```

5. Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Features
- User login/resume system
- Two frame sequence players for annotation
- Editable movement field
- Action list, next action, and video_action display
- Progress saved per user in `user_data/`

## Notes
- Make sure the frame images are accessible in the static directory as described.
- The app saves user progress in JSON files under `user_data/`. 