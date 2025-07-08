# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python action_planning_inference.py

import os
import json
import copy
import numpy as np
from PIL import Image
import torch
import warnings

from torch.cuda.amp import autocast
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from datetime import datetime


warnings.filterwarnings("ignore")

scene_id_to_intrinsics_id = {
 'P01': 'P01_01', 'P02': 'P02_01', 'P03': 'P03_02', 'P04': 'P04_01',
 'P05': 'P05_01', 'P06': 'P06_03', 'P07': 'P07_04', 'P08': 'P08_01',
 'P09': 'P09_02', 'P10': 'P10_03', 'P11': 'P11_04', 'P12': 'P12_01',
 'P13': 'P13_01', 'P14': 'P14_01', 'P15': 'P15_02', 'P16': 'P16_01',
 'P17': 'P17_03', 'P18': 'P18_07', 'P19': 'P19_06', 'P20': 'P20_03',
 'P21': 'P21_01', 'P22': 'P22_02', 'P23': 'P23_02', 'P24': 'P24_01',
 'P25': 'P25_04', 'P26': 'P26_02', 'P27': 'P27_01', 'P28': 'P28_01',
 'P29': 'P29_01', 'P30': 'P30_02'
}

# some hyperparameters
resized_height = 336
resized_width = 336
# ! we are now using multi images with scaled size to 336x336

def get_jpg_filenames(txt_dir):
    return [f.replace('.txt', '.jpg') for f in os.listdir(txt_dir) if f.endswith('.txt')]

def load_frames_from_dir(rgb_dir, video_id, start_frame, end_frame, fps=50, max_frames_num=32, should_sample=True, pose_dir=None, pcd_folder=None):
    """
    Load frames from a directory, uniformly sampling at 1 FPS. If the number of sampled frames
    exceeds `max_frames_num`, uniformly sample `max_frames_num` frames.

    Args:
        rgb_dir (str): Path to the directory containing the video frames.
        video_id (str): ID of the video (used to locate subdirectory).
        start_frame (str): Filename of the start frame (e.g., "frame_0000000001.jpg").
        end_frame (str): Filename of the end frame (e.g., "frame_0000000113.jpg").
        fps (int): Target sampling rate in frames per second (default=1).
        max_frames_num (int): Maximum number of frames to sample.

    Returns:
        np.ndarray: Sampled frames as a NumPy array (N, H, W, C).
        str: Frame time information in seconds (comma-separated).
        float: Total video time in seconds.
    """
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if pose_dir is not None:
        if "recon" in pose_dir:
            c2ws_path = os.path.join(pcd_folder, pose_dir, "train", "poses")
        else:
            c2ws_path = os.path.join(pcd_folder, pose_dir)
        print(f'c2ws_path: {c2ws_path}')
        # a list of filenames baseline model used for query video
        frame_files = sorted(get_jpg_filenames(c2ws_path))

    total_frame_num = len(frame_files)

    video_dir = os.path.join(rgb_dir, 'test', video_id.split('_')[0], video_id)
    if not os.path.exists(video_dir):
        video_dir = os.path.join(rgb_dir, 'train', video_id.split('_')[0], video_id)
    # dataset_all_frames = sorted(os.listdir(video_dir))  # Ensure frames are loaded in order

    # start_frame = f"frame_{start_frame:010d}.jpg"
    # end_frame = f"frame_{end_frame:010d}.jpg"
    # Calculate the new fps using original frames
    # start_idx = dataset_all_frames.index(start_frame)
    # end_idx = dataset_all_frames.index(end_frame) + 1
    # selected_frames = dataset_all_frames[start_idx:end_idx]

    if should_sample==True:
        # ! total frame num is images that I can choose from
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        # get list indexes in the frame_files
        frame_idx = uniform_sampled_frames.tolist()
    else:
        frame_idx = list(range(len(frame_files)))

    # Compute the total video time
    # total_frames = len(selected_frames)
    # original_total_frames = int(end_idx) - int(start_idx) + 1
    # original_fps = 50  # The directory is sampled at 50 FPS
    # video_time = original_total_frames / original_fps
    # current_fps = max_frames_num / video_time

    # Get sampled frames and their corresponding timestamps
    sampled_frames = [frame_files[i] for i in frame_idx]
    # frame_time = ",".join([f"{i / current_fps:.2f}s" for i in frame_idx])
    frame_time = 0
    video_time = 0

    # Load and preprocess the sampled frames
    frames = []
    frame_paths = []
    for frame_name in sampled_frames:
        video_dir = os.path.join(rgb_dir, 'test', video_id.split('_')[0], video_id)
        frame_path = os.path.join(video_dir, frame_name)
        if not os.path.exists(frame_path):
            video_dir = os.path.join(rgb_dir, 'train', video_id.split('_')[0], video_id)
            frame_path = os.path.join(video_dir, frame_name)
            if not os.path.exists(frame_path):
                print(f'frame_path not exists: {frame_path}')
                continue
        frame_paths.append(frame_path)
    
    spare_frames = []

    return spare_frames, frame_time, video_time, frame_paths

# Inference for a single query
def infer_single_query(query, rgb_dir, model, processor, device, scene_level_recon=True, pcd_folder=None):
    """
    Perform inference for a single query.

    Args:
        query (dict): Query object containing details like start_frame, end_frame, etc.
        rgb_dir (str): Path to the directory containing the video frames.
        model, tokenizer, image_processor: Pretrained model and associated utilities.
        conv_template (str): Conversation template.
        device (str): Device to run inference on.

    Returns:
        str: Model output.
    """
    video_id = query["scene_id"]
    start_frame = query["metadata"]["query_start_frame"]
    end_frame = query["metadata"]["query_stop_frame"]
    pointcloud_postfix = query["pointcloud"]
    question = query["conversations"][0]["value"]
    question = question.replace("<image>", "")
    question = question.replace("<pointcloud>", "")
    query_id = query["id"]

    # question = "describe the images?" # ! change it back

    pose_dir = query["c2ws"]
    path_to_recon_videoid = pointcloud_postfix.split("/")
    path_to_recon_videoid = "/".join(path_to_recon_videoid[:4])
    print(f'path to recon pose: {path_to_recon_videoid}')
    # full_path_to_recon_videoid = os.path.join(pcd_folder, path_to_recon_videoid)
    
    max_frames_num = 32
    recon_max_frame_num = 25

    # Construct frame paths
    # Load query video frame paths
    _, _, _, frame_paths = load_frames_from_dir(rgb_dir, video_id, start_frame, end_frame, 50, max_frames_num, should_sample=True, pose_dir=pose_dir, pcd_folder=pcd_folder)
    # Load reconstruction video frames
    if scene_level_recon:
        recon_video_id = scene_id_to_intrinsics_id[video_id.split("_")[0]]
    else:
        recon_video_id = video_id
    _, _, _, recon_frame_paths = load_frames_from_dir(rgb_dir, recon_video_id, start_frame, end_frame, 50, recon_max_frame_num, should_sample=False, pose_dir=path_to_recon_videoid, pcd_folder=pcd_folder)
    all_frame_paths = recon_frame_paths + frame_paths 
    # all_frame_paths = frame_paths # ! change it back
    # build image content
    image_content = []
    for path in all_frame_paths:
        temp = {}
        temp["type"] = "image"
        temp["image"] = path
        temp["resized_height"] = resized_height
        temp["resized_width"] = resized_width
        image_content.append(temp)

    # Construct prompt
    time_instruciton = f"The first {recon_max_frame_num} images provide multi-view observations of the current scene the person is in, while the next {max_frames_num} frames depict egocentric actions—please refer to both to answer the question. {question}. Give Explanation and reasoning for your answer. Answer in detail, and be specific. Do not random guess. If you don't know say 'I don't know'."
    # build text content
    text_content = []
    text_content.append({"type": "text", "text": time_instruciton})

    # build all content
    all_content = image_content + text_content

    # all_content = text_content # ! change it back

    messages = [
    {
        "role": "user",
        "content": all_content,
    }
    ]
    

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # ! change it back
    # for i in range(len(image_inputs)):
    #     query_image_verify_folder = "temp_image_verify"
    #     if not os.path.exists(query_image_verify_folder):
    #         os.makedirs(query_image_verify_folder)
    #     cur_query_folder = os.path.join(query_image_verify_folder, query_id)
    #     if not os.path.exists(cur_query_folder):
    #         os.makedirs(cur_query_folder)
    #     cur_image = image_inputs[i]
    #     # save the image to the folder
    #     image_origin_name = all_frame_paths[i].split("/")[-1]
    #     cur_image_path = os.path.join(cur_query_folder, image_origin_name)
    #     cur_image.save(cur_image_path)
    #     print(f"Saved image {i} to {cur_image_path}")

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text

# Main script
if __name__ == "__main__":

    device_map = {"": "cuda:0"} # ! change to your GPU
    device = "cuda:0" # ! change to your GPU
    json_path = "../REA_dataset/qa_val_1757_v20.json"   # ! change to your json path
    rgb_dir = '/path/to/EPIC-KITCHENS/rgb'   # ! change to your rgb directory
    pcd_folder = "/path/to/epic-kitchens-vggt-anyloc-val-scene"   # ! change to your pcd folder
    scene_level_recon = True   # ! default is True 

    # Load the pretrained model
    model = None
    processor = None
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map=device_map
    )
    processor  = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model.to(device, non_blocking=True)
    model.eval()

    # Step2 - Load queries
    with open(json_path, "r") as f:
        queries = json.load(f)

    # Perform inference for all queries
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'./baseline_result/baseline_qwen2vl7binstruct', exist_ok=True)
    save_path = f'./baseline_result/baseline_qwen2vl7binstruct/all_{date_str}.json'

    out_json = []

    print(f"model dtype: {model.dtype}")
    with autocast(dtype=model.dtype):
        for query in tqdm(queries):
            try:
                question = query["conversations"][0]["value"].replace("<image>", "").replace("<pointcloud>", "")
                gt = query["conversations"][1]["value"]

                output = infer_single_query(query, rgb_dir, model, processor, device, scene_level_recon=scene_level_recon, pcd_folder=pcd_folder)
                print(f'output: {output}')

                output = output[0]  # ! here the output is a list of length 1

                record = {'pred': output, 'gt': gt, 'query': question}
                out_json.append(record)

                # Append after each successful inference
                with open(save_path, "w") as f:
                    json.dump(out_json, f, indent=2)

            except Exception as e:
                print(f"[ERROR] Failed to process query: {e}")
                continue

