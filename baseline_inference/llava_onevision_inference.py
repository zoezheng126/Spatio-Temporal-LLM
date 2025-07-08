# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python action_planning_inference.py

import os
import json
import copy
import numpy as np
from PIL import Image
import torch
import warnings

from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates

from torch.cuda.amp import autocast
from tqdm import tqdm


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
def get_jpg_filenames(txt_dir):
    return [f.replace('.txt', '.jpg') for f in os.listdir(txt_dir) if f.endswith('.txt')]

def load_frames_from_dir(rgb_dir, video_id, start_frame, end_frame, fps=50, max_frames_num=32, should_sample=True, pose_dir=None):
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
        frame_files = sorted(get_jpg_filenames(c2ws_path))

    total_frame_num = len(frame_files)

    video_dir = os.path.join(rgb_dir, 'test', video_id.split('_')[0], video_id)

    start_frame = f"frame_{start_frame:010d}.jpg"
    end_frame = f"frame_{end_frame:010d}.jpg"


    if should_sample==True:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    else: 
        frame_idx = list(range(total_frame_num))

    sampled_frames = [frame_files[i] for i in frame_idx]
    frames = []
    for frame_name in sampled_frames:
        video_dir = os.path.join(rgb_dir, 'test', video_id.split('_')[0], video_id)
        frame_path = os.path.join(video_dir, frame_name)
        if not os.path.exists(frame_path):
            video_dir = os.path.join(rgb_dir, 'train', video_id.split('_')[0], video_id)
            frame_path = os.path.join(video_dir, frame_name)
            if not os.path.exists(frame_path):
                print(f'frame_path not exists: {frame_path}')
                continue
        frame = Image.open(frame_path).convert("RGB")  # Ensure the frame is in RGB format
        frame = frame.resize((336, 336))  # Resize to match model input size
        frames.append(np.array(frame))
    
    spare_frames = np.stack(frames)

    frame_time = 0
    video_time = 0
    return spare_frames, frame_time, video_time



# Inference for a single query
def infer_single_query(query, rgb_dir, model, tokenizer, image_processor, conv_template, device, scene_level_recon=True):
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
    pose_dir = query["c2ws"]
    path_to_recon_videoid = pointcloud_postfix.split("/")
    path_to_recon_videoid = "/".join(path_to_recon_videoid[:4])
    
    max_frames_num = 32
    recon_max_frame_num = 25

    # Load query video frames
    frames, frame_time, video_time = load_frames_from_dir(rgb_dir, video_id, start_frame, end_frame, 50, max_frames_num, should_sample=True, pose_dir=pose_dir)
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # Convert to PyTorch format
    # Load reconstruction video frames
    if scene_level_recon:
        recon_video_id = scene_id_to_intrinsics_id[video_id.split("_")[0]]
    else:
        recon_video_id = video_id
    recon_frames, recon_frame_time, recon_video_time = load_frames_from_dir(rgb_dir, recon_video_id, start_frame, end_frame, 50, recon_max_frame_num, should_sample=False, pose_dir=path_to_recon_videoid)
    recon_frames = torch.tensor(recon_frames).permute(0, 3, 1, 2)
    # Concat two videos
    all_frames = torch.cat([recon_frames, frames], dim=0)
    # Preprocess frames
    video = image_processor.preprocess(all_frames, return_tensors="pt")["pixel_values"].to(device).half()
    video = [video]

    # Construct prompt
    time_instruciton = f"The first {recon_max_frame_num} images provide multi-view observations of the current scene the person is in, while the next {max_frames_num} frames depict egocentric actions—please refer to both to answer the question."
    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n{question} Give Explanation and reasoning for your answer. Answer in detail, and be specific. Do not random guess. If you don't know say 'I don't know'."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    print(f'prompt_question: {prompt_question}')
    # Tokenize input and generate response
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    model.config.use_cache = False
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=True,
            temperature=0.8,
            max_new_tokens=64,
        )
    text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    for t in [video, frames, recon_frames, all_frames, input_ids, output_ids]:
        if isinstance(t, torch.Tensor):
            t.cpu()
    del video, frames, recon_frames, all_frames, input_ids, output_ids
    torch.cuda.empty_cache()

    return text_outputs

def main(args):
    # Set CUDA device
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device_map = {"": f"cuda:{args.cuda}"}
    device = f"cuda:{args.cuda}"
    torch.cuda.set_device(args.cuda)

    pcd_folder = args.pcd_folder

    # Load model and processor
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.pretrained, None, args.model_name, torch_dtype=torch.bfloat16, device_map=device_map
    )
    model.eval()
    conv_template = "qwen_1_5"

    # Load queries
    with open(args.json_path, "r") as f:
        queries = json.load(f)

    # Slice based on start and end
    if args.end == -1:
        args.end = len(queries)
    queries = queries[args.start:args.end]

    # Inference loop
    out_json = []
    with autocast(dtype=torch.bfloat16):
        for query in tqdm(queries):
            question = query["conversations"][0]["value"]
            question = question.replace("<image>", "").replace("<pointcloud>", "")
            gt = query["conversations"][1]["value"]
            output = infer_single_query(query, args.rgb_dir, model, tokenizer, image_processor, conv_template, device)
            print(f'output: {output}')
            out_json.append({'pred': output, 'gt': gt, 'query': question})

    # Save results
    try:
        os.makedirs(f'./baseline_result/baseline_llava_onevision', exist_ok=True)
        output_path = f'./baseline_result/baseline_llava_onevision/part_{args.start}_{args.end}.json'
        with open(output_path, "w") as f:
            json.dump(out_json, f)
    except Exception as e:
        with open(f'part_{args.start}_{args.end}.json', "w") as f:
            json.dump(out_json, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index for query slicing")
    parser.add_argument("--end", type=int, default=500, help="End index for query slicing")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index")
    parser.add_argument("--json_path", type=str, default="../REA_dataset/qa_val_1780.json", help="Path to input JSON file")
    parser.add_argument("--rgb_dir", type=str, default='path/to/EPIC-KITCHENS/rgb', help="Path to RGB image directory")
    parser.add_argument("--pretrained", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov", help="Pretrained model path")
    parser.add_argument("--model_name", type=str, default="llava_qwen", help="Model name identifier")
    parser.add_argument("--scene_level_recon", type=bool, default=True, help="Whether to use scene-level reconstruction")
    parser.add_argument("--pcd_folder", type=str, default="/path/to/epic-kitchens-vggt-anyloc-val-scene", help="Path to EPIC-KITCHENS")

    args = parser.parse_args()
    main(args)
