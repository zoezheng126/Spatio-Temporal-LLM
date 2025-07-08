from llava.model.detector_Vote2Cap_DETR.aligner_our_0519 import LlavaQwenPcdVidLM, LlavaQwenPcdAlignerConfig
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
from torch.cuda.amp import autocast
import torch
import transformers
from torch.utils.data import DataLoader
from llava.train.train_3d_our_0519 import DataArguments, LazySupervisedDataset, DataCollatorForSupervisedDataset
from llava import conversation as conversation_lib
import pdb

from peft import PeftModelForCausalLM
from transformers import AutoConfig

from torch.utils import model_zoo
from accelerate import init_empty_weights
from safetensors import safe_open

from transformers import PretrainedConfig
import os
import json


def move_batch_to_cuda(batch, device="cuda:0"):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
        elif isinstance(value, list):
            # Move each element in the list if it's a tensor
            batch[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        # If there are nested dicts, you could recurse here if needed
    return batch

def remove_first_prefix(name, prefix):
    if name.startswith(prefix):
        return name[len(prefix):]
    return name

if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    gt_json = "path/to/qa_val_1757_v20.json"
    with open(gt_json, 'r') as f:
        gt_data = json.load(f)
    device = "cuda:0"
    torch.cuda.set_device(0)
    pretrained_model_name_or_path = "lmms-lab/LLaVA-Video-7B-Qwen2"

    model_name_or_path = "/work/nvme/bczf/zoezheng126/work_dirs/ft-REA-all-1024npoints-32frames-qwen_32nquery_reav20_0702"
    data_path = "/path/to/LLaVA-NeXT/scripts/video/train/exp_rea_test.yaml"
    data_args = {"frames_upbound": 32, 
                "lazy_preprocess": True, 
                "image_folder":"path/to/datasets",
                "video_folder":"path/to/datasets",
                "pointcloud_folder":"path/to/epic-kitchens-vggt-anyloc-val-scene"
                }
                
    config = LlavaQwenPcdAlignerConfig.from_json_file(
        f"{model_name_or_path}/config.json",)
    customized_kwargs = dict()
    customized_kwargs["config"] = config

    model = LlavaQwenPcdVidLM.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        # device_map='auto',
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        output_loading_info=False,
        **customized_kwargs
    )

    model.to(device, non_blocking=True)
    print(f'model device: {next(model.parameters()).device}')

    model.eval()

    state_dict = model_zoo.load_url(model.pcd_encoder_config.model_path, progress=True)
    missing_keys, unexpected_keys = model.pcd_encoder.load_state_dict(state_dict['state_dict'], strict=False)
    print(f'model.pcd_encoder device: {next(model.pcd_encoder.parameters()).device}')

    model.to(dtype=torch.bfloat16)
    model.pcd_encoder.to(torch.float32)

    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=None, model_max_length=32768, padding_side="right")
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_1_5"]


    data_args = DataArguments(**data_args)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = False

    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    inference_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  
        shuffle=False,
        collate_fn=data_collator
    )

    print(f'length of inference_dataloader: {len(inference_dataloader)}')
    outputs = []
    with torch.no_grad():
        for i, batch in enumerate(inference_dataloader):
            qa_info = batch["qa_info"][0]
            id = int(qa_info["id"])
            gt_answer = gt_data[id]['conversations'][1]['value']
            print(f'qa_info: {qa_info}')

            batch = move_batch_to_cuda(batch, device=device)
            output = model.generate(
                **batch,
                do_sample=True,
                temperature=0.8,
                use_cache=True,
                num_beams=5,
                max_new_tokens=256)
            print('---------------------------------------------------------------')

            decoded_texts = tokenizer.batch_decode(
                output, 
                skip_special_tokens=True, 
            )[0].strip()
            print('pred: ', decoded_texts)
            print("gt: ", gt_answer)
            print("gt_question: ", gt_data[id]['conversations'][1]['value'])
            
            outputs.append({"query": qa_info['conversations'][0]['value'], 
                            "gt": gt_answer,
                            "pred": decoded_texts})
        
    with open("./inference_result/rea_inference_results_32nquery_32frames_question_only_v20.json", "w") as f:
        json.dump(outputs, f, indent=2)


