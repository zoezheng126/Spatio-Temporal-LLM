# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_POINTCLOUD_TOKEN, POINTCLOUD_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord

from llava.train.aligner import LlavaQwenPcdVidLM, LlavaQwenPcdAlignerConfig

import torch.nn as nn

import open3d as o3d

from llava.train import pc_util

import trimesh

torch.multiprocessing.set_sharing_strategy("file_system")



GROUNDING_META_PROMPT="The <box> tag contains the 3D coordinates of the 8 corners of the object's bounding box in the point cloud, formatted as (x1,y1,z1),(x2,y2,z2),...,(x8,y8,z8), representing its full spatial extent. The <loc> tag contains the 3D coordinates of the center of the bounding box, formatted as (x,y,z), indicating the object's central position in space.\n"
REA_META_PROMPT="The first 64 tokens by user encode learnable queries representing objects and locations in the 3D scene. The following tokens represent egocentric video of recent actions. Use both to reason about spatial references and temporal context when answering."

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# ！config for ScanNet bbox 
nyu40ids = np.array([
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
                32, 33, 34, 35, 36, 37, 38, 39, 40
             ])
nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(nyu40ids))
        }

type2class = {
            'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 
            'curtain':11, 'refrigerator':12, 'shower curtain':13, 'toilet':14, 
            'sink':15, 'bathtub':16, 'others':17
        }
DATASET_METADATA_DIR = "/scratch/bczf/zoezheng126/4dllm/LL3DA/data/scannet/meta_data"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model", "pcd_decoder", "pcd_encoder"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

    pcd_encoder: Optional[str] = field(default="Model_Vote2Cap_DETR")
    pcd_encoder_ckpt: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    pointcloud_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)
    dataset_name: Optional[str] = field(default="REA")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})


# @dataclass
# class EvaluationArguments:
#     eval_num_processes: int = field(default=1)
#     task_names: str = field(default=None)
#     model: str = field(default="llava")
#     model_args: Optional[str] = field(default=None)
#     num_fewshot: Optional[int] = field(default=None)
#     batch_size: int = field(default=1)
#     device: Optional[str] = field(default=None)
#     limit: Optional[int] = field(default=None)
#     check_integrity: Optional[bool] = field(default=False)
#     show_task_to_terminal: Optional[bool] = field(default=False)
#     log_samples: Optional[bool] = field(default=True)
#     gen_kwargs: Optional[str] = field(default="")
#     log_samples_suffix: Optional[str] = field(default="")
#     output_path: Optional[str] = field(default="./logs/")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler", "detector", "decoder_layer", "GenericMLP", "pcd_dim_proj", "img_dim_proj", "rev_dim_proj", "decoder_layer", "decoder", "pcd_ray_encoder", "img_ray_encoder"]
    # ! change back
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler", "pcd_encoder", "mlp_module", "qformer", "img_encoder_to_qformer_projection", "text_embed_to_qformer_projection", "latent_query", "qformer_to_language_projection", "pcd_ray_encoder", "img_ray_encoder"]
    allowed_prefixes = ["model.layers"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        # if any(part in name.split('.') for part in multimodal_keywords):
        #     print(f'263 train.py remove {name}')
        #     continue
        # if isinstance(module, cls):
        #     names = name.split(".")
        #     lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        # if isinstance(module, cls):
        #     lora_module_names.add(name)
        if any(name.startswith(prefix) for prefix in allowed_prefixes) and isinstance(module, cls):
            print(f'270 train.py add {name}')
            lora_module_names.add(name)

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    print(f'275 train.py lora_module_names: {lora_module_names}')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources

def preprocess_3d_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    sources = preprocess_multimodal(sources, data_args)
    for source in sources:
        for sentence in source:
            num_pc = len(re.findall(DEFAULT_POINTCLOUD_TOKEN, sentence["value"]))
            if num_pc == 1 and DEFAULT_POINTCLOUD_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_POINTCLOUD_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_POINTCLOUD_TOKEN, "").strip()
                sentence["value"] = DEFAULT_POINTCLOUD_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_POINTCLOUD_TOKEN, "<pointcloud>" + DEFAULT_POINTCLOUD_TOKEN + "</pointcloud>")
            replace_token_pc = DEFAULT_POINTCLOUD_TOKEN
            if data_args.mm_use_im_start_end:  # TODO New config flag for point clouds
                replace_token_pc = DEFAULT_IM_START_TOKEN + replace_token_pc + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_POINTCLOUD_TOKEN, replace_token_pc)

            # Clean noisy data
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")
    return sources

def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.", dataset_name: str = "REA") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        special_tokens = ["<image>", "<pointcloud>"]

        # Check which tokens are missing
        missing_tokens = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]

        if missing_tokens:
            print(f"Adding missing special tokens: {missing_tokens}")
            tokenizer.add_tokens(missing_tokens, special_tokens=True)

        # tokenizer.add_tokens(["<image>", "<pointcloud>", "<bbox>", "</bbox>", "<loc>", "</loc>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    pointcloud_token_index = tokenizer.convert_tokens_to_ids("<pointcloud>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[-2:]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    user_input_spans = []
    updated_system_message = system_message
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        # ! add meta prompt for grouding task
        if "<box>" in source[0]["value"] or "<loc>" in source[0]["value"]:
            updated_system_message += GROUNDING_META_PROMPT
        elif dataset_name == "REA":
            updated_system_message += REA_META_PROMPT
        else: 
            updated_system_message = system_message

        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : updated_system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            start_idx = len(input_id)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
            end_idx = len(input_id)
            if role == "user":
                user_input_spans.append((start_idx, end_idx))



                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == pointcloud_token_index:
                input_id[idx] = POINTCLOUD_TOKEN_INDEX
            
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    # ! get qformer text input (question only)
    user_input_ids = torch.tensor([input_id[start:end] for start, end in user_input_spans], dtype=torch.long)
    user_labels = torch.tensor([target[start:end] for start, end in user_input_spans], dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
        qformer_input_ids=user_input_ids,
        qformer_labels=user_labels,  # tensor(bs x seq_len)
    ), tokenizer


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, dataset_name: str = "REA") -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image, dataset_name=dataset_name)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


#### ScanNet bbox preprocessing ####
def flip_axis_to_camera_np(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = pc.copy()
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2

def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output

def get_3d_box_batch_np(box_size, angle, center):
    input_shape = angle.shape
    R = roty_batch(angle)
    l = np.expand_dims(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = np.expand_dims(box_size[..., 1], -1)
    h = np.expand_dims(box_size[..., 2], -1)
    corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
    corners_3d[..., :, 0] = np.concatenate(
        (l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1
    )
    corners_3d[..., :, 1] = np.concatenate(
        (h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1
    )
    corners_3d[..., :, 2] = np.concatenate(
        (w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1
    )
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d

def box_parametrization_to_corners_np(box_center_unnorm, box_size, box_angle):
        # box_center_upright = flip_axis_to_camera_np(box_center_unnorm) # ! DEBUG
        # boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_unnorm)
        return boxes


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        self.SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
        self.ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                            np.pi))
        self.TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
        self.ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

        self.ROTATION_AXIS = 'z'
        self.LOCFEAT_IDX = 2
        self.tokenizer = tokenizer
        self.list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        ##### new area for OpenScene #####
        from llava.model.openscene.dataset.voxelizer import Voxelizer
        self.voxel_size = 0.06
        # if data_args.dataset_name == "REA":
        #     self.voxel_size = 0.12
        self.aug = False  # ! not sure the camera poses also handled in the augmentation
        self.eval_all = True
        self.input_color = True
        self.voxelizer = Voxelizer(
            voxel_size=self.voxel_size,
            clip_bound=None,
            use_augmentation=False,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)
        
        if self.aug:
            import llava.model.openscene.dataset.augmentation as t
            data_aug_color_trans_ratio=0.1
            data_aug_color_jitter_std=0.05
            data_aug_hue_max=0.5
            data_aug_saturation_max=0.2
            prevoxel_transform_train = [
                t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
            self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
            input_transforms = [
                t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(
                    data_aug_hue_max, data_aug_saturation_max),
            ]
            self.input_transforms = t.Compose(input_transforms)
        
        ##### scannet bbox config #####
        self.type2class = {
            'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 
            'curtain':11, 'refrigerator':12, 'shower curtain':13, 'toilet':14, 
            'sink':15, 'bathtub':16, 'others':17
        }
        self.nyu40ids = np.array([
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
                32, 33, 34, 35, 36, 37, 38, 39, 40
             ])
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }
        self.nyu40id2class = self._get_nyu40id2class()

    def _get_nyu40id2class(self):
        lines = [line.rstrip() for line in open(os.path.join(DATASET_METADATA_DIR, 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        nyu40ids2class = {}
        for i in range(len(lines)):
            label_classes_set = set(type2class.keys())
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in nyu40ids:
                if nyu40_name not in label_classes_set:
                    nyu40ids2class[nyu40_id] = type2class["others"]
                else:
                    nyu40ids2class[nyu40_id] = type2class[nyu40_name]
        return nyu40ids2class
    
    def get_scan_data_bbox_only(self, prefix, num_points=2048, rigid_tranformations=None):
        center_normalizing_range = [
                np.zeros((1, 3), dtype=np.float32),
                np.ones((1, 3), dtype=np.float32),
            ]
        # --- Load bbox and instance metadata ---
        mesh_vertices = np.load(prefix + "_vert.npy") 
        # instance_labels = np.load(prefix + "_ins_label.npy")
        # semantic_labels = np.load(prefix + "_sem_label.npy")
        instance_bboxes = np.load(prefix + "_bbox.npy")

        point_cloud = mesh_vertices[:, 0:3]  # Only use XYZ for bbox alignment
        pcl_color = mesh_vertices[:, 3:6]    # Optional: for referencing clicked object

        MAX_NUM_OBJ = 128
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        object_ids = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)

        # Downsample point cloud for faster bbox indexing
        point_cloud, choices = pc_util.random_sampling(point_cloud, num_points, return_choices=True)
        # instance_labels = instance_labels[choices]
        # semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        # Fill bbox-related fields
        target_bboxes_mask[:instance_bboxes.shape[0]] = 1
        target_bboxes[:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        raw_sizes = target_bboxes[:, 3:6]
        box_centers = target_bboxes[:, 0:3]

        # Normalize bbox centers to 3D grid  # ! not needed at the moment
        # point_cloud_dims_min = point_cloud[..., :3].min(axis=0)
        # point_cloud_dims_max = point_cloud[..., :3].max(axis=0)
        # box_centers_normalized = pc_util.shift_scale_points(
        #     box_centers[None, ...],
        #     src_range=[point_cloud_dims_min[None, ...], point_cloud_dims_max[None, ...]],
        #     dst_range=center_normalizing_range,
        # ).squeeze(0)
        # box_centers_normalized *= target_bboxes_mask[..., None]

        # mult_factor = point_cloud_dims_max - point_cloud_dims_min
        # box_sizes_normalized = pc_util.scale_points(
        #     raw_sizes[None, ...], mult_factor=1.0 / mult_factor[None, ...]
        # ).squeeze(0)

        # Compute corners from center/size/angle
        box_corners = box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes[None, ...],
            raw_angles[None, ...]
        ).squeeze(0)
        
        if rigid_tranformations is not None:
            box_corners_homo = np.concatenate([box_corners, np.ones((box_corners.shape[0], 8, 1))], axis=-1)
            box_corners = (rigid_tranformations @ box_corners_homo.transpose(0, 2, 1)).transpose(0, 2, 1)[..., :3]
            box_centers_homo = np.concatenate([box_centers, np.ones((box_centers.shape[0], 1))], axis=1)  # (N, 4)
            box_centers = (rigid_tranformations @ box_centers_homo.T).T[:, :3] 
            raw_sizes = box_corners.max(1) - box_corners.min(1)

        object_ids[:instance_bboxes.shape[0]] = instance_bboxes[:, -1]
        # Semantic class labels for boxes # ! not needed at the moment
        # box_semcls = np.zeros((MAX_NUM_OBJ,))
        # box_semcls[:instance_bboxes.shape[0]] = [
        #     self.nyu40id2class.get(x, -1)
        #     for x in instance_bboxes[:, -2][:instance_bboxes.shape[0]]
        # ]
        # --- Return only bbox-related data ---
        return {
            "gt_box_corners": box_corners.astype(np.float32),              # (MAX_NUM_OBJ, 8, 3)
            "gt_box_centers": box_centers.astype(np.float32),              # (MAX_NUM_OBJ, 3)
            # "gt_box_centers_normalized": box_centers_normalized.astype(np.float32),
            # "gt_box_sizes": raw_sizes.astype(np.float32),                  # (MAX_NUM_OBJ, 3)
            # "gt_box_sizes_normalized": box_sizes_normalized.astype(np.float32),
            "gt_box_present": target_bboxes_mask.astype(np.float32),       # (MAX_NUM_OBJ,)
            # "gt_box_sem_cls_label": box_semcls.astype(np.int64),           # (MAX_NUM_OBJ,)
            "gt_box_angles": raw_angles.astype(np.float32),
            "gt_object_ids": object_ids.astype(np.int64),
            # "pcl_color": pcl_color,                                        # optional, for click-based interaction
            # "point_clouds": point_cloud.astype(np.float32),                # to match with clicked or masked points
            # "point_cloud_dims_min": point_cloud_dims_min.astype(np.float32),
            # "point_cloud_dims_max": point_cloud_dims_max.astype(np.float32),
            # "instance_labels": instance_labels,
            # "semantic_labels": semantic_labels,
        }

 
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"
    
    def process_pointcloud(self, sources, pointcloud_folder, normalize=False, augment=False, frame_idx=None, flip_yz=False):
        """
        Returns:
            torch.Tensor: A tensor of shape (N, 3) containing the point cloud data.
        """
        pointcloud_file = sources[0]["pointcloud"]
        path = os.path.join(pointcloud_folder, pointcloud_file)
        if path.endswith("ply"):
            ply_path = path
            pcd = o3d.io.read_point_cloud(ply_path)
            points = torch.tensor(pcd.points) 

            if normalize:
                centroid = points.mean(dim=0) 
                points -= centroid 
                
                max_dist = torch.linalg.norm(points, dim=1).max()  # Find max distance
                points /= max_dist  # Scale to unit sphere

            if pcd.has_colors():
                colors = np.asarray(pcd.colors, dtype=np.float32)
                colors_tensor = torch.tensor(colors)
                points = torch.cat([points, colors_tensor], dim=1)

            # Load normals (Nx3)
            if pcd.has_normals():
                normals = np.asarray(pcd.normals, dtype=np.float32)
                normals_tensor = torch.tensor(normals)
                points = torch.cat([points, normals_tensor], dim=1)

            floor_height = torch.quantile(points[:, 2], 0.99)  # Compute 99th percentile
            height = points[:, 2] - floor_height  # Compute height difference
            points = torch.cat([points, height.unsqueeze(1)], dim=1)
        elif path.endswith('npy'):
            if True: 
                prefix = path.replace('_vert.npy', '')
                locs_in = np.load(prefix + '_vert.npy')[:, :3]
                feats_in = np.load(prefix + '_vert.npy')[:, 3:6] if self.input_color else np.zeros_like(locs_in)

                locs = locs_in
                locs, feats, labels, inds_reconstruct, rigid_transformation, min_coords = self.voxelizer.voxelize(locs, feats_in)

                if self.aug:
                    locs, feats, labels = self.input_transforms(locs, feats, labels)

                coords = torch.from_numpy(locs).int()
                feats = torch.from_numpy(feats).to(torch.bfloat16) / 127.5 - 1. if self.input_color else torch.ones(coords.shape[0], 3)
                if 'c2ws' in sources[0] and 'K' in sources[0] and frame_idx != None:
                    c2ws_path = os.path.join(pointcloud_folder, sources[0]["c2ws"])
                    Ks_path = os.path.join(pointcloud_folder, sources[0]["K"])
                    K = torch.from_numpy(np.loadtxt(Ks_path))

                    c2ws = []
                    for idx in frame_idx:
                        c2w_file = os.path.join(c2ws_path, f'{idx}.txt')
                        c2w = np.loadtxt(c2w_file)  # shape (4, 4)
                        if np.isnan(c2w).any() or np.isinf(c2w).any():
                            print(f'1498 train.py c2w = {c2w}')

                        if flip_yz:  # Apply OpenCV correction
                            flip_yz_ = np.diag([1, -1, -1, 1])
                            c2w = c2w @ flip_yz_
                        # Apply inverse rigid transformation
                        # c2w_aug = rigid_transformation @ c2w  # make camera consistent with augmented PCD
                        c2w[:3, 3] = c2w[:3, 3] * rigid_transformation[0][0]
                        c2w[:, -1][:3] += rigid_transformation[:, -1][:3]
                        c2ws.append(torch.from_numpy(c2w).to(torch.bfloat16))
                        

                    c2ws = torch.stack(c2ws)
                    if self.eval_all: # always assume to be True in current case
                        return coords, feats, None, torch.from_numpy(inds_reconstruct).long(), c2ws, K, rigid_transformation
                    return coords, feats, labels, c2ws, K, rigid_transformation
                    
        elif path.endswith('glb'):

            mesh_path = os.path.join(pointcloud_folder, sources[0]["pointcloud"])
            mesh = trimesh.load(mesh_path, process=False)
            locs_in = mesh.geometry["geometry_0"].vertices
            locs_in = np.array(locs_in)
            if self.input_color:
                feats_in = mesh.geometry["geometry_0"].visual.vertex_colors[:, :3]
                feats_in = np.array(feats_in)
            else:
                feats_in = np.zeros_like(locs_in)
            
            if locs_in.shape[0] > 1000000:
                locs_in = locs_in[::10]
                feats_in = feats_in[::10]

            locs, feats, labels, inds_reconstruct, rigid_transformation, min_coords = self.voxelizer.voxelize(locs_in, feats_in)

            if self.aug:
                locs, feats, labels = self.input_transforms(locs, feats, labels)

            coords = torch.from_numpy(locs).int()
            feats = torch.from_numpy(feats).to(torch.bfloat16) / 127.5 - 1. if self.input_color else torch.ones(coords.shape[0], 3)

            if 'c2ws' in sources[0] and 'K' in sources[0] and frame_idx != None:
                c2ws_path = os.path.join(pointcloud_folder, sources[0]["c2ws"])
                Ks_path = os.path.join(pointcloud_folder, sources[0]["K"])
                K = torch.from_numpy(np.loadtxt(Ks_path))

                c2ws = []
                for idx in frame_idx:
                    c2w_file = os.path.join(c2ws_path, f'{idx}.txt')
                    c2w = np.loadtxt(c2w_file)  # shape (4, 4)
                    if np.isnan(c2w).any() or np.isinf(c2w).any():
                        print(f'1498 train.py c2w = {c2w}')

                    if flip_yz:  # Apply OpenCV correction
                        flip_yz_ = np.diag([1, -1, -1, 1])
                        c2w = c2w @ flip_yz_
                    c2w[:3, 3] = c2w[:3, 3] * rigid_transformation[0][0]
                    c2w[:, -1][:3] += rigid_transformation[:, -1][:3]
                    c2ws.append(torch.from_numpy(c2w).to(torch.bfloat16))
                    
                c2ws = torch.stack(c2ws)
                if self.eval_all:
                    return coords, feats, None, torch.from_numpy(inds_reconstruct).long(), c2ws, K, rigid_transformation
                return coords, feats, labels, c2ws, K, rigid_transformation
        return coords, feats, None
           

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:

        # TODO need to skip the sample with pcd num_points < 2048
        sources = self.list_data_dict[i]
        dataset_name = sources[0]["metadata"]['dataset']
        
        # ! inference prompt start
        # self.list_data_dict[i]["conversations"][0]["value"] += "Give Explanation and reasoning for your answer. Answer in detail, and be specific. Do not random guess. If you don't know say 'I don't know'."
        # ! inference prompt end

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad 
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif "video" in sources[0]:

            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)

            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            # try:
            if "shareVideoGPTV" in video_file:
                frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                if self.data_args.force_sample:
                    num_frames_to_sample = self.data_args.frames_upbound
                else:
                    num_frames_to_sample = 10

                avg_fps = 2
                
                total_frames = len(frame_files)
                sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


                frame_time = [i/2 for i in sampled_indices]
                frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                video_time = total_frames / avg_fps

                # Read and store the sampled frames
                video = []
                for idx in sampled_indices:
                    frame_path = frame_files[idx]
                    try:
                        with Image.open(frame_path) as img:
                            frame = img.convert("RGB")
                            video.append(frame)
                    except IOError:
                        print(f"Failed to read frame at path: {frame_path}")
            else: # video frame dir also enter here for sampling purpose
                self.list_data_dict[i]["pointcloud_folder"] = self.data_args.pointcloud_folder
                video, video_time, frame_time, num_frames_to_sample, frame_idx = process_video_with_decord(video_file, self.data_args, sources[0])
            processor = self.data_args.image_processor
            image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
            if self.data_args.add_time_instruction:
                time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
            image = [(image, video.shape, "video")]

            sources = preprocess_3d_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif "pointcloud" not in sources[0]:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        if "pointcloud" in self.list_data_dict[i]:
            rigid_tranformations = None
            pointcloud_folder = self.data_args.pointcloud_folder
            pointcloud_tensor, color, _, _, c2ws, K, rigid_tranformations = self.process_pointcloud([self.list_data_dict[i]], pointcloud_folder, normalize=False, augment=False, frame_idx=frame_idx)

            pointcloud = [
                (pointcloud_tensor, color, "pointcloud", c2ws, K, True)
            ]

            pcd_path = os.path.join(pointcloud_folder, self.list_data_dict[i]["pointcloud"])
            prefix = pcd_path.replace("_vert.npy", "")
            if os.path.exists(prefix + "_bbox.npy") and "<location>" in sources[0][0]["value"]:
                bbox_data_dict = self.get_scan_data_bbox_only(prefix, rigid_tranformations=rigid_tranformations)
                target_object_id = int(self.list_data_dict[i]["object_id"])  # id from scanrefer annotation file
                # target_object_name = self.list_data_dict[i]["object_name"]
                match_mask = (bbox_data_dict["gt_object_ids"] == target_object_id)
              
                match_bbox_indices = np.where(match_mask)[0]
                target_bbox_corner = torch.from_numpy(bbox_data_dict["gt_box_corners"][match_bbox_indices[0]]).to(torch.bfloat16)  # shape (8, 3)
                target_bbox_center = torch.from_numpy(bbox_data_dict["gt_box_centers"][match_bbox_indices[0]]).to(torch.bfloat16)  # shape (3,)

                target_bbox_corner_str = ','.join(
                    f"({x:.2f},{y:.2f},{z:.2f})" for x, y, z in target_bbox_corner.reshape(-1, 3).tolist()
                )   # ! save tokenizer
                target_bbox_center_str = ','.join(
                    f"({x:.2f},{y:.2f},{z:.2f})" for x, y, z in target_bbox_center.reshape(-1, 3).tolist()
                )   # ! save tokenizer
                # target_bbox_rays_str = ' '.join([f"{x.item():.2f}" for x in target_bbox_rays[0].reshape(-1)])
                # location = f"<bbox>{target_bbox_corner_str}</bbox><loc>{target_bbox_center_str}</loc>"
                location = f"<box>{target_bbox_corner_str}</box><loc>{target_bbox_center_str}</loc>"
                # sources[0][0]["value"] = sources[0][0]["value"].replace("<location>", location)   
                sources[0][0]["value"] = sources[0][0]["value"].replace("<location>", location)   # ! save tokenizer
            
        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i]) or ("pointcloud" in self.list_data_dict[i])
        data_dict, self.tokenizer = preprocess(sources, self.tokenizer, has_image=has_image, dataset_name=dataset_name)
        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], 
                             labels=data_dict["labels"][0],
                             qformer_input_ids=data_dict["qformer_input_ids"][0],
                             qformer_labels=data_dict["qformer_labels"][0],)

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        if "pointcloud" in self.list_data_dict[i]:
            data_dict["pointcloud"] = pointcloud
            
        else:
            dummy_c2ws = torch.eye(4).repeat(self.data_args.frames_upbound, 1, 1)
            dummy_K = torch.tensor([[500.0, 0, 192], [0, 500.0, 192], [0, 0, 1]])
            data_dict["pointcloud"] = [
                (torch.zeros(1, 2048, 3), torch.zeros(1, 2048, 3), torch.zeros(1, 2048, 1), "text", dummy_c2ws, dummy_K, False), # TODO: 10 is hardcoded for adding color, normal, and height
            ]

        data_dict["id"] = self.list_data_dict[i].get("id", i)
        data_dict["qa_info"] = self.list_data_dict[i]
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    detector: torch.nn.Module = None

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, qformer_input_ids, qformer_labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "qformer_input_ids", "qformer_labels")
        )

        input_ids = [_[: self.tokenizer.model_max_length] for _ in input_ids]
        labels = [_[: self.tokenizer.model_max_length] for _ in labels]
        qformer_input_ids = [_[: self.tokenizer.model_max_length] for _ in qformer_input_ids]
        qformer_labels = [_[: self.tokenizer.model_max_length] for _ in qformer_labels]

        # Ensure pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0  # LLaMA3 fallback

        # Pad sequences
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        qformer_input_ids = self.pad_sequence(qformer_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        qformer_labels = self.pad_sequence(qformer_labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Return batch
        batch = dict(
            input_ids=input_ids,
            labels=labels.long() if labels.dtype == torch.int32 else labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            qformer_input_ids=qformer_input_ids,
            qformer_labels=qformer_labels.long() if qformer_labels.dtype == torch.int32 else qformer_labels,
            qformer_attention_mask=qformer_input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]
            batch["images"] = images

        if "pointcloud" in instances[0]:
            pointclouds = [instance["pointcloud"] for instance in instances]

            batch["pointclouds"] = [pcd[0] for pcd_list in pointclouds for pcd in pcd_list]
            batch["pointclouds_colors"] = [pcd[1] for pcd_list in pointclouds for pcd in pcd_list]
            # batch["pointclouds_labels"] = [pcd[2] for pcd_list in pointclouds for pcd in pcd_list]
            batch["pointcloud_modalities"] = [pcd[2] for pcd_list in pointclouds for pcd in pcd_list]
            batch["c2ws"] = [pcd[3] for pcd_list in pointclouds for pcd in pcd_list]
            batch["K"] = [pcd[4] for pcd_list in pointclouds for pcd in pcd_list]
            batch["pointclouds_attn_mask"] = [pcd[5] for pcd_list in pointclouds for pcd in pcd_list]

            # if "bboxes" in instances[0]:
            #     bboxes = [instance["bboxes"] for instance in instances]
            #     batch["target_bbox_corner"] = [bbox["target_bbox_corner"] for bbox_list in bboxes for bbox in bbox_list]
            #     batch["target_bbox_center"] = [bbox["target_bbox_center"] for bbox_list in bboxes for bbox in bbox_list]
            #     batch["target_object_id"] = [bbox["target_object_id"] for bbox_list in bboxes for bbox in bbox_list]
            #     batch["target_object_name"] = [bbox["target_object_name"] for bbox_list in bboxes for bbox in bbox_list]

            batch["pointclouds_feature"], batch["pointclouds_xyz"] = [], []
            if self.detector is not None:  #! currently not used
                for pcd_list in pointclouds:
                    for pcd in pcd_list:
                        device = next(self.detector.parameters()).device 
                        points = pcd[0].to(device)
                        pcd_xyz, pcd_feature, _ = self.detector.run_encoder(points)
                        pcd_xyz = pcd_xyz.to('cpu')
                        pcd_feature = pcd_feature.to('cpu')
                        points = points.to('cpu')
                        batch["pointclouds_feature"].append(pcd_feature)
                        batch["pointclouds_xyz"].append(pcd_xyz)
            else:
                batch["pointclouds_feature"] = [[] for _ in range(len(batch["pointclouds"]))]
                batch["pointclouds_xyz"] = [[] for _ in range(len(batch["pointclouds"]))]
            # print(f'collate_fn device: {pointclouds[0][0].device}')
        if "qa_info" in instances[0]:
            batch['qa_info'] = [instance["qa_info"] for instance in instances]

            
        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, detector_model=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, detector=detector_model)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path) 

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                pcd_llava_config = LlavaQwenConfig.from_pretrained("lmms-lab/LLaVA-Video-7B-Qwen2") 
                pcd_llava_config.model_type = "pcd_aligner"
                pcd_llava_config.architectures = ["LlavaQwenPcdVidLM"]
                pcd_llava_config = LlavaQwenPcdAlignerConfig.from_dict(pcd_llava_config.to_dict())
                pcd_llava_customized_kwargs = dict()
                pcd_llava_customized_kwargs["config"] = pcd_llava_config
                model = LlavaQwenPcdVidLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    output_loading_info=False,
                    **pcd_llava_customized_kwargs,   # ! this should be added to prevent config auto load to the grandparent model
                )

# ! hardcode load weights start for peft model
                # def remove_first_prefix(name, prefix):
                #     if name.startswith(prefix):
                #         return name[len(prefix):]
                #     return name

                # non_lora_weights = torch.load(f"/scratch/bczf/zoezheng126/4dllm/LLaVA-NeXT/scripts/video/train/work_dirs/isolate_second_stage_config/non_lora_trainables.bin", map_location='cpu')
                # new_state_dict = {}
                # for k, v in non_lora_weights.items():
                #     if "mlp_module" not in k:
                #         new_k = remove_first_prefix(k, "base_model.model.base_model.model.")
                #         new_state_dict[new_k] = v
                # result = model.load_state_dict(new_state_dict, strict=False)
# ! hardcode load weights end
                
        elif "gemma" in model_args.model_name_or_path.lower():
            model = LlavaGemmaForCausalLM.from_pretrained(
                # model_args.model_name_or_path,
                pretrained_model_name_or_path=None,
                state_dict={},
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        # if training_args.bits == 16:  # ! change back
        #     if training_args.bf16:
        #         model.to(torch.bfloat16)
        #     if training_args.fp16:
        #         model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")

        # ! load non lora weights from peft model
        if os.path.exists(os.path.join(model_args.model_name_or_path, "adapter_config.json")):  
            from peft import PeftModelForCausalLM
            peft_model_path = model_args.model_name_or_path
            # model = PeftModelForCausalLM.from_pretrained(model, peft_model_path)
            model = PeftModelForCausalLM.from_pretrained(model, model_id=model_args.model_name_or_path) # ! notice: this will not load non_lora_weight
            if os.path.exists(f"{model_args.model_name_or_path}/non_lora_trainables.bin"):
                non_lora_weights = torch.load(f"{model_args.model_name_or_path}/non_lora_trainables.bin", map_location='cpu')
                
            # ! second stage training weight loading model.base_model.model.base_model...
                def remove_first_prefix(name, prefix):
                    if name.startswith(prefix):
                        return name[len(prefix):]
                    return name
                new_state_dict = {}
                for k, v in non_lora_weights.items():
                    new_k = remove_first_prefix(k, "base_model.model.")
                    new_state_dict[new_k] = v
                model.load_state_dict(new_state_dict, strict=False)
            # ! end 
                # model.load_state_dict(non_lora_weights, strict=False)
            # model.merge_adapter()
        else:  
            model = get_peft_model(model, lora_config)

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
        token_id = tokenizer.vocab_size - 1
        token_str = tokenizer.convert_ids_to_tokens(token_id)
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.add_time_instruction = data_args.add_time_instruction
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride 

        ### Deciding train which part of the model
        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)

        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                        param.requires_grad_(True)

        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if model_args.pcd_encoder is not None:
        from torch.utils import model_zoo
        state_dict = model_zoo.load_url(model.pcd_encoder_config.model_path, progress=True)

        new_state_dict = state_dict
        missing_keys, unexpected_keys = model.pcd_encoder.load_state_dict(new_state_dict['state_dict'], strict=False)
        
        # ! manually convert pcd encoder to float32
        for name, param in model.pcd_encoder.named_parameters():
            param.data = param.data.to(torch.float32)

        if training_args.lora_enable:
            from peft.tuners.lora import LoraLayer

            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)

        tunable_parts = model_args.mm_tunable_parts.split(",")
        modules_to_parts = [
            (model.pcd_encoder, "pcd_encoder"),
            (model.mlp_module, "qformer"), 
            (model.qformer, "qformer"),
            (model.img_encoder_to_qformer_projection, "qformer"),
            (model.text_embed_to_qformer_projection, "qformer"),
            (model.latent_query, "qformer"),
            (model.qformer_to_language_projection, "qformer"),
            (model.pcd_ray_encoder, "ray_encoder"),
            (model.img_ray_encoder, "ray_encoder"),
        ]

        for module, part_name in modules_to_parts:
            if module is None:
                continue
            requires_grad = part_name in tunable_parts
            for name, param in module.named_parameters():
                param.requires_grad_(requires_grad)

        def register_nan_hooks(model):
            real_model = model.module if hasattr(model, "module") else model

            def forward_hook(name, module):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        x = input[0]
                        if isinstance(x, torch.Tensor):
                            print(f"✅ FORWARD {name}: input: has_nan:{torch.isnan(x).any()}, has_inf: {torch.isinf(x).any()}, "
                                f"output: mean={output.mean().item():.4e}, std={output.std().item():.4e}, max={output.max().item():.4e}")
                        else:
                            print(f"✅ FORWARD {name}: output: mean={output.mean().item():.4e}, std={output.std().item():.4e}, max={output.max().item():.4e}")

                        if torch.isnan(output).any() or torch.isinf(output).any():
                            print(f"🚨 FORWARD NaN/Inf in {name}")

                        # Print weight stats
                        for pname, param in module.named_parameters(recurse=False):
                            print(f"   🔍 weights {name}.{pname}: min={param.data.min().item():.4e}, max={param.data.max().item():.4e}")
                return hook

            def backward_hook(name):
                def hook(grad):
                    print(f"✅ BACKWARD {name}: grad norm={grad.norm().item():.4e}, has_nan={torch.isnan(grad).any()}, has_inf={torch.isinf(grad).any()}")
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print(f"🚨 BACKWARD NaN/Inf in {name}")
                return hook

            for name, module in real_model.named_modules():
                try:
                    module.register_forward_hook(forward_hook(name, module))
                except Exception as e:
                    print(f"⚠️ Could not register forward hook on {name}: {e}")

                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        try:
                            param.register_hook(backward_hook(f"{name}.{param_name}"))
                        except Exception as e:
                            print(f"⚠️ Could not register backward hook on {name}.{param_name}: {e}")
        
        # register_nan_hooks(model)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, detector_model=None)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train() 
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    train()
