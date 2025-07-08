#!/bin/bash

# Set up the data folder
IMAGE_FOLDER="PATH/TO/Epic_Kitchen"
VIDEO_FOLDER="PATH/TO/Epic_Kitchen"
POINTCLOUD_FOLDER="PATH/TO/Epic_Kitchen"
DATA_YAML="exp_rea_train.yaml" # e.g exp.yaml
EVAL_DATA_YAML="exp_rea_test.yaml"
############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
alias python=python3
############### Show Envs ####################

nvidia-smi

################ Arnold Jobs ################

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="ft-REAv20-1024npoints-32frames-qwen-32nquery"   # ! train v20 data
PREV_STAGE_CHECKPOINT="lmms-lab/LLaVA-Video-7B-Qwen2" # lmms-lab/LLaVA-Video-7B-Qwen2, lmms-lab/llava-onevision-qwen2-7b-si

echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# ! Add these if you want to use lora
# --lora_enable \
# --lora_r 64 \
# --lora_alpha 16 \

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port 43000 \
# export TORCH_CUDA_ARCH_LIST="8.0"
# CUDA_VISIBLE_DEVICES=0 
# ! change evaluation_strategy to "steps" if you want to evaluate
CUDA_LAUNCH_BLOCKING=1 deepspeed --master_port 30001 \
    llava/train/train_mem_3d.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --eval_data_path $EVAL_DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --eval_steps 100 \
    --per_gpu_eval_batch_size 1 \
    --pointcloud_folder $POINTCLOUD_FOLDER \
    --mm_tunable_parts="mm_mlp_adapter,qformer,ray_encoder" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir /work/nvme/bczf/zoezheng126/work_dirs/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 120 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768  \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "eager" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2 
exit 0;
