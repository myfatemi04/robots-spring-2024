#!/bin/sh

# Before running, make sure to call `accelerate config` (only needs to be run once).

# Make sure CUDA is loaded. This will set the CUDA_HOME variable.
module load cuda/12.2.2

# --main_process_port 0 automatically selects a port for the main process. However, I found that
#   this can cause the process to hang, so I just replaced with a reasonable alternative port.
# --gradient_checkpointing is a way to perform backpropagation in a more memory efficient way.
# --rank specifies the rank for LoRA.

export HUGGINGFACE_HUB_CACHE=/scratch/$USER/huggingface_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/home/gsk6me/miniconda3/envs/py310/lib:$LD_LIBRARY_PATH
# https://github.com/wandb/wandb/issues/5214
export WANDB__SERVICE_WAIT=300

accelerate launch \
	--main_process_port 29505 \
	train_textimage2video.py \
	--pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
	--pretrained_text_encoder_model_name_or_path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
	--rt1_dataset_root /scratch/gsk6me/WORLDMODELS/datasets/rt-1-data-release \
	--input_perturbation 0.1 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
	--cache_dir /scratch/gsk6me/huggingface_cache \
	--image_width 1024 \
	--image_height 576 \
	--gradient_checkpointing \
	--rank 16 \
	--report_to wandb \
	--mixed_prediction fp16
