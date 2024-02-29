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
# There are 128 cores on this node in total; this means 16 threads per GPU.
# The number of cores is accessible via `lscpu`.
export OMP_NUM_THREADS=16

# Set the image width and height
# image_width=1024
# image_height=576
# train_batch_size=1

# Use the original size of RT-1
# Use a larger batch because of smaller images.
image_width=320
image_height=256
train_batch_size=8

# Train on first 200 samples.
max_train_samples=200

# Using rank=16 reduces to ~6m trainable parameters.
LoRA_rank=16

accelerate launch \
	--main_process_port 29505 \
	train_textimage2video.py \
	--pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt \
	--pretrained_text_encoder_model_name_or_path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
	--rt1_dataset_root /scratch/gsk6me/WORLDMODELS/datasets/rt-1-data-release \
	--input_perturbation 0.1 \
  --train_batch_size $train_batch_size \
  --gradient_accumulation_steps 4 \
	--cache_dir /scratch/gsk6me/huggingface_cache \
	--image_width $image_width \
	--image_height $image_height \
	--gradient_checkpointing \
	--rank $LoRA_rank \
	--report_to wandb \
	--mixed_precision fp16 \
	--max_train_samples $max_train_samples
