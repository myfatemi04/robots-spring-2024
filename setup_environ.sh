#!/bin/sh

# Used for flash attention.
module load cuda/12.2.2

# Don't store models in the home directory.
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/huggingface_cache

