#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --job-name=download
#SBATCH --mem=32G
#SBATCH --account=collabrobogroup
#SBATCH --partition=standard
#SBATCH -e slurm-%j-download.err
#SBATCH -o slurm-%j-download.out

source ~/.bashrc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

mamba activate base

cd /scratch/gsk6me
gsutil -m cp -r gs://gresearch/robotics/droid /scratch/gsk6me/droid-dataset

