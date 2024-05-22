#!/bin/bash
# if we are in an environment
conda deactivate
# deactivate the base environment as well
conda deactivate

export PS1="\[\]\[\]\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ \[\]\[\]"
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/huron/Repos_2/polymetis_miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/huron/Repos_2/polymetis_miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/huron/Repos_2/polymetis_miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/huron/Repos_2/polymetis_miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/home/huron/Repos_2/polymetis_miniconda3/etc/profile.d/mamba.sh" ]; then
    . "/home/huron/Repos_2/polymetis_miniconda3/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

# Activate the Polymetis environment.
conda activate polymetis_again

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export CUDA_HOME=/home/huron/Repos_2/polymetis_miniconda3/envs/robo
