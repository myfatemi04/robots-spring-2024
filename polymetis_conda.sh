#!/bin/bash
export PS1='\w\$ '
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
# conda activate robo

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# export CUDA_HOME=/home/huron/Repos_2/polymetis_miniconda3/envs/robo


