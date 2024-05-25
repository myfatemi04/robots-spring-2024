#!/bin/sh
export PATH=/home/crl/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
export PYTHONPATH=
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/crl/Desktop/alt_miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/crl/Desktop/alt_miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/crl/Desktop/alt_miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/crl/Desktop/alt_miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
