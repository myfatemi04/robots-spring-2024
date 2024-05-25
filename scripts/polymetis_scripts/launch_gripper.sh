#!/bin/sh

# Author: Michael Yoo Fatemi (gsk6me)
# This uses a Polymetis configuration from `alt_miniconda3`, which can
# be loaded with the command `source alt_conda.sh`. You can then activate
# the correct Conda environment using `conda activate polymetis_alt`.

# This denotes the proportional gains in joint space.
# Joints are ordered from the base of the robot towards the end.
# robot_client.metadata_cfg.default_Kq=[40,30,50,25,35,25,10]
# robot_client.metadata_cfg.default_Kqd=[4,6,5,5,3,2,1]

# This file (`launch_gripper.py`) is part of the Polymetis library.
launch_gripper.py \
    gripper=franka_hand \
    gripper.executable_cfg.robot_ip=192.168.1.163
