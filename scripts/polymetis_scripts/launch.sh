#!/bin/sh

# Author: Michael Yoo Fatemi (gsk6me)
# This uses a Polymetis configuration from `alt_miniconda3`, which can
# be loaded with the command `source alt_conda.sh`. You can then activate
# the correct Conda environment using `conda activate polymetis_alt`.

# This denotes the proportional gains in joint space.
# Joints are ordered from the base of the robot towards the end.
# robot_client.metadata_cfg.default_Kq=[40,30,50,25,35,25,10]
# robot_client.metadata_cfg.default_Kqd=[4,6,5,5,3,2,1]

# This file (`launch_robot.py`) is part of the Polymetis library.
launch_robot.py \
    robot_client=franka_hardware \
    robot_client.executable_cfg.robot_ip=192.168.1.163 \
    robot_client.executable_cfg.control_port=50051 \
    hz=1000 \
    robot_client.metadata_cfg.default_Kq=[400,400,500,500,400,200,100] \
    robot_client.metadata_cfg.default_Kqd=[40,20,20,20,20,10,5] \
    robot_client.metadata_cfg.default_Kx=[650,650,650,50,50,50] \
    robot_client.metadata_cfg.default_Kxd=[40,40,40,10,10,10]
