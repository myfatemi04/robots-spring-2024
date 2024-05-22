from dataclasses import dataclass

import torch


@dataclass
class RobotConfig:
    reset_joints = torch.tensor([
        0.00010811047832248732,
        -0.39990323781967163,
        -0.00043402286246418953,
        -1.954284906387329,
        0.0010219240793958306,
        1.6010528802871704,
        0.7012835144996643
    ])
    position_limits = torch.tensor([
        [0.025, 0.8],
        [-0.55, 0.55],
        [0.005, 0.6],
    ])
    control_kp = torch.tensor([80, 80, 80, 50.0, 40.0, 50.0]) / 2
    # control_kp = torch.tensor([80, 80, 80, 50.0, 40.0, 50.0])
    control_kv = torch.ones((6,)) * torch.sqrt(control_kp) # * 2.0
