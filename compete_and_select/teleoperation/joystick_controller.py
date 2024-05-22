import time

import numpy as np
import pygame


class JoystickController:
    def __init__(self):
        self.joystick_data = {
            'axis_data': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'button': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'hat_data': (0, 0)
        }

        self.pos = np.zeros(3)  # Store x, y, z
        self.rot = np.zeros(3)  # Store roll, pitch, yaw
        self.grasp = np.array([-1])  # Initialize grasp to -1

        self.ROTATION_MODE_STATUS = False
        self.TRANSLATION_MODE_STATUS = True

        pygame.init()
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick(s) found.")
            return
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def update_modes_and_actions(self):

        if self.joystick_data['button'][2] == 1 and self.joystick_data['button'][3] == 0:
            self.ROTATION_MODE_STATUS = True
            self.TRANSLATION_MODE_STATUS = False
            
        elif self.joystick_data['button'][3] == 1 and self.joystick_data['button'][2] == 0:
            self.TRANSLATION_MODE_STATUS = True
            self.ROTATION_MODE_STATUS = False

        if self.TRANSLATION_MODE_STATUS:
            self.pos = np.array([self.joystick_data['axis_data'][4],
                                 self.joystick_data['axis_data'][3],
                                 -self.joystick_data['axis_data'][1]])

        if self.ROTATION_MODE_STATUS:
            self.rot = np.array([-self.joystick_data['axis_data'][3],
                                 self.joystick_data['axis_data'][4],
                                 -self.joystick_data['axis_data'][0]])

        if self.joystick_data['button'][0] == 1 and self.joystick_data['button'][1] == 0:
            self.grasp = np.array([1])
        elif self.joystick_data['button'][0] == 0 and self.joystick_data['button'][1] == 1:
            self.grasp = np.array([-1])
        else:
            self.grasp = np.array([0])

    def get_data(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        axis_data = [self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())]
        axis_data = [round(value, 1) for value in axis_data]
        self.joystick_data['axis_data'] = [0.0 if -0.1 <= value <= 0.1 else value for value in axis_data]

        self.joystick_data['button']  = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]

        self.update_modes_and_actions()

        return self.joystick_data, np.concatenate([self.pos, self.rot, self.grasp])

def test_joystick_controller():
    device = JoystickController()

    while True:
        device_data, robot_action = device.get_data()
        print("device_data:", device_data)
        print("type(device_data):", type(device_data))
        # print("data:", robot_action)

        time.sleep(0.2)
