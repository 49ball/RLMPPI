import time
import numpy as np
import torch
from math import pi, radians


class KinematicBicycleModel:
    def __init__(self, wheelbase: float = 2.5, max_steer: float = 0.6981, delta_time: float = 0.1):
        self.delta_time = torch.tensor(delta_time, device='cuda')
        self.wheelbase = torch.tensor(wheelbase, device='cuda')
        self.max_steer = torch.tensor(max_steer, device='cuda')

    def update(self, states, actions):
        # states와 actions가 1차원 텐서인 경우 2차원 텐서로 변환
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        # states는 (N, 4) 크기의 텐서
        # actions는 (N, 2) 크기의 텐서
        x = states[:, 0]
        y = states[:, 1]
        yaw = states[:, 2]
        velocity = states[:, 3]
        
        acceleration = actions[:, 0]
        steering_angle = actions[:, 1]
        new_velocity = velocity + self.delta_time * acceleration
        new_velocity = torch.clamp(new_velocity, -10, 10)  # new_velocity 클램핑
        angular_velocity = new_velocity * torch.tan(steering_angle) / self.wheelbase
        
        new_x = x + new_velocity * torch.cos(yaw) * self.delta_time
        new_y = y + new_velocity * torch.sin(yaw) * self.delta_time
        new_yaw = yaw + angular_velocity * self.delta_time
        new_yaw = torch.remainder(new_yaw, 2 * pi)
        
        new_states = torch.stack((new_x, new_y, new_yaw, new_velocity), dim=1)
        
        # 입력이 단일 데이터인 경우 1차원 텐서로 반환
        if new_states.shape[0] == 1:
            new_states = new_states.squeeze(0)

        return new_states