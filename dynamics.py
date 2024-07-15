import numpy as np
import torch
from math import pi, radians

def normalise_angle_vectorized(angles, max_angle_deg=40):
    max_angle = radians(max_angle_deg)  # 40도를 라디안으로 변환
    min_angle = -max_angle
    
    if isinstance(angles, np.ndarray):
        angles = np.where(angles > max_angle, max_angle, angles)
        angles = np.where(angles < min_angle, min_angle, angles)
    elif isinstance(angles, torch.Tensor):
        angles = torch.where(angles > max_angle, max_angle, angles)
        angles = torch.where(angles < min_angle, min_angle, angles)
    
    return angles

class KinematicBicycleModel:
    def __init__(self, wheelbase: float = 2.5, max_steer: float = 0.6981, delta_time: float = 0.1):
        self.delta_time = delta_time
        self.wheelbase = wheelbase
        self.max_steer = max_steer

    def update(self, states, actions):
        return_tensor = False
        
        # states와 actions가 텐서인 경우 NumPy 배열로 변환
        if isinstance(states, torch.Tensor):
            device = states.device
            states = states.cpu().numpy()
            return_tensor = True
        else:
            device = torch.device('cpu')

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # states와 actions가 1차원 배열인 경우 2차원 배열로 변환
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
        if actions.ndim == 1:
            actions = np.expand_dims(actions, axis=0)
        
        # states는 (N, 4) 크기의 배열
        # actions는 (N, 2) 크기의 배열
        x = states[:, 0]
        y = states[:, 1]
        yaw = states[:, 2]
        velocity = states[:, 3]
        
        acceleration = actions[:, 0]
        steering_angle = actions[:, 1]

        # acceleration = np.clip(acceleration, -2, 2)
        # steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        new_velocity = velocity + self.delta_time * acceleration
        angular_velocity = new_velocity * np.tan(steering_angle) / self.wheelbase
        
        new_x = x + new_velocity * np.cos(yaw) * self.delta_time
        new_y = y + new_velocity * np.sin(yaw) * self.delta_time
        new_yaw = normalise_angle_vectorized(yaw + angular_velocity * self.delta_time)
        
        new_states = np.stack((new_x, new_y, new_yaw, new_velocity), axis=1)
        
        # 입력이 단일 데이터인 경우 1차원 배열로 반환
        if new_states.shape[0] == 1:
            new_states = new_states.squeeze(0)
        
        # 결과를 다시 텐서로 변환
        if return_tensor:
            new_states = torch.tensor(new_states, device=device)
        
        return new_states