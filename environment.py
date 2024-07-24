import time
import numpy as np
import random
import torch
from math import pi
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dynamics import KinematicBicycleModel


class MapWithObstacles:
    def __init__(self, map_size=100, obstacle_count=10, obstacle_radius=3, seed=None):
        self.map_size = map_size
        self.obstacle_count = obstacle_count
        self.obstacle_radius = obstacle_radius
        self.seed = seed
        self.obstacles = []
        if seed is not None:
            np.random.seed(seed)  # 재현 가능한 결과를 위해 시드 설정
        else:
            np.random.seed()
        self.generate_obstacles()

    def generate_obstacles(self):
        # 지정된 위치에 장애물 생성
        self.obstacles = []

    def plot_map(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)

        for (x, y) in self.obstacles:
            circle = Circle((x, y), self.obstacle_radius, fc='gray', edgecolor='black')
            ax.add_patch(circle)

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(f'{self.map_size}x{self.map_size} Map with Circular Obstacles')
        plt.grid(True)
        plt.show()

    def calculate_reward(self, states, prev_states):
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        if len(prev_states.shape)==1:
            prev_states = prev_states.reshape(1,-1)
        
        prev_states = torch.tensor(prev_states,dtype=torch.float32,device='cuda')
        states = torch.tensor(states, dtype=torch.float32, device='cuda')
        prev_x, prev_y, prev_yaw, prev_v = prev_states[:, 0], prev_states[:, 1], prev_states[:, 2], prev_states[:, 3]
        x, y, yaw, v = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        goal = torch.tensor([100, 100], dtype=torch.float32, device='cuda')
        
        done = False
        
        max_reward = 10.0
        max_velocity = 10.0
        min_velocity = 1.0

        # destination reward
        prev_distance_to_goal = torch.sqrt((prev_x-goal[0])**2+(prev_y-goal[1])**2)
        distance_to_goal = torch.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        reward = prev_distance_to_goal - distance_to_goal

        # velocity reward
        speed_reward = 1 - v / max_velocity
        
        reward += speed_reward
        
        # heading reward
        heading_reward = 1 - yaw / (pi/4)
        
        reward += heading_reward
        
        if distance_to_goal <= 0.2:
            reward = torch.tensor(max_reward, dtype=torch.float32, device='cuda')
            done = True

        elif torch.any(x < 0) or torch.any(x > 100) or torch.any(y < 0) or torch.any(y > 100): #차가 경계선 밖을 나갈때
            reward = torch.tensor(-max_reward, dtype=torch.float32, device='cuda')
            done = True

        return reward.cpu().numpy().item(), done


class Env(object):
    def __init__(self, x, y, yaw, velocity, action_repeat):
        self.state = torch.tensor([x, y, yaw, velocity], dtype=torch.float32, device='cuda')
        self.shape = self.state.shape  # 관찰 공간의 형태 (x, y, yaw, velocity)
        self.action_space = (2,)  # 액션 공간의 형태 (acceleration, steering_angle)
        self.map = MapWithObstacles()
        self.ep_len = 1000 // action_repeat
        self.model = KinematicBicycleModel()
        self.t = 0

    def reset(self):
        # 환경을 초기화하는 메서드
        self.state = torch.tensor([0.0, 0.0, random.uniform(0, pi / 2), 0.0], device='cuda')
        self.t = 0
        return self.observe()

    def step(self, action):
        # action을 기반으로 환경 상태를 업데이트하는 메서드
        prev_state=self.state.clone()
        self.state[0], self.state[1], self.state[2], self.state[3] = self.model.update(self.state, action)
        reward,done =self.map.calculate_reward(self.state,prev_state)
        # done 상태와 reward, 그리고 관찰(observation) 값 리턴
        self.t += 1
        return self.observe(), reward, done or self.t == self.ep_len, {}

    def observe(self):
        # 현재 환경 상태를 반환하는 메서드
        return self.state

def make_env(cfg):
    # 환경 초기화
    initial_x = random.uniform(-10, 10)
    initial_y = random.uniform(-10, 10)
    initial_yaw = random.uniform(-3.14, 3.14)  # -π to π
    initial_velocity = random.uniform(0, 5)
    
    env = Env(initial_x, initial_y, initial_yaw, initial_velocity,cfg.action_repeat)
    cfg.s_shape = tuple(int(x) for x in env.shape)
    cfg.action_shape = tuple(env.action_space)
    cfg.action_dim = 2

    return env

# map_with_obstacles = MapWithObstacles()
# state = np.array([3, 3, 0, 1.1])
# reward, done = map_with_obstacles.calculate_reward(state)
# print(f"Reward for state {state} : {reward}, Done: {done}")

