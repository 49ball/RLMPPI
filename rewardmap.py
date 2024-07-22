import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class RewardCalculator:
    def __init__(self, obstacles):
        self.obstacles = obstacles

    def calculate_reward(self, x, y, v):
        goal = torch.tensor([100, 100], dtype=torch.float32, device='cuda')
        k_att = 10.0
        k_obs = 50.0
        sigma_x = 4.0
        sigma_y = 4.0
        max_reward=1.42
        
        distance_to_goal = torch.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        reward = max_reward-distance_to_goal*0.01
        reward = k_att*reward
        
        obstacles = torch.tensor(self.obstacles, dtype=torch.float32, device='cuda')
        obs_x = obstacles[:, 0]
        obs_y = obstacles[:, 1]
        
        cost = torch.exp(-(((x - obs_x)**2 / sigma_x**2) + ((y - obs_y)**2 / sigma_y**2)))
        reward -= k_obs * torch.sum(cost)*0.01
        
        speed_reward = torch.tensor(1.0, dtype=torch.float32, device='cuda') if v > 0.0 else torch.tensor(0.0, dtype=torch.float32, device='cuda')
        reward += speed_reward
        
        # if torch.any(((x - obs_x)**2 + (y - obs_y)**2) < 3.5**2):
        #     reward = torch.tensor(-1000.0, dtype=torch.float32, device='cuda')

        if distance_to_goal <= 0.5:
            reward = torch.tensor(10.0, dtype=torch.float32, device='cuda')

        elif x <= 0 or x >= 100 or y <= 0 or y >= 100:
            reward = torch.tensor(-5.0, dtype=torch.float32, device='cuda')

        return reward.cpu().numpy().item()

# Define the obstacle positions
obstacles = [
    (10, 10), (35, 20), (45, 50), (70, 70), (50, 80),
    (25, 35), (30, 70), (50, 30), (70, 45), (90, 50)
]

# Parameters
map_size = 100

# Initialize reward map
reward_map = np.zeros((map_size, map_size))

reward_calculator = RewardCalculator(obstacles)

# Calculate the reward for each cell in the grid
for i in range(map_size):
    for j in range(map_size):
        x = torch.tensor(i, dtype=torch.float32, device='cuda')
        y = torch.tensor(j, dtype=torch.float32, device='cuda')
        v = torch.tensor(1.0, dtype=torch.float32, device='cuda')  # Assume a constant speed for simplicity
        reward_map[i, j] = reward_calculator.calculate_reward(x, y, v)
    print(i)

# Plot the reward map as a 3D surface
X, Y = np.meshgrid(range(map_size), range(map_size))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, reward_map, cmap='viridis')

ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_zlabel('Reward')
ax.set_title('3D Surface Plot of Reward Map')
plt.show()
