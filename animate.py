import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Rectangle
from math import sqrt,cos,sin
import os

class MapWithObstacles:
    def __init__(self, map_size=100, obstacle_count=10, obstacle_radius=3, seed=None):
        self.map_size = map_size
        self.obstacle_count = obstacle_count
        self.obstacle_radius = obstacle_radius
        self.seed = seed
        self.obstacles = []
        if seed is not None:
            np.random.seed(seed)  # 재현 가능한 결과를 위해 시드 설정
        self.generate_obstacles()

    def generate_obstacles(self):
        # 지정된 위치에 장애물 생성
        self.obstacles = [
            (10, 10), (35, 20), (45, 50), (70, 70), (50, 80),
            (25, 35), (30, 70), (50, 30), (70, 45), (90, 50)
        ]

    def plot_map(self, ax):
        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        # for (x, y) in self.obstacles:
        #     circle = Circle((x, y), self.obstacle_radius, fc='gray', edgecolor='black')
        #     ax.add_patch(circle)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(f'{self.map_size}x{self.map_size} Map with Circular Obstacles')
        plt.grid(True)

class CarAnimation:
    def __init__(self, csv_file, car_length, car_width, dt, map_obj):
        self.car_length = car_length
        self.car_width = car_width
        self.dt = dt
        self.positions = pd.read_csv(csv_file, delimiter='\t').values[1:, :3]  
        self.car_diagonal = sqrt((car_length / 2)**2 + car_width**2)
        self.map_obj = map_obj

    def plot_simulation(self, filename='./video/car_path_simulation.mp4'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        line, = ax.plot([], [], 'b-', label='Path of the car')
        car_marker = Rectangle((0, 0), width=self.car_length, height=self.car_width, fc='red', ec='black', angle=0)
        self.map_obj.plot_map(ax)

        def init():
            line.set_data([], [])
            ax.add_patch(car_marker)
            return line, car_marker

        def update(frame):
            line.set_data(self.positions[:frame, 0], self.positions[:frame, 1])
            x_dif = (self.car_length / self.car_diagonal) * cos(self.positions[frame-1, 2]) - (self.car_width / self.car_diagonal) * sin(self.positions[frame-1, 2])
            y_dif = (self.car_width / self.car_diagonal) * cos(self.positions[frame-1, 2]) + (self.car_length / self.car_diagonal) * sin(self.positions[frame-1, 2])
            car_marker.set_xy([self.positions[frame-1, 0] - self.car_diagonal * x_dif / 2, self.positions[frame-1, 1] - self.car_diagonal * y_dif / 2])
            car_marker.angle = np.rad2deg(self.positions[frame-1, 2])

            return line, car_marker

        ani = FuncAnimation(fig, update, frames=len(self.positions), init_func=init, blit=True, interval=self.dt * 1000)
        ax.scatter(0, 0, color='red', s=1000, label='Start')
        ax.scatter(100, 100, color='green', s=1000, label='Goal')
        ax.set_xlim(0, 100)
        ax.set_xlim(0, 100)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title('Car Path Simulation')
        ax.legend()
        ax.grid()
        ax.axis('equal')

        writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(filename, writer=writer)

# 파라미터 설정
car_length = 3.0  # 차량의 길이(미터)
car_width = 1.5   # 차량의 너비(미터)
dt = 0.1  # 시간 간격(초)
csv_file = './logs/car/eval/replay_buffer_change/3/states_3_4_(54).csv'  # 입력 데이터 파일 경로

# 맵 생성
map_with_obstacles = MapWithObstacles()

# 애니메이션 생성 및 저장
car_animation = CarAnimation(csv_file, car_length, car_width, dt, map_with_obstacles)
car_animation.plot_simulation()
print("----------Animate Success----------")
