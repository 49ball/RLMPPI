import os
import datetime
import pandas as pd
import torch
import wandb
import re
import numpy as np
from termcolor import colored
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Rectangle
from math import sqrt, cos, sin
from environment import MapWithObstacles

CONSOLE_FORMAT = [('episode', 'E', 'int'), ('env_step', 'S', 'int'), ('episode_reward', 'R', 'float'), ('total_time', 'T', 'time')]
AGENT_METRICS = ['reward_loss', 'value_loss', 'total_loss', 'weighted_loss', 'pi_loss', 'grad_norm']



def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path

def print_run(cfg, reward=None):
	"""Pretty-printing of run information. Call at start of training."""
	prefix, color, attrs = '  ', 'green', ['bold']
	def limstr(s, maxlen=32):
		return str(s[:maxlen]) + '...' if len(str(s)) > maxlen else s
	def pprint(k, v):
		print(prefix + colored(f'{k.capitalize()+":":<16}', color, attrs=attrs), limstr(v))
	kvs = [('task', cfg.task_title),
		   ('train steps', f'{int(cfg.train_steps*cfg.action_repeat):,}'),
		   ('observations', 'x'.join([str(s) for s in cfg.s_shape])),
		   ('actions', cfg.action_dim),
		   ('experiment', cfg.exp_name)]
	if reward is not None:
		kvs.append(('episode reward', colored(str(int(reward)), 'white', attrs=['bold'])))
	w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
	div = '-'*w
	print(div)
	for k,v in kvs:
		pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""Return a wandb-safe group name for logging. Optionally returns group name as list."""
	lst = [cfg.task, cfg.modality, re.sub('[^0-9a-zA-Z]+', '-', cfg.exp_name)]
	return lst if return_list else '-'.join(lst)


class CarAnimationRecorder:
	def __init__(self, wandb, csv_file, render_size=384, fps=15):
		self.car_length = 3
		self.car_width = 1.5
		self._wandb=wandb
		self.dt = 0.1
		self.car_diagonal = sqrt((self.car_length / 2)**2 + self. car_width**2)
		self.map_obj = MapWithObstacles()
		self.render_size = render_size
		self.fps = fps
		self.states = []
		self.frames = []
		self.enabled = False

	def init(self, enabled=True):
		self.frames = []
		self.states = []
		self.enabled = enabled

	def record(self, state, obstacles):
		if self.enabled:
			fig, ax = plt.subplots(figsize=(10, 10))
			self.map_obj.plot_map(ax, obstacles)
			car_marker = Rectangle((0, 0), width=self.car_length, height=self.car_width, fc='red', ec='black', angle=0)
			ax.add_patch(car_marker)
			line, = ax.plot([], [], 'b-', label='Path of the car')

			x, y, theta = state[0].item(), state[1].item(), state[2].item()

			line.set_data(x,y)
			x_dif = (self.car_length / self.car_diagonal) * cos(theta) - (self.car_width / self.car_diagonal) * sin(theta)
			y_dif = (self.car_width / self.car_diagonal) * cos(theta) + (self.car_length / self.car_diagonal) * sin(theta)
			car_marker.set_xy([x - self.car_diagonal * x_dif / 2, y - self.car_diagonal * y_dif / 2])
			car_marker.angle = np.rad2deg(theta)

			ax.scatter(0, 0, color='red', s=180, label='Start')
			ax.scatter(100, 100, color='green', s=180, label='Goal')
			ax.set_xlim(0, 100)
			ax.set_ylim(0, 100)
			ax.set_xlabel('X position (m)')
			ax.set_ylabel('Y position (m)')
			ax.set_title('Car Path Simulation')
			ax.legend()
			ax.grid()
			ax.axis('equal')

			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			self.frames.append(image)
			plt.close(fig)

	def save(self, step):
		if self.enabled:
			frames = np.stack(self.frames).transpose(0, 3, 1, 2)
			self._wandb.log({'eval_video': self._wandb.Video(frames, fps=self.fps, format='mp4')}, step=step)
			print("Animation saved successfully")


class Logger:
	def __init__(self, log_dir, eval_dir, cfg):
		self._log_dir = make_dir(log_dir)
		self._eval_dir = make_dir(eval_dir)
		self._model_dir = make_dir(self._log_dir / 'models')
		self._save_model = cfg.save_model
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._cfg = cfg
		self._eval = []
		print_run(cfg)
		project, entity = cfg.get('wandb_project', 'none'), cfg.get('wandb_entity', 'none')
		run_offline = not cfg.get('use_wandb', False) or project == 'none' or entity == 'none'
		if run_offline:
			print(colored('Logs will be saved locally.', 'yellow', attrs=['bold']))
			self._wandb = None
		else:
			try:
				os.environ["WANDB_SILENT"] = "true"
				wandb.init(project=project,
						   entity=entity,
						   name=str(cfg.seed),
						   group=self._group,
						   tags=cfg_to_group(cfg, return_list=True) + [f'seed:{cfg.seed}'],
						   dir=self._log_dir,
						   config=OmegaConf.to_container(cfg, resolve=True))
				print(colored('Logs will be synced with wandb.', 'blue', attrs=['bold']))
				self._wandb = wandb
			except:
				print(colored('Warning: failed to init wandb. Logs will be saved locally.', 'yellow'), attrs=['bold'])
				self._wandb = None
		self._video = CarAnimationRecorder(self._wandb, eval_dir) if self._wandb and cfg.save_video else None

	@property
	def video(self):
		return self._video

	def finish(self, agent):
		if self._save_model:
			fp = self._model_dir / f'model_{self._cfg.project}.pt'
			torch.save(agent.state_dict(), fp)
			if self._wandb:
				artifact = self._wandb.Artifact(self._group + '-' + str(self._seed), type='model')
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)
				
	def _format(self, key, value, ty):
		if ty == 'int':
			return f'{colored(key+":", "grey")} {int(value):,}'
		elif ty == 'float':
			return f'{colored(key+":", "grey")} {value:.01f}'
		elif ty == 'time':
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "grey")} {value}'
		else:
			raise f'invalid log format type: {ty}'

	def _print(self, d, category):
		category = colored(category, 'blue' if category == 'train' else 'green')
		pieces = [f' {category:<14}']
		for k, disp_k, ty in CONSOLE_FORMAT:
			pieces.append(f'{self._format(disp_k, d.get(k, 0), ty):<26}')
		print('   '.join(pieces))

	def log(self, d, category='train'):
		assert category in {'train', 'eval'}
		if self._wandb is not None:
			for k,v in d.items():
				self._wandb.log({category + '/' + k: v}, step=d['env_step'])
		if category == 'eval':
			keys = ['env_step', 'episode_reward']
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(self._log_dir / 'eval.log', header=keys, index=None)
		self._print(d, category)

	def log_state_action(self, state, action, step, category='train'):
		if self._wandb is not None:
			log_data = {
				f'{category}/state': wandb.Histogram(state),
				f'{category}/action': wandb.Histogram(action)
			}
			self._wandb.log(log_data, step=step)