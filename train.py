import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import torch
import numpy as np
import time
import random
from pathlib import Path
from environment import make_env
import logger
from cfg import parse_cfg #바꿔야 할 부분
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import argparse


__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def evaluate(env, agent, num_episodes, step, eval_numbering):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		all_states=[]
		s, done, ep_reward, t = env.reset(), False, 0, 0
		all_states.append(s.copy())
		while not done:
			action = agent.plan(s, eval_mode=True, step=step, t0=t==0)
			s, reward, done, _ = env.step(action.cpu().numpy())
			all_states.append(s.copy())
			ep_reward += reward
			t += 1
		directory = f'./logs/car/eval/{eval_numbering}'
		if not os.path.exists(directory):
			os.makedirs(directory)
		episode_rewards.append(ep_reward)
		states_df=pd.DataFrame(all_states)
		file_name = f'states_{eval_numbering}_{i}_({int(ep_reward)}).csv'
		file_path = os.path.join(directory, file_name)  # 전체 파일 경로를 만듦
		states_df.to_csv(file_path, index=False)
	return np.nanmean(episode_rewards)


def train(cfg):
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	eval_numbering=0
	work_dir = Path().cwd() / __LOGS__ / cfg.task / str(cfg.seed)
	env,agent,buffer=make_env(cfg),TDMPC(cfg),ReplayBuffer(cfg)
	L=logger.Logger(work_dir,cfg) #로거 시작, wandb 연결
	episode_idx, start_time = 0, time.time() #훈련 에피소드 인덱스와 시작시간 초기화
	for step in range(0, cfg.train_steps+cfg.episode_length,cfg.episode_length):
		s=env.reset() #s 초기화
		episode=Episode(cfg,s) #여기서 s는 state차원과 action 차원을 보내줘야함
		while not episode.done:
			action = agent.plan(s,step=step,t0=episode.first)
			s, reward, done, _ = env.step(action.cpu().numpy()) #다음 state, reward, 끝났는지 확인 ->step으로 episode가 done 이 될때까지!!!!!!!!!!!!!!!!!
			episode += (s, action, reward, done)
		# if len(episode) != cfg.episode_length:
		# 	iter=cfg.episode_length-len(episode)
		# 	flag=0
		# 	while flag < iter:
		# 		episode += (s, action, reward, done) 
		# 		flag+=1
		buffer += episode
		#update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step ==cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer,step+i))
	
		# Log training episode 훈련결과 로그 기록
		episode_idx += 1 
		env_step = int(step*cfg.action_repeat) #액션 반복횟수 * step
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward} #common metrics 초기화
		train_metrics.update(common_metrics) #train metrics에 초기화
		L.log(train_metrics, category='train') #로그 기록

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, eval_numbering)
			eval_numbering+=1
			L.log(common_metrics, category='eval') 

	L.finish(agent) #끝나는 프린트 구문
	print('Training completed successfully')
		
	
	# car_length = cfg.car_length
	# car_width = cfg.car_width
	# car_model = KinematicBicycleModel(cfg.wheelbase, cfg.max_steer, cfg.dt)
	# x, y, yaw, velocity= car_model.update(x=0,y=0,yaw=0,velocity=0,acceleration=1,steering_angle= 0.3)
	# print(x,y,yaw,velocity)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type='str', default='default.yaml', help='insert config file')
	args = parser.parse_args()
	config = args.config

	train(parse_cfg(Path().cwd() / __CONFIG__, config=config))