import csv
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import torch
import numpy as np
import time
import random
from pathlib import Path
from environment import make_env, MapWithObstacles
import logger
from cfg import parse_cfg #바꿔야 할 부분
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer

__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def evaluate(env, agent, num_episodes, step, env_step, eval_numbering, cfg, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	directory = f'./logs/car/eval/{cfg.project}/{eval_numbering}'
	if not os.path.exists(directory):
		os.makedirs(directory)    
	for i in range(num_episodes):
		temp_name = f'states_and_obstacles_{eval_numbering}_{i}.csv'
		temp_file_path = os.path.join(directory, temp_name)
		obs = MapWithObstacles()
		states = []
		with open(temp_file_path, mode='w', newline='') as temp_file:
			writer = csv.writer(temp_file, delimiter='\t')
			s, done, ep_reward, t = env.reset(), False, 0, 0
			obstacle_str = ','.join(f'({x},{y})' for x, y in obs.obstacles)
			writer.writerow([obstacle_str])
			states.append(s.tolist())  # 초기 상태 저장
			if video: video.init(enabled=(i==0))
			while not done:
				action = agent.plan(s, eval_mode=True, step=step, t0=t == 0)
				s, reward, done, _ = env.step(action)
				states.append(s.tolist())  # 상태를 메모리에 저장
				ep_reward += reward
				if video: video.record(s)
				t += 1
			writer.writerows(states)  # 모든 상태를 한 번에 파일에 저장		
		# 에피소드가 끝난 후 파일 이름을 변경하여 보상을 포함
		final_file_name = f'states_and_obstacles_{eval_numbering}_{i}_({int(ep_reward)}).csv'
		final_file_path = os.path.join(directory, final_file_name)
		os.rename(temp_file_path, final_file_path)
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	
	return np.nanmean(episode_rewards)

# def evaluate(env, agent, num_episodes, step, eval_numbering, cfg):
# 	"""Evaluate a trained agent and optionally save a video."""
# 	episode_rewards = []
# 	for i in range(num_episodes):
# 		all_states = []
# 		s, done, ep_reward, t, action = env.reset(), False, 0, 0, [0, 0]
# 		all_states.append(np.append(s.copy(), action))
# 		while not done:
# 			action = agent.plan(s, eval_mode=True, step=step, t0=t==0)
# 			s, reward, done, _ = env.step(action.cpu().numpy())
# 			all_states.append(np.append(s.copy(), action.cpu().numpy()))
# 			ep_reward += reward
# 			t += 1
# 		directory = f'./logs/car/eval/{cfg.project}/{eval_numbering}'
# 		if not os.path.exists(directory):
# 			os.makedirs(directory)
# 		episode_rewards.append(ep_reward)
# 		states_df = pd.DataFrame(all_states)
# 		file_name = f'states_{eval_numbering}_{i}_({int(ep_reward)}).csv'
# 		file_path = os.path.join(directory, file_name)  # 전체 파일 경로를 만듦
# 		states_df.to_csv(file_path, index=False, float_format='%.4f', sep='\t')
# 	return np.nanmean(episode_rewards)

def train(cfg):
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	eval_numbering=0
	work_dir = Path().cwd() / __LOGS__ / cfg.task / str(cfg.seed)
	eval_dir = Path().cwd() / __LOGS__ / cfg.task / 'eval' / str(cfg.project)
	env,agent,buffer=make_env(cfg),TDMPC(cfg),ReplayBuffer(cfg)
	L=logger.Logger(work_dir, eval_dir, cfg) #로거 시작, wandb 연결
	episode_idx, start_time = 0, time.time() #훈련 에피소드 인덱스와 시작시간 초기화
	for step in range(0, cfg.train_steps+cfg.episode_length,cfg.episode_length):
		s=env.reset() #s 초기화
		episode=Episode(cfg,s) #여기서 s는 state차원과 action 차원을 보내줘야함
		while not episode.done:
			action = agent.plan(s,step=step,t0=episode.first)
			s, reward, done, _ = env.step(action) #다음 state, reward, 끝났는지 확인 ->step으로 episode가 done 이 될때까지!!!!!!!!!!!!!!!!!
			episode += (s, action, reward, done, 0)
		episode.real_length=len(episode)
		if len(episode) != cfg.episode_length:
			iter=cfg.episode_length-len(episode)
			flag=0
			while flag < iter:
				episode += (s, action, 0.0, done, 1) 
				flag+=1
		
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
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step ,eval_numbering, cfg, L.video)
			eval_numbering+=1
			L.log(common_metrics, category='eval') 

	L.finish(agent) #끝나는 프린트 구문
	print('Training completed successfully')

	
if __name__ == "__main__":
	train(parse_cfg(Path().cwd() / __CONFIG__))