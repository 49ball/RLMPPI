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
from environment import make_env
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

def evaluate(env, agent, num_episodes, step, eval_numbering, cfg):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	directory = f'./logs/car/eval/{cfg.project}/{eval_numbering}'
	if not os.path.exists(directory):
		os.makedirs(directory)
	for i in range(num_episodes):
		temp_file_path = os.path.join(directory, f'temp_states_{i}.csv')
		with open(temp_file_path, mode='w', newline='') as temp_file:
			writer = csv.writer(temp_file, delimiter='\t')
			s, done, ep_reward, t = env.reset(), False, 0, 0
			writer.writerow(s.tolist())  # 초기 상태 저장
			while not done:
				action = agent.plan(s, eval_mode=True, step=step, t0=t == 0)
				s, reward, done, _ = env.step(action)
				writer.writerow(s.tolist())  # 상태를 파일에 저장
				ep_reward += reward
				t += 1
		# 에피소드가 끝난 후에 파일 이름을 변경
		final_file_name = f'states_{eval_numbering}_{i}_({int(ep_reward)}).csv'
		final_file_path = os.path.join(directory, final_file_name)
		os.rename(temp_file_path, final_file_path)
		episode_rewards.append(ep_reward)
	
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
	env,agent,buffer=make_env(cfg),TDMPC(cfg),ReplayBuffer(cfg)
	L=logger.Logger(work_dir,cfg) #로거 시작, wandb 연결
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
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, eval_numbering, cfg)
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
	train(parse_cfg(Path().cwd() / __CONFIG__))