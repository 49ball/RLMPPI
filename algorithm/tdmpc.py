import time
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h
from dynamics import KinematicBicycleModel

class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._dynamics=KinematicBicycleModel()
		self._reward = h.mlp(cfg.state_dim+cfg.action_dim, cfg.mlp_dim, 1)
		self._pi = h.mlp(cfg.state_dim, cfg.mlp_dim, cfg.action_dim)
		self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
		self.apply(h.orthogonal_init)
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

	def next(self, s, a):
		"""Predicts next latent state (d) and single-step reward (R)."""		
		x = torch.cat([s, a], dim=-1)
		return self._dynamics.update(s,a), self._reward(x) #mlp로 뽑아낸 값들

	def pi(self, s, std=0): #policy로부터 sampling, 평균 반환
		"""Samples an action from the learned policy (pi)."""
		mu = torch.tanh(self._pi(s)) #행동공간 정규화
		if std > 0:
			std = torch.ones_like(mu) * std #mu와 같은 크기의 텐서 만듬
			return h.TruncatedNormal(mu, std).sample(clip=0.3) #샘플링된 값을 -0.3에서 0.3으로 제한
		return mu 

	def Q(self, s, a): # final state-action value prediction
		"""Predict state-action value (Q)."""
		x = torch.cat([s, a], dim=-1)
		return self._Q1(x), self._Q2(x)


class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		self.model = TOLD(cfg).cuda()
		self.model_target = deepcopy(self.model) #target 모델 형성
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr) #TOLD 모델 optimizer
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr) #policy 모델 optimizer
		self.model.eval()
		self.model_target.eval()

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict()}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])

	@torch.no_grad()#no_grad는 미분기능을 비활성화 시키는것
	def estimate_value(self, s, actions, horizon): #Horizon의 Total Return을 계산하는 함수 -> 추론된 리워드 모델을 바탕으로 reward 계산
		"""Estimate value of a trajectory starting at state s and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			s, reward = self.model.next(s, actions[t]) # s도 계속 업데이트 해줘야함!!!
			G += discount * reward #Reward 합
			discount *= self.cfg.discount
		G += discount * torch.min(*self.model.Q(s, self.model.pi(s, self.cfg.min_std))) #terminal reward, s와 action 값을 넣고 value를 계산 
		return G

	@torch.no_grad()
	def plan(self, s, eval_mode=False, step=None, t0=True): #액션을 뽑아내는 함수(MPPI)
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

		# Sample policy trajectories
		s1 = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step))) #horizon 의 위치가 어느정도인지 표현
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples) #policy의 trajectory 개수
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
			s = s1.repeat(num_pi_trajs, 1) #TOLD를 활용해서 encoding observation, 배열이 만들어짐(num_pio_trajs,1) 크기
			
			#breakpoint()
			for t in range(horizon): #이부분은 pi_actions[t]를 채우기 위해 필요한 과정
				pi_actions[t] = self.model.pi(s, self.cfg.min_std) #samples an action and save
				s, _ = self.model.next(s, pi_actions[t])  #next space and single-step reward -> pi_actions를 뽑아내주기 위해서 s도 같이 뽑아냄

		# Initialize state and parameters 초기화하는 작업
		s = s1.repeat(self.cfg.num_samples+num_pi_trajs, 1) #샘플링 개수에 policy trajectory개수까지 추가
		mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device) #평균 초기화
		std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device) #표준편차 초기화
		if not t0 and hasattr(self, '_prev_mean'): #첫스텝이 아니고 _prev_mean속성값이 존재하면 mean값 업그레이드
			mean[:-1] = self._prev_mean[1:] #이전평균의 두번째 요소부터 마지막 요소까지를 현재 평균의 첫번째 요소부터 마지막 -1 요소까지로 설정

		# Iterate CEM
		for i in range(self.cfg.iterations):

			#mean 텐서의 두번째 차원에 새로운 차원 추가 ex)3x2 -> 3x1x2, clamp함수는 일정범위안으로 바꿔주는것, 랜덤 샘플링해서 -1,1사이로 샘플링 
			#torch randn 함수는 정규분포라서 아래와같은 작업 수행 (X=μ+σZ,Z=N(0,1))
			actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -2, 2)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=1) #샘플링한 action과 policy action합치기
			actions[:, :, 1] = torch.clamp(actions[:, :, 1], -0.6981, 0.6981)  # Second action dimension: -0.6981 to 0.6981
			# Compute elite actions
			value = self.estimate_value(s, actions, horizon).nan_to_num_(0) #추론된 reward 모델을 통해 reward계산
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices #reward 가장 높은 k개의 action 뽑기
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs] #각 값들 저장
			# Update parameters
			max_value = elite_value.max(0)[0] #reward 값중 제일 큰값, max(0)은 행렬에서 제일 큰값, [0]은 인덱스대신 값을 반환
			score = torch.exp(self.cfg.temperature*(elite_value - max_value)) #expotential 값 반환 -> cost function 정의 없이 elite값들끼리 비교해서 가중치 매김
			score /= score.sum(0) #점수를 정규화하여 모든점수의 합이 1이 되도록 함
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9) #엘리트 행동의 가중평균을 계산해서 새로운 평균을 얻음, 1e-9는 계산안전성을위해 추가된 작은값
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))#엘리트 행동의 가중표준편차 계산
			_std = _std.clamp_(self.std, 2) # 표준편차를 self.std와 2사이로 clamping, 일관된 탐색을 위해 제한을 걸어둠, clamp_는 원래 텐서값을 변경함
			mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std #momentum을 이용해 평균을 업데이트함

		# Outputs
		score = score.squeeze(1).cpu().numpy() #점수텐서를 1차원으로 변환하고 확률분포 기반 샘플링을 위해 numpy로 변환
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)] #엘리트 행동중 무작위로 action 저장
		self._prev_mean = mean #현재 mean값을 저장하여 다음반복에서 사용하도록 설정
		mean, std = actions[0], _std[0] #선택된 행동의 첫번째값과 현재 표준편차의 첫번째값을 사용하여 새로운 평균과 표준편차 설정

		a = mean
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device) #평가모드가 아니면 (X=μ+σZ,Z=N(0,1)) 행동 생성

		return a

	def update_pi(self, ss): #학습중인 Q 함수를 통해 pi loss 업데이트
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True) #optimizer gradient 초기화, 메모리절약을 위해 gradient none으로 설정
		self.model.track_q_grad(False) #Q-value네트워크의 gradient tracking 중지(Q-value 네트워크의 파라미터가 업데이트되지 않도록 하기위함)

		# Loss is a weighted sum of Q-values
		pi_loss = 0
		for t,s in enumerate(ss):
			a = self.model.pi(s, self.cfg.min_std) #action sampling
			Q = torch.min(*self.model.Q(s, a)) #2개의 Q-function,state,action pair로부터 predict한 value 두개중 minimum을 Q로 저장 -> Q학습 안된것이 더 잘되게 하기 위해
			pi_loss += -Q.mean() * (self.cfg.rho ** t) #정책손실을 누적, rho**t는 시간에 따른 할인율

		pi_loss.backward() #정책손실 gradient 계산
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False) #gradient exploding 방지용 
		self.pi_optim.step() #정책 optimizer을 사용해서 네트워크 파라미터 업데이트
		self.model.track_q_grad(True) #Q-Value 네트워크 그래디언트 추적 다시 활성화
		return pi_loss.item() #계산된 정책손실값 반환

	@torch.no_grad()
	def _td_target(self, next_ss, reward):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q(next_ss, self.model.pi(next_ss, self.cfg.min_std))) #reward+감가율*value(target model에서 encoding 된 obs와 action)
		return td_target

	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		s, next_ses, action, reward, idxs, weights = replay_buffer.sample() #버퍼들중 무작위 값 불러와서 값들 초기화
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train() #training 모드

		# Representation
		ss = [s.detach()] #메모리 사용량을 줄이기위해 따로 변수 저장(그래디언트가 전달되지않음)
		reward_loss, value_loss, priority_loss = 0, 0, 0
		for t in range(self.cfg.horizon):#horizon 만큼

			# Predictions
			Q1, Q2 = self.model.Q(s, action[t]) #state action value 뽑아냄
			s, reward_pred = self.model.next(s, action[t]) #told 모델로부터 리워드 뽑기
			with torch.no_grad():				
				td_target = self._td_target(next_ses[t], reward[t]) #다음 step의 td target을 계산함
			ss.append(s.detach())
			# Losses
			rho = (self.cfg.rho ** t)
			reward_loss += rho * h.mse(reward_pred, reward[t]) #reward prediction loss ->리워드 업데이트 목적함수
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target)) #value prediction loss -> value업데이트 목적함수
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target)) #리플레이 버퍼 우선순위 설정 

		# Optimize model
		total_loss = self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4)
		weighted_loss = (total_loss.squeeze(1) * weights).mean() #total loss에 각 샘플 가중치 들을 곱해서 평균내 최종 weighted loss를 구함
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon)) #gradient를 후크로 등록해서 역전파과정에서 gradient를 수정함. gradient를 1/self.cfg.horizon 배율로 조정
		weighted_loss.backward() #역전파를 통해 gradient 계산
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False) #gradient clippping
		self.optim.step() #optimizer을 이용해 파라미터 업데이트
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach()) #리플레이 버퍼의 우선순위 업데이트, priority loss는 0.001을 초과하지 않도록 클램핑함, gradient 추적되지않도록 detach
		# Update policy + target network
		pi_loss = self.update_pi(ss) #policy의 loss 반환 및 policy 모델 파라미터 업데이트
		if step % self.cfg.update_freq == 0: #업데이트 freq마다 target network의 파라미터를 온라인 네트워크의 파라미터로 업데이트함
			h.ema(self.model, self.model_target, self.cfg.tau) 

		self.model.eval() #모델 평가모드
		return {'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm)}
