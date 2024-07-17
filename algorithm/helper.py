import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


__REDUCE__ = lambda b: 'mean' if b else 'none'


def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
	"""Utility function. Returns the output shape of a network for a given input shape."""
	x = torch.randn(*in_shape).unsqueeze(0)
	return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


def ema(m, m_target, tau):  #targent network와 원래 network의 파라미터에 점진적으로 수렴하도록 업데이트하는 과정
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau) # ptarget=tau*p+(1-tau)*ptarget


def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)


class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))

def q(cfg, act_fn=nn.ELU()):
	"""Returns a Q-function that uses Layer Normalization."""
	return nn.Sequential(nn.Linear(cfg.state_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
						 nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
						 nn.Linear(cfg.mlp_dim, 1))


class RandomShiftsAug(nn.Module): #data augmentation
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, cfg):
		super().__init__()
		self.pad = int(cfg.img_size/21) if cfg.modality == 'pixels' else None

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)



class Episode(object):
	"""Storage object for a single episode."""
	def __init__(self, cfg, init_s):
		self.cfg = cfg
		self.device = torch.device(cfg.device) #무슨 device인지
		dtype = torch.float32 if cfg.modality == 'state' else torch.uint8 
		self.s = torch.empty((cfg.episode_length+1, *init_s.shape), dtype=dtype, device=self.device) #state의 에피소드 크기와 각각의 state
		self.s[0] = torch.tensor(init_s, dtype=dtype, device=self.device) #에피소드 첫번째의 state
		self.action = torch.empty((cfg.episode_length, cfg.action_dim), dtype=torch.float32, device=self.device) #에피소드별 액션 차원
		self.reward = torch.empty((cfg.episode_length,), dtype=torch.float32, device=self.device) #에피소드별 보상
		self.cumulative_reward = 0
		self.done = False
		self._idx = 0
	
	def __len__(self):
		return self._idx

	@property
	def first(self):
		return len(self) == 0
	
	def __add__(self, transition):
		self.add(*transition)
		return self

	def add(self, s, action, reward, done):
		self.s[self._idx+1] = torch.tensor(s, dtype=self.s.dtype, device=self.s.device)
		self.action[self._idx] = action
		self.reward[self._idx] = reward
		self.cumulative_reward += reward
		self.done = done
		self._idx += 1


class ReplayBuffer():
	"""
	Storage and sampling functionality for training TD-MPC / TOLD.
	The replay buffer is stored in GPU memory when training from state.
	Uses prioritized experience replay by default.
	"""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
		dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
		s_shape = cfg.s_shape if cfg.modality == 'state' else (3, *cfg.s_shape[-2:])
		self._s = torch.empty((self.capacity+1, *s_shape), dtype=dtype, device=self.device)
		self._last_s = torch.empty((self.capacity//cfg.episode_length, *cfg.s_shape), dtype=dtype, device=self.device)
		self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
		self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
		self._eps = 1e-6
		self._full = False
		self.idx = 0

	def __add__(self, episode: Episode):
		self.add(episode)
		return self

	def add(self, episode: Episode):
		self._s[self.idx:self.idx+self.cfg.episode_length] = episode.s[:-1] if self.cfg.modality == 'state' else episode.s[:-1, -3:]
		self._last_s[self.idx//self.cfg.episode_length] = episode.s[-1]
		self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action #epsiode_length만큼 계속 채우기
		self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward #epsiode_length만큼 계속 채우기

		if self._full:
			max_priority = self._priorities.max().to(self.device).item()
		else:
			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
		mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.horizon
		new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
		new_priorities[mask] = 0
		self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
		self.idx = (self.idx + self.cfg.episode_length) % self.capacity
		self._full = self._full or self.idx == 0

	def update_priorities(self, idxs, priorities):
		self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

	def _get_s(self, arr, idxs):
		if self.cfg.modality == 'state':
			return arr[idxs]
		s = torch.empty((self.cfg.batch_size, 3*self.cfg.frame_stack, *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
		s[:, -3:] = arr[idxs].cuda()
		_idxs = idxs.clone()
		mask = torch.ones_like(_idxs, dtype=torch.bool)
		for i in range(1, self.cfg.frame_stack):
			mask[_idxs % self.cfg.episode_length == 0] = False
			_idxs[mask] -= 1
			s[:, -(i+1)*3:-i*3] = arr[_idxs].cuda()
		return s.float()

	def sample(self):
		probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
		probs /= probs.sum()
		total = len(probs)
		idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
		weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
		weights /= weights.max()

		s = self._get_s(self._s, idxs)
		next_s_shape = self._last_s.shape[1:] if self.cfg.modality == 'state' else (3*self.cfg.frame_stack, *self._last_s.shape[-2:])
		next_s = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *next_s_shape), dtype=s.dtype, device=s.device)
		action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
		reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)

		# 유효한 인덱스 찾기
		valid_idx = (self._reward != 0.0).nonzero(as_tuple=False).squeeze() #리워드가 0이 아닌 부분 찾기

		for t in range(self.cfg.horizon+1): #horizon만큼 진행함
			if t==0:
				_idxs = idxs + t
			else:
				_idxs=_idxs + t
			invalid_mask = self._reward[_idxs] == 0.0 #배치사이즈만큼 뽑아서 리워드가0인 부분 마스킹
			if invalid_mask.any():
				valid_indices = valid_idx.unsqueeze(0).expand(invalid_mask.sum(), -1) #invalid_mask의 true인 개수만큼 차원 증가
				invalid_indices = _idxs[invalid_mask].unsqueeze(1) #invalid_mask가 true인 부분만 저장
				next_valid_indices = valid_indices > invalid_indices #값들 비교해서 valid_indices가 큰부분이 True로
				next_valid_indices = next_valid_indices.to(dtype=torch.int) #자료형 변환
				#_idxs 중에서 invalid_mask가 true인 값들은 바로 다음 valid한 값으로 넘어가도록함
				_idxs[invalid_mask] = valid_idx[next_valid_indices.argmax(dim=1)] #다음값은 무조건 done이든 뭐든 reward가 존재하니까 가능
			next_s[t] = self._get_s(self._s, _idxs+1)
			action[t] = self._action[_idxs]
			reward[t] = self._reward[_idxs]

			mask=self._reward[_idxs+1]==0.0
			next_s[t,mask]=self._last_s[_idxs[mask]//self.cfg.episode_length].cuda().float()

		# mask = (_idxs+1) % self.cfg.episode_length == 0 #마지막인지 확인
		# #next_s 마지막 타임스텝에서 mask 가 true일때 마지막 state를 넣어줌
		# next_s[-1, mask] = self._last_s[_idxs[mask]//self.cfg.episode_length].cuda().float() 
		if not action.is_cuda:
			action, reward, idxs, weights = action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()

		return s, next_s, action, reward.unsqueeze(2), idxs, weights

# class ReplayBuffer():
# 	"""
# 	Storage and sampling functionality for training TD-MPC / TOLD.
# 	The replay buffer is stored in GPU memory when training from state.
# 	Uses prioritized experience replay by default."""
# 	def __init__(self, cfg):
# 		self.cfg = cfg
# 		self.device = torch.device(cfg.device)
# 		self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
# 		dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
# 		s_shape = cfg.s_shape if cfg.modality == 'state' else (3, *cfg.s_shape[-2:])
# 		self._s = torch.empty((self.capacity+1, *s_shape), dtype=dtype, device=self.device) #(max 버퍼사이즈 , s_shape)
# 		self._last_s = torch.empty((self.capacity//cfg.episode_length, *cfg.s_shape), dtype=dtype, device=self.device)
# 		self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
# 		self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
# 		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
# 		self._eps = 1e-6
# 		self._full = False
# 		self.idx = 0

# 	def __add__(self, episode: Episode):
# 		self.add(episode)
# 		return self

# 	def add(self, episode: Episode):
# 		self._s[self.idx:self.idx+self.cfg.episode_length] = episode.s[:-1] if self.cfg.modality == 'state' else episode.s[:-1, -3:]
# 		self._last_s[self.idx//self.cfg.episode_length] = episode.s[-1]
# 		self._action[self.idx:self.idx+self.cfg.episode_length] = episode.action
# 		self._reward[self.idx:self.idx+self.cfg.episode_length] = episode.reward
# 		if self._full:
# 			max_priority = self._priorities.max().to(self.device).item()
# 		else:
# 			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
# 		mask = torch.arange(self.cfg.episode_length) >= self.cfg.episode_length-self.cfg.horizon
# 		new_priorities = torch.full((self.cfg.episode_length,), max_priority, device=self.device)
# 		new_priorities[mask] = 0
# 		self._priorities[self.idx:self.idx+self.cfg.episode_length] = new_priorities
# 		self.idx = (self.idx + self.cfg.episode_length) % self.capacity
# 		self._full = self._full or self.idx == 0

# 	def update_priorities(self, idxs, priorities):
# 		self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

# 	def _get_s(self, arr, idxs):
# 		if self.cfg.modality == 'state':
# 			return arr[idxs]
# 		s = torch.empty((self.cfg.batch_size, 3*self.cfg.frame_stack, *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
# 		s[:, -3:] = arr[idxs].cuda()
# 		_idxs = idxs.clone()
# 		mask = torch.ones_like(_idxs, dtype=torch.bool)
# 		for i in range(1, self.cfg.frame_stack):
# 			mask[_idxs % self.cfg.episode_length == 0] = False
# 			_idxs[mask] -= 1
# 			s[:, -(i+1)*3:-i*3] = arr[_idxs].cuda()
# 		return s.float()

# 	def sample(self):
# 		probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
# 		probs /= probs.sum()
# 		total = len(probs)
# 		idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
# 		weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
# 		weights /= weights.max()

# 		s = self._get_s(self._s, idxs)
# 		next_s_shape = self._last_s.shape[1:] if self.cfg.modality == 'state' else (3*self.cfg.frame_stack, *self._last_s.shape[-2:])
# 		next_s = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *next_s_shape), dtype=s.dtype, device=s.device)
# 		action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
# 		reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)
# 		for t in range(self.cfg.horizon+1):
# 			_idxs = idxs + t
# 			next_s[t] = self._get_s(self._s, _idxs+1)
# 			action[t] = self._action[_idxs]
# 			reward[t] = self._reward[_idxs]

# 		mask = (_idxs+1) % self.cfg.episode_length == 0
# 		next_s[-1, mask] = self._last_s[_idxs[mask]//self.cfg.episode_length].cuda().float()
# 		if not action.is_cuda:
# 			action, reward, idxs, weights = \
# 				action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()

# 		return s, next_s, action, reward.unsqueeze(2), idxs, weights


def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)
