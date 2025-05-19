import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import Double_Q_Net, Policy_Net, ReplayBuffer
import matplotlib.pyplot as plt


class SAC():
	def __init__(self, state_dim, action_dim):
		self.dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.gamma = 0.99
		self.hid_shape = [128, 128]
		self.lr = 3e-4
		self.batch_size = 256
		self.alpha = 0.8
		self.adaptive_alpha = True
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.tau = 0.005
		self.H_mean = 0
		self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))

		self.actor = Policy_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

		self.q_critic = Double_Q_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		for p in self.q_critic_target.parameters(): p.requires_grad = False

		if self.adaptive_alpha:
			self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)


	def select_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
			probs = self.actor(state)
			a = Categorical(probs).sample().item()
			
			return a


	def train(self):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		#------------------------------------------ Train Critic ----------------------------------------#
		'''Compute the target soft Q value'''
		with torch.no_grad():
			next_probs = self.actor(s_next)
			next_log_probs = torch.log(next_probs+1e-8)
			next_q1_all, next_q2_all = self.q_critic_target(s_next)
			min_next_q_all = torch.min(next_q1_all, next_q2_all)
			v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True)
			target_Q = r + (~dw) * self.gamma * v_next

		'''Update soft Q net'''
		q1_all, q2_all = self.q_critic(s)
		q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a)
		q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		#------------------------------------------ Train Actor ----------------------------------------#
		probs = self.actor(s)
		log_probs = torch.log(probs + 1e-8)
		with torch.no_grad():
			q1_all, q2_all = self.q_critic(s)
		min_q_all = torch.min(q1_all, q2_all)

		a_loss = torch.sum(probs * (self.alpha*log_probs - min_q_all), dim=1, keepdim=False)

		self.actor_optimizer.zero_grad()
		a_loss.mean().backward()
		self.actor_optimizer.step()

		#------------------------------------------ Train Alpha ----------------------------------------#
		if self.adaptive_alpha:
			with torch.no_grad():
				self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
			alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()

			self.alpha = self.log_alpha.exp().item()

		#------------------------------------------ Update Target Net ----------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self):
		torch.save(self.actor.state_dict(), f"./model/sac_actor_FB.pth")
		torch.save(self.q_critic.state_dict(), f"./model/sac_critic_FB.pth")


	def load(self):
		self.actor.load_state_dict(torch.load(f"./model/sac_actor_FB.pth", map_location=self.dvc))
		self.q_critic.load_state_dict(torch.load(f"./model/sac_critic_FB.pth", map_location=self.dvc))
	

	def save_graph(self, data):
		fig = plt.figure(1)

		plt.xlabel('Кол-во шагов')
		plt.ylabel('Награда')
		plt.plot([x[0] for x in data], [x[1] for x in data])
		plt.subplots_adjust(wspace=1.0, hspace=1.0)

		fig.savefig('./graph.png')
		plt.close(fig)