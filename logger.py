import numpy as np

class EpisodeLogger():
	def init(self, initial_state):
		self.states = [initial_state]
		self.rewards = []
		self.actions = []

	def log_step(action, reward, next_state, is_terminal):
		self.states.append(next_state)
		self.rewards.append(reward)
		self.actions.append(action)

	@property
	def step_count(self):
		return len(self.actions)
	
	@property
	def mean_reward(self):
		return mean(self.rewards)