#!/usr/bin/env python3
import torch, numpy as np, gym, sys, copy, argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque
import random
import pdb

class LinearQNetwork(nn.Module):
	def  __init__(self,obs_dim,action_dim):
		super().__init__()
		self.lin = nn.Linear(obs_dim,action_dim)

	def forward(self,obs):
		return F.softmax(self.lin(obs),dim=1)

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		pass

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		pass

	def load_model(self, model_file):
		# Helper function to load an existing model.
		pass

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		pass

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		self.Transition = namedtuple('Transition', ('state', 'action', 'done', 'next_state','reward'))
		self.memory = deque(maxlen=memory_size)

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		random_batch = random.sample(self.memory,batch_size)
		return self.Transition(*zip(*random_batch))

	def append(self, *args):
		# Appends transition to the memory.
		self.memory.append(self.Transition(*args))

	def len(self):
		return len(self.memory)

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 

		self.env =  gym.make(environment_name)
		self.qnet = LinearQNetwork(self.env.observation_space.shape[0],self.env.action_space.n)
		self.memory = Replay_Memory(memory_size=1)
		self.eps_start = 0.5
		self.eps_end = 0.05
		self.eps_iter = 100000
		self.eps = self.eps_start
		self.num_iter = 1e6
		self.num_episodes = 3000
		self.max_ep_length = 1000
		self.render = render
		self.batch_size = 1
		self.gamma = 0.99
		self.lr = 1e-4
		self.buffer_size = 50000
		self.loss = nn.MSELoss()
		self.optim = torch.optim.Adam(self.qnet.parameters(),lr=self.lr)

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		for ep in range(self.num_episodes):
			obs = self.env.reset()
			episode_reward = 0
			if self.eps>0.05:
				self.eps = -0.0009*ep + 0.5	 
			for iteration in range(self.max_ep_length):
				if self.render:
					self.env.render()
				if np.random.rand()<self.eps:
					act = self.env.action_space.sample()
				else:
					_,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0))),dim=1)
					act = act.data[0]
				next_obs,reward,done,_ = self.env.step(act)
				self.memory.append(obs,act,done,next_obs,reward)
				if self.memory.len()<self.batch_size:
					continue
				batch_obs,batch_act,batch_done,batch_next_obs,batch_reward = self.memory.sample_batch(batch_size=1)
				y = Variable(torch.zeros(len(batch_reward)),requires_grad=True)
				targetQ = torch.max(self.qnet(Variable(torch.FloatTensor(batch_next_obs))),dim=1)
				for j in range(len(batch_obs)):
					if batch_done[j]:
						y[j] = batch_reward[j]
					else:
						y[j] = batch_reward[j] + self.gamma*targetQ[j]
				realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs),volatile=True)),dim=1,index=Variable(torch.LongTensor(batch_act)).unsqueeze(1))
				loss = self.loss(y,realQ)
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()
				obs = next_obs
				episode_reward += reward
				if done:
					print ('|Reward: {:d}| Episode: {:d}'.format(int(episode_reward),ep))
					break


	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		pass

	def burn_in_memory():
		# Initialize your replay memory with a burn_in number of episodes / transitions. 

		pass

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	agent = DQN_Agent(environment_name,args.render)
	agent.train()
	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	#gpu_ops = tf.GPUOptions(allow_growth=True)
	#config = tf.ConfigProto(gpu_options=gpu_ops)
	#sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	#keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)

