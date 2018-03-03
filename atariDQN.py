#!/usr/bin/env python3
import torch, numpy as np, gym, sys, copy, argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque
import random
import pdb
import cv2
import copy

class DQN(nn.Module):
	def  __init__(self,obs_dim,action_dim):
		super().__init__()
		self.conv1 = nn.Conv2d(4,16,kernel_size=8,stride=4)
		self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2)
		self.fc1 = nn.Linear(2592,256)
		self.action = nn.Linear(256,action_dim)

	def forward(self,obs):
		x = F.relu(self.conv1(obs))
		x = F.relu(self.conv2(x))
		x = x.view(-1,2592)
		x = F.relu(self.fc1(x))
		x = self.action(x)
		return x

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		self.Transition = namedtuple('Transition', ('state', 'action', 'done', 'next_state','reward'))
		self.memory = deque(maxlen=memory_size)

	def sample_batch(self, batch_size=32):
		random_batch = random.sample(self.memory,batch_size)
		return self.Transition(*zip(*random_batch))

	def append(self, *args):
		self.memory.append(self.Transition(*args))

	def len(self):
		return len(self.memory)

class DQN_Agent():
	
	def __init__(self, environment_name, render=False, use_cuda = False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.env_name = environment_name
		self.env =  gym.make(self.env_name)
		self.qnet = DQN(self.env.observation_space.shape[0],self.env.action_space.n)
		self.qnet_target = copy.deepcopy(self.qnet)
		self.memory = Replay_Memory(memory_size=100000)
		self.eps_start = 0.5
		self.eps_end = 0.05
		self.eps_iter = 100000
		self.eps = self.eps_start
		self.num_iter = 1e6
		self.num_episodes = 300000
		self.max_ep_length = 1000
		self.render = render
		self.batch_size = 16
		self.gamma = 0.99
		self.lr = 1e-3
		self.buffer_size = 50000
		self.loss = nn.MSELoss()
		self.optim = torch.optim.Adam(self.qnet.parameters(),lr=self.lr)
		self.use_cuda = use_cuda
		self.stacked_obs = np.zeros((4,84,84))

	def stackAndGray(self,obs):
		frame = cv2.cvtColor(obs,cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame,(84,84),interpolation=cv2.INTER_AREA)
		frame = frame.astype(float)
		frame -= np.mean(frame,axis=0)
		frame /= np.std(frame,axis=0)
		for j in reversed(range(3)):
			self.stacked_obs[-2-j] = self.stacked_obs[-1-j]
		self.stacked_obs[-1] = frame

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		for ep in range(self.num_episodes):
			obs = self.env.reset()
			self.stackAndGray(obs)
			episode_reward = 0
			reward_list = np.zeros(10)
			if self.eps>0.05:
				self.eps = -0.0015*ep + 0.9	 
			for iteration in range(self.max_ep_length):
				if self.render:
					self.env.render()
				if np.random.rand()<self.eps:
					act = self.env.action_space.sample()
				else:
					_,act = torch.max(self.qnet(Variable(torch.from_numpy(self.stacked_obs).float().unsqueeze(0))),dim=1)
					act = act.data[0]
				next_obs,reward,done,_ = self.env.step(act)
				soCopy = self.stacked_obs.copy()
				self.stackAndGray(next_obs)
				self.memory.append(soCopy,act,done,self.stacked_obs,reward)
				if self.memory.len()<self.batch_size:
					continue
				batch_obs,batch_act,batch_done,batch_next_obs,batch_reward = self.memory.sample_batch(batch_size=self.batch_size)
				y = Variable(torch.zeros(len(batch_reward)))
				var_batch_no = Variable(torch.FloatTensor(batch_next_obs),volatile=True)
				next_a = self.qnet(var_batch_no)
				targetQ,_ = torch.max(next_a,dim=1)
				targetQ.volatile = False
				for j in range(len(batch_obs)):
					if batch_done[j]:
						y[j] = batch_reward[j]
					else:
						y[j] = batch_reward[j] + self.gamma*targetQ[j]
				realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs))),dim=1,index=Variable(torch.LongTensor(batch_act)).unsqueeze(1))
				loss = self.loss(realQ,y)
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()
				obs = next_obs
				episode_reward += reward
				reward_list[ep%10] = episode_reward
				if done:
					if ep%10==0:
						print ('|Reward: {:d}| Episode: {:d}'.format(int(episode_reward),ep))
					if ep%500==0:
						torch.save(self.qnet.state_dict,'results/'+self.env_name+'.dqn.pt')
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
	parser.add_argument('--env',dest='env',type=str,default='SpaceInvaders-v0')
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument('--dueling',dest='dueling',type=int,default=0)
	parser.add_argument('--no-cuda',action='store_true',default=False)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env
	args.cuda = not args.no_cuda and torch.cuda.is_available()

	agent = DQN_Agent(environment_name,args.render,args.cuda)

	agent.train()

if __name__ == '__main__':
	main(sys.argv)

