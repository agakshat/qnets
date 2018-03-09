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

class LinearQNetwork(nn.Module): # network definition
    def  __init__(self,obs_dim,action_dim):
        super().__init__()
        self.lin = nn.Linear(obs_dim,action_dim)

    def forward(self,obs):
        return self.lin(obs)

class DQN(nn.Module): # network definition
    def  __init__(self,obs_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64 ,action_dim)
        nn.init.xavier_normal(self.fc1.weight.data)
        nn.init.xavier_normal(self.fc2.weight.data)
        nn.init.uniform(self.fc3.weight.data, a = -2e-3, b = 2e-3)

    def forward(self,obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingQN(nn.Module): # network definition
    def  __init__(self,obs_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim,64)
        self.fc2 = nn.Linear(64,32)
        self.adv = nn.Linear(32,action_dim)
        self.val = nn.Linear(32,1)
        nn.init.xavier_normal(self.fc1.weight.data)
        nn.init.xavier_normal(self.fc2.weight.data)
        nn.init.uniform(self.adv.weight.data, a = -2e-3, b = 2e-3)
        nn.init.uniform(self.val.weight.data, a = -2e-3, b = 2e-3)

    def forward(self,obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        adv = self.adv(x)
        val = self.val(x)
        return adv,val

class AtariDQN(nn.Module): # network definition
    def  __init__(self,obs_dim,action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(4,16,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.fc1 = nn.Linear(2592,256)
        self.action = nn.Linear(256,action_dim)
        nn.init.orthogonal(self.conv1.weight.data)
        nn.init.orthogonal(self.conv2.weight.data)
        nn.init.xavier_normal(self.fc1.weight.data)
        nn.init.uniform(self.action.weight.data, a = -2e-3, b = 2e-3)

    def forward(self,obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(-1,2592)
        x = F.relu(self.fc1(x))
        x = self.action(x)
        return x

class Replay_Memory(): # uses namedtuple and deque collections in Python for easy implementation of memory

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

def soft_update(target, source, tau=0.001): # does soft update of target network parameters
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau) 
    return target

def hard_update(target, source): # does hard update (direct copy) of target network parameters
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class LinearQN_Agent():
    
    def __init__(self, environment_name, render=False, use_cuda = False, use_target = False):

        self.env_name = environment_name
        self.env =  gym.make(self.env_name)
        self.qnet = LinearQNetwork(self.env.observation_space.shape[0],self.env.action_space.n)
        self.qnet_target = LinearQNetwork(self.env.observation_space.shape[0],self.env.action_space.n)
        if use_cuda:
            self.qnet.cuda()
            self.qnet_target.cuda() 
        hard_update(self.qnet_target,self.qnet)
        self.buffer_size = 50000 # make this equal to 1 for no experience replay (online update)
        self.memory = Replay_Memory(memory_size=self.buffer_size)
        self.num_iter = 1e6
        self.num_episodes = 10000
        self.max_ep_length = 1000
        self.eps = 0.5
        self.render = render
        self.batch_size = 256 # make this equal to 1 for no experience replay (online update)
        if self.env_name == 'MountainCar-v0':
            self.gamma = 1.0
        else:
            self.gamma = 0.99
        self.lr = 1e-3
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.qnet.parameters(),lr=self.lr)
        self.use_cuda = use_cuda
        self.update = 'soft_update'
        self.tau = 0.001 # parameter for soft update of target network
        self.use_target = use_target # true if target network used

    def train(self):

        reward_list = np.zeros(10)
        for ep in range(self.num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            if self.eps>0.05: # linearly decaying greedy parameter epsilon
                self.eps = -0.0009*ep + 0.5  
            for iteration in range(self.max_ep_length):
                if self.render:
                    self.env.render()
                if np.random.rand()<self.eps: # doing epsilon greedy sampling during training
                    act = self.env.action_space.sample()
                else:
                    if not self.use_cuda:
                        _,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0))),dim=1)
                    else:
                        _,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0).cuda())),dim=1)
                    act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                self.memory.append(obs,act,done,next_obs,reward) # add transition to replay memory
                if self.memory.len()<self.batch_size: # allows burn in of batch size
                    continue
                batch_obs,batch_act,batch_done,batch_next_obs,batch_reward = self.memory.sample_batch(batch_size=self.batch_size) # sample from memory
                if self.use_cuda:
                    y = Variable(torch.zeros(len(batch_reward)).cuda())
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs).cuda(),volatile=True)
                else:
                    y = Variable(torch.zeros(len(batch_reward)))
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs),volatile=True)
                if self.use_target:
                    next_a = self.qnet_target(var_batch_no) # 'next_a' is a misnomer - this is the Q-value obtained by a forward pass through Q-network
                else:
                    next_a = self.qnet(var_batch_no)
                targetQ,_ = torch.max(next_a,dim=1)
                targetQ.volatile = False
                for j in range(len(batch_obs)): # to deal with episode ending case (could be optimized without a loop too)
                    if batch_done[j]:
                        y[j] = batch_reward[j]
                    else:
                        y[j] = batch_reward[j] + self.gamma*targetQ[j]
                if self.use_cuda:
                    realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs).cuda())),dim=1,index=Variable(torch.LongTensor(batch_act).cuda()).unsqueeze(1))
                else:
                    realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs))),dim=1,index=Variable(torch.LongTensor(batch_act)).unsqueeze(1))
                loss = self.loss(realQ,y) # calculate loss
                self.optim.zero_grad() # zero out the gradients of the network from previous backprops
                loss.backward() # back-propagate the loss
                self.optim.step() # take a step acc. to Adam optimizer
                obs = next_obs
                episode_reward += reward
                if self.use_target: # updates target network
                    if self.update == 'hard_update':
                        self.qnet_target = hard_update(self.qnet_target, self.qnet)
                    else:
                        self.qnet_target = soft_update(self.qnet_target, self.qnet, self.tau)
                if done: # this is all book-keeping, checkpointing and writing rewards to text file etc.
                    reward_list[ep%10] = episode_reward
                    if ep%1==0:
                        print ('|Reward: {:d}| Episode: {:d}'.format(int(episode_reward),ep))
                    if ep%10==0:
                        if self.use_target:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'.target.lqn.pt')
                            with open("results/"+self.env_name+'.linear.target.txt',"a") as f:
                                f.write('|Reward: {:d}| Episode: {:d}\n'.format(int(np.mean(reward_list)),ep))
                        else:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'.lqn.pt')
                            with open("results/"+self.env_name+'.linear.txt',"a") as f:
                                f.write('|Reward: {:d}| Episode: {:d}\n'.format(int(np.mean(reward_list)),ep))
                    if ep%200==0:
                        if self.use_target:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'_'+str(ep)+'.target.lqn.pt')
                        else:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'_'+str(ep)+'.lqn.pt')
                    break # breaks the loop and starts next episode when envt returns done


    def test(self, model_file=None, render = True):
        """
        If render is True, renders but runs only for 5 episodes
        If render is False, does not render but runs for 100 episodes and returns mean, std_dev and list of episode rewards
        Note: We did not use Gym Monitor to capture videos, instead used recordmydesktop application in Linux
        """

        self.env =  gym.make(self.env_name) # just to be safe, initialized again
        params = torch.load(model_file,map_location = lambda storage, loc: storage)() # ensures models trained on GPU can be loaded onto CPU
        self.qnet.load_state_dict(params) # load weights into the agent
        obs = self.env.reset() 
        if render:
            num = 5
            self.env.render()
            input("Press Enter to continue") # to allow to start recording, just put a break in here!
        else:
            num = 100
        reward_list = []
        for ep in range(num):
            episode_reward = 0
            while True:
                if render:
                    self.env.render()
                _,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0))),dim=1)
                act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                episode_reward += reward
                if done:
                    self.env.reset() # reset the envt when done
                    break
                obs = next_obs
            reward_list.append(episode_reward)
        self.env.close()
        reward_list = np.array(reward_list)
        print (np.mean(reward_list),np.std(reward_list),reward_list)

class DQN_Agent():

    def __init__(self, environment_name, render=False, use_cuda = False, use_target = False):

        self.env_name = environment_name
        self.env =  gym.make(self.env_name)
        self.qnet = DQN(self.env.observation_space.shape[0],self.env.action_space.n)
        self.qnet_target = DQN(self.env.observation_space.shape[0],self.env.action_space.n)
        if use_cuda:
            self.qnet.cuda()
            self.qnet_target.cuda() 
        hard_update(self.qnet_target,self.qnet)
        self.buffer_size = 50000
        self.memory = Replay_Memory(memory_size=self.buffer_size)
        self.num_iter = 1e6
        self.num_episodes = 8000
        self.max_ep_length = 1000
        self.eps = 0.9
        self.render = render
        self.batch_size = 256
        if self.env_name == 'MountainCar-v0':
            self.gamma = 1.0
        else:
            self.gamma = 0.99
        self.lr = 1e-3
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.qnet.parameters(),lr=self.lr)
        self.use_cuda = use_cuda
        self.update = 'soft_update'
        self.tau = 0.001 # parameter for soft update of target network
        self.use_target = use_target # true if target network used

    def train(self):
        reward_list = np.zeros(10)
        for ep in range(self.num_episodes):
            obs = self.env.reset()
            episode_reward = 0

            if self.eps>0.05: # linearly decaying greedy parameter epsilon
                self.eps = -0.0009*ep + 0.5  
            for iteration in range(self.max_ep_length):
                if self.render:
                    self.env.render()
                if np.random.rand()<self.eps: # doing epsilon greedy sampling during training
                    act = self.env.action_space.sample()
                else:
                    if not self.use_cuda:
                        _,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0))),dim=1)
                    else:
                        _,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0).cuda())),dim=1)
                    act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                self.memory.append(obs,act,done,next_obs,reward) # add transition to replay memory
                if self.memory.len()<self.batch_size: # allows burn in of batch size
                    continue
                batch_obs,batch_act,batch_done,batch_next_obs,batch_reward = self.memory.sample_batch(batch_size=self.batch_size)
                if self.use_cuda:
                    y = Variable(torch.zeros(len(batch_reward)).cuda())
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs).cuda(),volatile=True)
                else:
                    y = Variable(torch.zeros(len(batch_reward)))
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs),volatile=True)
                if self.use_target:
                    next_a = self.qnet_target(var_batch_no) # 'next_a' is a misnomer - this is the Q-value obtained by a forward pass through Q-network
                else:
                    next_a = self.qnet(var_batch_no)
                targetQ,_ = torch.max(next_a,dim=1)
                targetQ.volatile = False
                for j in range(len(batch_obs)): # to deal with episode ending case (could be optimized without a loop too)
                    if batch_done[j]:
                        y[j] = batch_reward[j]
                    else:
                        y[j] = batch_reward[j] + self.gamma*targetQ[j]
                if self.use_cuda:
                    realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs).cuda())),dim=1,index=Variable(torch.LongTensor(batch_act).cuda()).unsqueeze(1))
                else:
                    realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs))),dim=1,index=Variable(torch.LongTensor(batch_act)).unsqueeze(1))
                loss = self.loss(realQ,y)
                self.optim.zero_grad() # gradients of network params zeroed out
                loss.backward() # backpropagate loss
                self.optim.step() # optimizer takes a step
                obs = next_obs
                episode_reward += reward
                if self.use_target:
                    if self.update == 'hard_update':
                        self.qnet_target = hard_update(self.qnet_target, self.qnet)
                    else:
                        self.qnet_target = soft_update(self.qnet_target, self.qnet, self.tau)
                
                if done: # this is all book-keeping, checkpointing and writing rewards to text file etc.
                    reward_list[ep%10] = episode_reward
                    if ep%1==0:
                        print ('|Reward: {:d}| Episode: {:d}'.format(int(episode_reward),ep))
                    if ep%10==0:
                        if self.use_target:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'.target.dqn.pt')
                            with open("results/"+self.env_name+'.target.txt',"a") as f:
                                f.write('|Reward: {:d}| Episode: {:d}\n'.format(int(np.mean(reward_list)),ep))
                        else:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'.dqn.pt')
                            with open("results/"+self.env_name+'.txt',"a") as f:
                                f.write('|Reward: {:d}| Episode: {:d}\n'.format(int(np.mean(reward_list)),ep))
                    if ep%200==0:
                        if self.use_target:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'_'+str(ep)+'.target.dqn.pt')
                        else:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'_'+str(ep)+'.dqn.pt')
                    break


    def test(self, model_file=None, render = True):
        """
        If render is True, renders but runs only for 5 episodes
        If render is False, does not render but runs for 100 episodes and returns mean, std_dev and list of episode rewards
        Note: We did not use Gym Monitor to capture videos, instead used recordmydesktop application in Linux
        """
        self.env =  gym.make(self.env_name)
        params = torch.load(model_file,map_location = lambda storage, loc: storage)()# ensures models trained on GPU can be loaded onto CPU
        self.qnet.load_state_dict(params)
        obs = self.env.reset()
        if render:
            num = 5
            self.env.render()
            input("Press Enter to continue")
        else:
            num = 100
        reward_list = []
        for ep in range(num):
            episode_reward = 0
            while True:
                if render:
                    self.env.render()
                _,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0))),dim=1)
                act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                episode_reward += reward
                if done:
                    self.env.reset()
                    break
                obs = next_obs
            reward_list.append(episode_reward)
        self.env.close()
        reward_list = np.array(reward_list)
        print (np.mean(reward_list),np.std(reward_list),reward_list)

class DuelingQN_Agent():
    
    def __init__(self, environment_name, render=False, use_cuda = False, use_target = False):

        self.env_name = environment_name
        self.env =  gym.make(self.env_name)
        self.qnet = DuelingQN(self.env.observation_space.shape[0],self.env.action_space.n)
        self.qnet_target = DuelingQN(self.env.observation_space.shape[0],self.env.action_space.n)
        if use_cuda:
            self.qnet = self.qnet.cuda()
            self.qnet_target = self.qnet_target.cuda()
        hard_update(self.qnet_target,self.qnet)
        self.memory = Replay_Memory(memory_size=100000)
        self.num_iter = 1e6
        self.num_episodes = 8000
        self.max_ep_length = 1000
        self.render = render
        self.batch_size = 64
        self.eps = 0.9
        if self.env_name == 'MountainCar-v0':
            self.gamma = 1
        else:
            self.gamma = 0.99
        self.lr = 1e-3
        self.buffer_size = 50000
        self.loss = nn.MSELoss()
        if use_cuda:
            self.loss = self.loss.cuda()
        self.optim = torch.optim.Adam(self.qnet.parameters(),lr=self.lr)
        self.use_cuda = use_cuda
        self.tau = 0.001 # parameter for soft update of target network
        self.update = 'soft_update' 
        self.use_target = use_target # true if target network used


    def train(self):

        reward_list = np.zeros(10)
        for ep in range(self.num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            
            if self.eps>0.05: # linearly decaying greedy parameter epsilon
                self.eps = -0.0009*ep + 0.5  
            for iteration in range(self.max_ep_length):
                if self.render:
                    self.env.render()
                if np.random.rand()<self.eps: # doing epsilon greedy sampling during training
                    act = self.env.action_space.sample()
                else:
                    if self.use_cuda:
                        adv,val = self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0).cuda()))
                    else:
                        adv,val = self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0)))
                    q = val + (adv - torch.sum(adv,dim=1)/self.env.action_space.n) # calculating q from advantage and value obtained from forward pass
                    _,act = torch.max(q,dim=1)
                    act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                self.memory.append(obs,act,done,next_obs,reward) # add transition to memory
                if self.memory.len()<self.batch_size: # allow burn in 
                    continue
                batch_obs,batch_act,batch_done,batch_next_obs,batch_reward = self.memory.sample_batch(batch_size=self.batch_size)
                if self.use_cuda:
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs).cuda(),volatile=True)
                    y = Variable(torch.zeros(len(batch_reward)).cuda())
                else:
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs),volatile=True)
                    y = Variable(torch.zeros(len(batch_reward)))
                if self.use_target:
                    batch_next_adv,batch_next_val = self.qnet_target(var_batch_no)
                else:
                    batch_next_adv,batch_next_val = self.qnet(var_batch_no)
                next_a = batch_next_val + batch_next_adv - (torch.sum(batch_next_adv,dim=1)/self.env.action_space.n).unsqueeze(1) # 'next_a' is a misnomer - this is the Q-value obtained by a forward pass through Q-network
                targetQ,_ = torch.max(next_a,dim=1)
                targetQ.volatile = False
                for j in range(len(batch_obs)): # to deal with episode ending case (could be optimized without a loop too)
                    if batch_done[j]:
                        y[j] = batch_reward[j]
                    else:
                        y[j] = batch_reward[j] + self.gamma*targetQ[j]
                if self.use_cuda:
                    batch_adv,batch_val = self.qnet(Variable(torch.FloatTensor(batch_obs).cuda()))
                else:
                    batch_adv,batch_val = self.qnet(Variable(torch.FloatTensor(batch_obs)))
                a = batch_val + batch_adv - (torch.sum(batch_adv,dim=1)/self.env.action_space.n).unsqueeze(1)
                if self.use_cuda:
                    realQ = torch.gather(a,dim=1,index=Variable(torch.LongTensor(batch_act).cuda()).unsqueeze(1))
                else:
                    realQ = torch.gather(a,dim=1,index=Variable(torch.LongTensor(batch_act)).unsqueeze(1))
                loss = self.loss(realQ,y)
                self.optim.zero_grad() # gradients of network params zeroed out
                loss.backward() # backpropagate loss
                self.optim.step() # optimizer takes a step
                obs = next_obs
                episode_reward += reward
                if self.use_target:
                    if self.update == 'hard_update':
                        self.qnet_target = hard_update(self.qnet_target, self.qnet)
                    else:
                        self.qnet_target = soft_update(self.qnet_target, self.qnet, self.tau)
                    
                if done: # this is all book-keeping, checkpointing and writing rewards to text file etc.
                    reward_list[ep%10] = episode_reward
                    if ep%1==0:
                        print ('|Reward: {:d}| Episode: {:d}'.format(int(episode_reward),ep))
                    if ep%10==0:
                        if self.use_target:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'.dueling.target.dqn.pt')
                            with open("results/"+self.env_name+'.dueling.target.txt',"a") as f:
                                f.write('|Reward: {:d}| Episode: {:d}\n'.format(int(np.mean(reward_list)),ep))
                        else:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'.dueling.dqn.pt')
                            with open("results/"+self.env_name+'.dueling.txt',"a") as f:
                                f.write('|Reward: {:d}| Episode: {:d}\n'.format(int(np.mean(reward_list)),ep))
                    if ep%200==0:
                        if self.use_target:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'_'+str(ep)+'.dueling.target.dqn.pt')
                        else:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'_'+str(ep)+'.dueling.dqn.pt')
                    break


    def test(self, model_file=None, render = True):
        """
        If render is True, renders but runs only for 5 episodes
        If render is False, does not render but runs for 100 episodes and returns mean, std_dev and list of episode rewards
        Note: We did not use Gym Monitor to capture videos, instead used recordmydesktop application in Linux
        """
        self.env =  gym.make(self.env_name)
        params = torch.load(model_file,map_location = lambda storage, loc: storage)()
        self.qnet.load_state_dict(params)
        obs = self.env.reset()
        if render:
            num = 5
            self.env.render()
            input("Press Enter to continue")
        else:
            num = 100
        reward_list = []
        for ep in range(num):
            episode_reward = 0
            while True:
                if render:
                    self.env.render()
                adv,val = self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0)))
                q = val + (adv - torch.sum(adv,dim=1)/self.env.action_space.n)
                _,act = torch.max(q,dim=1)
                act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                episode_reward += reward
                if done:
                    self.env.reset()
                    break
                obs = next_obs
            reward_list.append(episode_reward)
        self.env.close()
        reward_list = np.array(reward_list)
        print (np.mean(reward_list),np.std(reward_list),reward_list)

class AtariDQN_Agent():
    
    def __init__(self, environment_name, render=False, use_cuda = False, use_target = False):

        self.env_name = environment_name
        self.env =  gym.make(self.env_name)
        self.qnet = AtariDQN(self.env.observation_space.shape[0],self.env.action_space.n)
        self.qnet_target = AtariDQN(self.env.observation_space.shape[0],self.env.action_space.n)
        if use_cuda:
            self.qnet.cuda()
            self.qnet_target.cuda()
        hard_update(self.qnet_target,self.qnet)
        self.memory = Replay_Memory(memory_size=1000000)
        self.eps = 0.5
        self.num_iter = 1e6
        self.num_episodes = 300000
        self.max_ep_length = 1000
        self.render = render
        self.batch_size = 64
        if self.env_name == 'MountainCar-v0':
            self.gamma = 1
        else:
            self.gamma = 0.99
        self.lr = 1e-4
        self.buffer_size = 50000
        self.loss = nn.MSELoss()
        if use_cuda:
            self.loss = self.loss.cuda()
        self.optim = torch.optim.Adam(self.qnet.parameters(),lr=self.lr)
        self.use_cuda = use_cuda
        self.stacked_obs = np.zeros((4,84,84)) # will store last 4 observed frames
        self.tau = 0.001
        self.use_target = use_target
        self.update = 'soft_update'

    def stackAndGray(self,obs):
        frame = cv2.cvtColor(obs,cv2.COLOR_RGB2GRAY) # RGB to gray (single channel)
        frame = cv2.resize(frame,(84,84),interpolation=cv2.INTER_AREA) # resize (to 84x84)
        frame = frame.astype(float)
        frame -= np.mean(frame,axis=0) # normalizing
        frame /= np.std(frame,axis=0)
        for j in reversed(range(3)):
            self.stacked_obs[-2-j] = self.stacked_obs[-1-j] # stacking observations
        self.stacked_obs[-1] = frame

    def train(self):

        for ep in range(self.num_episodes):
            obs = self.env.reset() # obs is 210x160x3
            self.stackAndGray(obs) #self.stacked_obs is updated with the new 84x84 frame
            episode_reward = 0
            reward_list = np.zeros(10)
            if self.eps>0.05: # linearly decaying exploration parameter, start off with more randomness here but decay it more quickly too
                self.eps = -0.0015*ep + 0.9  
            for iteration in range(self.max_ep_length):
                if self.render:
                    self.env.render()
                if np.random.rand()<self.eps:
                    act = self.env.action_space.sample()
                else:
                    if self.use_cuda:
                        _,act = torch.max(self.qnet(Variable(torch.from_numpy(self.stacked_obs).float().cuda().unsqueeze(0))),dim=1)
                    else:
                        _,act = torch.max(self.qnet(Variable(torch.from_numpy(self.stacked_obs).float().unsqueeze(0))),dim=1)
                    act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                soCopy = self.stacked_obs.copy() # to add to replay memory
                self.stackAndGray(next_obs) # update self.stacked_obs with new observation
                self.memory.append(soCopy,act,done,self.stacked_obs,reward)
                if self.memory.len()<self.batch_size: # burn in
                    continue
                batch_obs,batch_act,batch_done,batch_next_obs,batch_reward = self.memory.sample_batch(batch_size=self.batch_size)
                if self.use_cuda:
                    y = Variable(torch.zeros(len(batch_reward)).cuda())
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs).cuda(),volatile=True)
                else:   
                    y = Variable(torch.zeros(len(batch_reward)))
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs),volatile=True)
                if self.use_target:
                    next_a = self.qnet_target(var_batch_no) # 'next_a' is a misnomer
                else:
                    next_a = self.qnet(var_batch_no)
                targetQ,_ = torch.max(next_a,dim=1)
                targetQ.volatile = False
                for j in range(len(batch_obs)):
                    if batch_done[j]:
                        y[j] = batch_reward[j]
                    else:
                        y[j] = batch_reward[j] + self.gamma*targetQ[j]
                if self.use_cuda:
                    realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs).cuda())),dim=1,index=Variable(torch.LongTensor(batch_act).cuda()).unsqueeze(1))
                else:
                    realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs))),dim=1,index=Variable(torch.LongTensor(batch_act)).unsqueeze(1))
                loss = self.loss(realQ,y)
                self.optim.zero_grad()
                loss.backward() # back-propagate loss
                self.optim.step() # optimizer takes a step
                obs = next_obs
                episode_reward += reward
                if self.use_target:
                    if self.update == 'hard_update':
                        self.qnet_target = hard_update(self.qnet_target, self.qnet)
                    else:
                        self.qnet_target = soft_update(self.qnet_target, self.qnet, self.tau)
                if done: # bookkeeping, checkpointing
                    reward_list[ep%10] = episode_reward
                    if ep%1==0:
                        print ('|Reward: {:d}| Episode: {:d}'.format(int(episode_reward),ep))
                    if ep%5==0:
                        if self.use_target:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'.target.dqn.pt')
                            with open("results/"+self.env_name+'.target.txt',"a") as f:
                                f.write('|Reward: {:d}| Episode: {:d}\n'.format(int(np.mean(reward_list)),ep))
                        else:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'.dqn.pt')
                            with open("results/"+self.env_name+'.txt',"a") as f:
                                f.write('|Reward: {:d}| Episode: {:d}\n'.format(int(np.mean(reward_list)),ep))
                    if ep%200==0:
                        if self.use_target:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'_'+str(ep)+'.target.dqn.pt')
                        else:
                            torch.save(self.qnet.state_dict,'results/'+self.env_name+'_'+str(ep)+'.dqn.pt')
                    break


    def test(self, model_file=None, render = True):
        """
        If render is True, renders but runs only for 5 episodes
        If render is False, does not render but runs for 100 episodes and returns mean, std_dev and list of episode rewards
        Note: We did not use Gym Monitor to capture videos, instead used recordmydesktop application in Linux
        """
        self.env =  gym.make(self.env_name)
        params = torch.load(model_file,map_location = lambda storage, loc: storage)()
        self.qnet.load_state_dict(params)
        obs = self.env.reset()
        self.stackAndGray(obs)
        if render:
            num = 5
            self.env.render()
            input("Press Enter to continue")
        else:
            num = 100
        reward_list = []
        for ep in range(num):
            episode_reward = 0
            while True:
                if render:
                    self.env.render()
                q = self.qnet(Variable(torch.from_numpy(self.stacked_obs).float().unsqueeze(0)))
                _,act = torch.max(q,dim=1)
                act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                self.stackAndGray(next_obs)
                episode_reward += reward
                if done:
                    self.env.reset()
                    break
                obs = next_obs
            reward_list.append(episode_reward)
        self.env.close()
        reward_list = np.array(reward_list)
        print (np.mean(reward_list),np.std(reward_list),reward_list)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,default='CartPole-v0') # environment name
    parser.add_argument('--render',dest='render',type=int,default=0) # pass 1 for rendering during test time. Never renders during training
    parser.add_argument('--type',dest='type',type=int,default=1) # 0 for Linear, 1 for DQN, 2 for Dueling, 3 for Atari (Space Invaders)
    parser.add_argument('--no-cuda',action='store_true',default=False) # pass True to not use CUDA if CUDA enabled machine
    parser.add_argument('--target',dest='target',type=int,default=1) # pass 1 to use target network, 0 to not use
    parser.add_argument('--test',dest='test',type=int,default=0) # pass 1 to test, 0 (default) starts training
    parser.add_argument('--load',dest='load',type=str) # used only during test time, path to model file storing trained weights
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    args.cuda = not args.no_cuda and torch.cuda.is_available() # ensure CUDA isn't used if not available

    if args.type==0:
        agent = LinearQN_Agent(environment_name,args.render,args.cuda,args.target)
    elif args.type==1:
        agent = DQN_Agent(environment_name,args.render,args.cuda,args.target)
    elif args.type==2:
        agent = DuelingQN_Agent(environment_name,args.render,args.cuda,args.target)
    elif args.type==3:
        if environment_name=='CartPole-v0': # for convenience!
            environment_name='SpaceInvaders-v0'
        agent = AtariDQN_Agent(environment_name,args.render,args.cuda,args.target)
    
    if not args.test:
        agent.train()
    else:
        agent.test(args.load,args.render)

if __name__ == '__main__':
    main(sys.argv)

