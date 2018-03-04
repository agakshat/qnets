#!/usr/bin/env python3
import torch, numpy as np, gym, sys, copy, argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque
import random
import pdb
import torch.backends.cudnn as cudnn

class LinearQNetwork(nn.Module):
    def  __init__(self,obs_dim,action_dim):
        super().__init__()
        self.lin = nn.Linear(obs_dim,action_dim)

    def forward(self,obs):
        return self.lin(obs)

class DQN(nn.Module):
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

class DuelingQN(nn.Module):
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

def soft_update(target, source, tau=0.001):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau) 
    return target

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class DQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment_name, render=False, use_cuda = False, use_target = False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.env_name = environment_name
        self.env =  gym.make(self.env_name)
        self.qnet = DQN(self.env.observation_space.shape[0],self.env.action_space.n)
        self.qnet_target = DQN(self.env.observation_space.shape[0],self.env.action_space.n)
        if use_cuda:
            self.qnet.cuda()
            self.qnet_target.cuda() 
        self.buffer_size = 50000
        self.memory = Replay_Memory(memory_size=self.buffer_size)
        self.num_iter = 1e6
        self.num_episodes = 300000
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
        self.tau = 0.001
        self.use_target = use_target

    def train(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        reward_list = np.zeros(10)
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
                    if not self.use_cuda:
                        _,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0))),dim=1)
                    else:
                        _,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0).cuda())),dim=1)
                    act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                self.memory.append(obs,act,done,next_obs,reward)
                if self.memory.len()<self.batch_size:
                    continue
                batch_obs,batch_act,batch_done,batch_next_obs,batch_reward = self.memory.sample_batch(batch_size=self.batch_size)
                #if (batch_obs[0]!=obs)
                #pdb.set_trace()
                if self.use_cuda:
                    y = Variable(torch.zeros(len(batch_reward)).cuda())
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs).cuda(),volatile=True)
                else:
                    y = Variable(torch.zeros(len(batch_reward)))
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs),volatile=True)
                if self.use_target:
                    next_a = self.qnet_target(var_batch_no)
                else:
                    next_a = self.qnet(var_batch_no)
                targetQ,_ = torch.max(next_a,dim=1)
                targetQ.volatile = False
                for j in range(len(batch_obs)):
                    if batch_done[j]:
                        y[j] = batch_reward[j]
                    else:
                        y[j] = batch_reward[j] + self.gamma*targetQ[j]
                #pdb.set_trace()
                if self.use_cuda:
                    realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs).cuda())),dim=1,index=Variable(torch.LongTensor(batch_act).cuda()).unsqueeze(1))
                else:
                    realQ = torch.gather(self.qnet(Variable(torch.FloatTensor(batch_obs))),dim=1,index=Variable(torch.LongTensor(batch_act)).unsqueeze(1))
                #print("realQ: {}, y: {}".format(realQ.data[0],y.data[0]))
                #loss = self.mse_loss(realQ,y)
                loss = self.loss(realQ,y)
                #print("loss: ",loss.data[0])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                obs = next_obs
                episode_reward += reward
                if self.use_target:
                    if self.update == 'hard_update':
                        self.qnet_target = hard_update(self.qnet_target, self.qnet)
                    else:
                        self.qnet_target = soft_update(self.qnet_target, self.qnet, self.tau)
                
                if done:

                    reward_list[ep%10] = episode_reward
                    if ep%1==0:
                        print ('|Reward: {:d}| Episode: {:d}'.format(int(np.mean(reward_list)),ep))
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


    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        pass

    def burn_in_memory():
        # Initialize your replay memory with a burn_in number of episodes / transitions. 

        pass

class DuelingQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment_name, render=False, use_cuda = False, use_target = False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.env_name = environment_name
        self.env =  gym.make(self.env_name)
        self.qnet = DuelingQN(self.env.observation_space.shape[0],self.env.action_space.n)
        self.qnet_target = DuelingQN(self.env.observation_space.shape[0],self.env.action_space.n)
        if use_cuda:
            self.qnet = self.qnet.cuda()
            self.qnet_target = self.qnet_target.cuda()
        self.memory = Replay_Memory(memory_size=100000)
        self.num_iter = 1e6
        self.num_episodes = 300000
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
        self.tau = 0.001
        self.update = 'soft_update'
        self.use_target = use_target


    def train(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        reward_list = np.zeros(10)
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
                    if self.use_cuda:
                        adv,val = self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0).cuda()))
                    else:
                        adv,val = self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0)))
                    #pdb.set_trace()
                    q = val + (adv - torch.sum(adv,dim=1)/self.env.action_space.n)
                    _,act = torch.max(q,dim=1)
                    act = act.data[0]
                next_obs,reward,done,_ = self.env.step(act)
                self.memory.append(obs,act,done,next_obs,reward)
                if self.memory.len()<self.batch_size:
                    continue
                batch_obs,batch_act,batch_done,batch_next_obs,batch_reward = self.memory.sample_batch(batch_size=self.batch_size)
                #if (batch_obs[0]!=obs)
                #pdb.set_trace()
                #y = Variable(torch.zeros(len(batch_reward)))
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
                next_a = batch_next_val + batch_next_adv - (torch.sum(batch_next_adv,dim=1)/self.env.action_space.n).unsqueeze(1)
                targetQ,_ = torch.max(next_a,dim=1)
                #pdb.set_trace()
                targetQ.volatile = False
                for j in range(len(batch_obs)):
                    if batch_done[j]:
                        y[j] = batch_reward[j]
                    else:
                        y[j] = batch_reward[j] + self.gamma*targetQ[j]
                #pdb.set_trace()
                if self.use_cuda:
                    batch_adv,batch_val = self.qnet(Variable(torch.FloatTensor(batch_obs).cuda()))
                else:
                    batch_adv,batch_val = self.qnet(Variable(torch.FloatTensor(batch_obs)))
                a = batch_val + batch_adv - (torch.sum(batch_adv,dim=1)/self.env.action_space.n).unsqueeze(1)
                if self.use_cuda:
                    realQ = torch.gather(a,dim=1,index=Variable(torch.LongTensor(batch_act).cuda()).unsqueeze(1))
                else:
                    realQ = torch.gather(a,dim=1,index=Variable(torch.LongTensor(batch_act)).unsqueeze(1))
                #print("realQ: {}, y: {}".format(realQ.data[0],y.data[0]))
                #loss = self.mse_loss(realQ,y)
                loss = self.loss(realQ,y)
                #print("loss: ",loss.data[0])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                obs = next_obs
                episode_reward += reward
                if self.use_target:
                    if self.update == 'hard_update':
                        self.qnet_target = hard_update(self.qnet_target, self.qnet)
                    else:
                        self.qnet_target = soft_update(self.qnet_target, self.qnet, self.tau)
                    
                if done:
                    reward_list[ep%10] = episode_reward
                    if ep%1==0:
                        print ('|Reward: {:d}| Episode: {:d}'.format(int(np.mean(reward_list)),ep))
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
    parser.add_argument('--dueling',dest='dueling',type=int,default=0)
    parser.add_argument('--no-cuda',action='store_true',default=False)
    parser.add_argument('--target',dest='target',type=int,default=0)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.dueling:
        agent = DuelingQN_Agent(environment_name,args.render,args.cuda,args.target)
    else:
        agent = DQN_Agent(environment_name,args.render,args.cuda,args.target)

    agent.train()

if __name__ == '__main__':
    main(sys.argv)

