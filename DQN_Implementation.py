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
        self.fc1 = nn.Linear(obs_dim,30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30 ,action_dim)

    def forward(self,obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingQN(nn.Module):
    def  __init__(self,obs_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim,30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30,30)
        self.adv = nn.Linear(30,action_dim)
        self.val = nn.Linear(30,1)

    def forward(self,obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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

class DQN_Agent():

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment_name, render=False, use_cuda = False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.env_name = environment_name
        self.env =  gym.make(self.env_name)
        self.qnet = DQN(self.env.observation_space.shape[0],self.env.action_space.n)
        if use_cuda:
            self.qnet = nn.DataParallel(self.qnet)
            self.qnet = self.qnet.cuda()
        self.memory = Replay_Memory(memory_size=100000)
        self.eps = 0.9
        self.num_iter = 1e6
        self.num_episodes = 300000
        self.max_ep_length = 1000
        self.render = render
        self.batch_size = 256
        if self.env_name == 'MountainCar-v0':
            self.gamma = 1.0
        else:
            self.gamma = 0.99
        self.lr = 1e-3
        self.buffer_size = 50000
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.qnet.parameters(),lr=self.lr)
        self.use_cuda = use_cuda

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
                    _,act = torch.max(self.qnet(Variable(torch.from_numpy(obs).float().unsqueeze(0))),dim=1)
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
                
                if done:
                    reward_list[ep%10] = episode_reward
                    if ep%1==0:
                        print ('|Reward: {:d}| Episode: {:d}'.format(int(np.mean(reward_list)),ep))
                    if ep%10==0:
                        torch.save(self.qnet.state_dict,'results/'+self.env_name+'.dqn.pt')
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
    
    def __init__(self, environment_name, render=False, use_cuda = False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.env_name = environment_name
        self.env =  gym.make(self.env_name)
        self.qnet = DuelingQN(self.env.observation_space.shape[0],self.env.action_space.n)
        if use_cuda:
            self.qnet = DataParallel(self.qnet)
            self.qnet = self.qnet.cuda()
        self.memory = Replay_Memory(memory_size=100000)
        self.eps = 0.5
        self.num_iter = 1e6
        self.num_episodes = 300000
        self.max_ep_length = 1000
        self.render = render
        self.batch_size = 256
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
                y = Variable(torch.zeros(len(batch_reward)))
                if self.use_cuda:
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs).cuda(),volatile=True)
                else:
                    var_batch_no = Variable(torch.FloatTensor(batch_next_obs),volatile=True)
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
                
                if done:
                    reward_list[ep%10] = episode_reward
                    if ep%1==0:
                        print ('|Reward: {:d}| Episode: {:d}'.format(int(np.mean(reward_list)),ep))
                    if ep%10==0:
                        torch.save(self.qnet.state_dict,'results/'+self.env_name+'.duelingqn.pt')
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
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        #Use CUDNN
        cudnn.benchmark = True

    if args.dueling:
        agent = DuelingQN_Agent(environment_name,args.render,args.cuda)
    else:
        agent = DQN_Agent(environment_name,args.render,args.cuda)

    #if args.cuda:
        #agent = nn.DataParallel(agent)
    #    agent = agent.cuda()
        
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
