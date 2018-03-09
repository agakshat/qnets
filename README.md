Homework Code for 10703: Deep Reinforcement Learning at CMU

Submitted by: Vasu Sharma (vasus@andrew.cmu.edu) and Akshat Agarwal (akshata@andrew.cmu.edu)

Requirements:
1. Pytorch (www.pytorch.org)
2. OpenAI Gym
3. Python 3 
4. OpenCV2 (`pip3 install python-opencv`)

Commands to run:

1. Linear Q-Networks:

Training: `python DQN_Implementation.py --env CartPole-v0 --type 0`

Testing: `python DQN_Implementation.py --env CartPole-v0 --type 0 --test 1 --load <path_to_file>`

Note: To run with no experience replay, please go into the code for LinearQ_Agent and change `self.buffer_size` and `self.batch_size` to 1.

2. DQN:

Training: `python DQN_Implementation.py --env CartPole-v0 --type 1`

Testing: `python DQN_Implementation.py --env CartPole-v0 --type 1 --test 1 --load <path_to_file>`

3. Dueling Q-Networks:

Training: `python DQN_Implementation.py --env CartPole-v0 --type 2`

Testing: `python DQN_Implementation.py --env CartPole-v0 --type 2 --test 1 --load <path_to_file>`

4. Atari DQN:

Training: `python DQN_Implementation.py --env SpaceInvaders-v0 --type 3`

Testing: `python DQN_Implementation.py --env SpaceInvaders-v0 --type 3 --test 1 --load <path_to_file>`

Additional Flags: 

The flags are parsed at the very end of the `DQN_Implementation.py` file, with comments on their usage. However, I am also mentioning here for completeness.

1. `--env` environment name (default: `CartPole-v0`)
2. `--render`  pass 1 for rendering during test time, never renders during training (default: 0)
3. `--type`  0 for Linear, 1 for DQN, 2 for Dueling, 3 for Atari (default: 1)
4. `--no-cuda` pass True to not use CUDA if CUDA enabled machine (default: False)
5. `--target`  pass 1 to use target network, 0 to not use (default: 1)
6. `--test` pass 1 to test, 0  starts training (default: 0)
7. `--load` used only during test time, path to model file storing trained weights (default: None)
