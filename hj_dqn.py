# Code from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

import gym_dubins


import numpy as np
from Grid.GridProcessing import Grid
from dynamics.DubinsCar import *
from spatialDerivatives.first_order_generic import spa_deriv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=20000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=10000,
                         help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.8,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=10000,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=1,
                        help="the frequency of training")
    parser.add_argument('--save_model', type=bool, default=False, help="save model after training")
    parser.add_argument('--save_freq', type=int, default=10_000, help="save every global steps")
    parser.add_argument('--eval', type=bool, default=False, help="path to saved model", required=False)
    parser.add_argument('--eval_path', type=str, default=False, help="path to saved model", required=False)
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b

class ProcessObsInputEnv(gym.ObservationWrapper):
    """
    This wrapper handles inputs from `Discrete` and `Box` observation space.
    If the `env.observation_space` is of `Discrete` type, 
    it returns the one-hot encoding of the state
    """
    def __init__(self, env):
        super().__init__(env)
        self.n = None
        if isinstance(self.env.observation_space, Discrete):
            self.n = self.env.observation_space.n
            self.observation_space = Box(0, 1, (self.n,))

    def observation(self, obs):
        if self.n:
            return one_hot(np.array(obs), self.n)
        return obs

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}_eval{args.eval}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.track:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")


# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
# env = ProcessObsInputEnv(gym.make(args.gym_id))
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
# respect the default timelimit
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')


dubins_car = DubinsCar(uMode='max', dMode='min')
V = np.load('V_r1.15_grid100.npy')
g = Grid(np.array([-4.0, -4.0, -np.pi]), np.array([4.0, 4.0, np.pi]), 3, np.array(V.shape), [2])

def opt_ctrl(state):
    spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
    opt_w = dubins_car.wMax
    if spa_derivatives[2] > 0:
        if dubins_car.uMode == "min":
            opt_w = -opt_w
    elif spa_derivatives[2] < 0:
        if dubins_car.uMode == "max":
            opt_w = -opt_w
    return opt_w

def is_safe(state):
    if g.get_value(V, state) < 0.26:
        ctrl = opt_ctrl(state)
        if ctrl == dubins_car.wMax:
            action = 0
        else:
            action = 1
        return False, action
    return True, None

collide_counter = 0
reach_goal_counter = 0
oob_counter = 0

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        # self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, env.action_space.n)
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, env.action_space.n)


    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_checkpoint(self, exp_name, global_step, ckpt_path=None):
        if not os.path.exists(f'checkpoints/{exp_name}/'):
            os.makedirs(f'checkpoints/{exp_name}/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/{}/dqn_checkpoint_{}.pth".format(exp_name, global_step)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'q_network_state_dict': self.state_dict()}, ckpt_path)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        assert ckpt_path is not None
        checkpoint = torch.load(ckpt_path)
        self.load_state_dict(checkpoint['q_network_state_dict'])

        if evaluate:
            self.eval()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

rb = ReplayBuffer(args.buffer_size)
q_network = QNetwork(env).to(device)
target_network = QNetwork(env).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)

# TRY NOT TO MODIFY: start the game
obs = env.reset()
episode_reward = 0
for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
    obs_is_safe, safe_action = is_safe(obs)
    if obs_is_safe:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            logits = q_network.forward(obs.reshape((1,)+obs.shape), device)
            action = torch.argmax(logits, dim=1).tolist()[0]
    else:
        action = safe_action

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, info = env.step(action)
    episode_reward += reward

    # ALGO LOGIC: training.
    rb.put((obs, action, reward, next_obs, done))
    if global_step > args.learning_starts and global_step % args.train_frequency == 0 and not args.eval:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            target_max = torch.max(target_network.forward(s_next_obses, device), dim=1)[0]
            td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
        old_val = q_network.forward(s_obs, device).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
        loss = loss_fn(td_target, old_val)

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)

        if global_step % args.save_freq == 0 and args.save_model:
            q_network.save_checkpoint(experiment_name, global_step)

        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook 
    obs = next_obs

    if done:
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        print(f"global_step={global_step}, episode_reward={episode_reward}")
        writer.add_scalar("charts/episodic_return", episode_reward, global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)
        obs, episode_reward = env.reset(), 0

        if info['collide_with_obs']:
            collide_counter += 1
        writer.add_scalar("safety/collide_counter", collide_counter, global_step)

        if info['reach_goal']:
            print('reach goal')
            reach_goal_counter += 1
        writer.add_scalar("sanity/goal", reach_goal_counter, global_step)

        if info['out_of_bounds']:
            oob_counter += 1
        writer.add_scalar("sanity/oob", oob_counter, global_step)





q_network.save_checkpoint(experiment_name, 'final')
env.close()
writer.close()
