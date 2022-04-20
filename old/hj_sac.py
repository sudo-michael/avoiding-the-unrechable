# https://github.com/pranz24/pytorch-soft-actor-critic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, RecordVideo
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

from operator import itemgetter

import gym_dubins

from safe_controller import SafeController


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC with 2 Q functions, Online updates"
    )
    # Common arguments
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--gym-id",
        type=str,
        default="dubins3d-v0",
        help="the id of the gym environment",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=7e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument("--seed", type=int, default=2, help="seed of the experiment")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=0,
        help="the maximum length of each episode",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=4000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will not be enabled by default",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)",
    )
    parser.add_argument(
        "--autotune",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="automatic tuning of the entropy coefficient.",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--buffer-size", type=int, default=1000000, help="the replay memory buffer size"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=1,  # Denis Yarats' implementation delays this by 2.
        help="the timesteps it takes to update the target network",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,  # Worked better in my experiments, still have to do ablation on this. Please remind me
        help="the batch size of sample from the reply memory",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="target smoothing coefficient (default: 0.005)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Entropy regularization coefficient."
    )
    parser.add_argument(
        "--learning-starts", type=int, default=5e3, help="timestep to start learning"
    )

    parser.add_argument(
        "--use-hj",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="use hj controller",
    )

    parser.add_argument(
        "--use-hirl",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="use hirl replay buffer",
    )

    parser.add_argument(
        "--hirl-penalty",
        type=int,
        default=100,
        nargs="?",
        const=True,
        help="hirl penalty for unsafe action, will be negated",
    )

    parser.add_argument(
        "--eval",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="eval model",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="path to checkpoint")

    # Additional hyper parameters for tweaks
    ## Separating the learning rate of the policy and value commonly seen: (Original implementation, Denis Yarats)
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=3e-4,
        help="the learning rate of the policy network optimizer",
    )
    parser.add_argument(
        "--q-lr",
        type=float,
        default=1e-3,
        help="the learning rate of the Q network network optimizer",
    )
    parser.add_argument(
        "--policy-frequency",
        type=int,
        default=1,
        help="delays the update of the actor, as per the TD3 paper.",
    )
    # NN Parameterization
    parser.add_argument(
        "--weights-init",
        default="xavier",
        const="xavier",
        nargs="?",
        choices=["xavier", "orthogonal", "uniform"],
        help="weight initialization scheme for the neural networks.",
    )
    parser.add_argument(
        "--bias-init",
        default="zeros",
        const="xavier",
        nargs="?",
        choices=["zeros", "uniform"],
        help="weight initialization scheme for the neural networks.",
    )

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)


# TRY NOT TO MODIFY: seeding
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
input_shape = env.observation_space.shape[0]
output_shape = env.action_space.shape[0]
# respect the default timelimit
assert isinstance(env.action_space, Box), "only continuous action space is supported"
if args.capture_video:
    env = RecordVideo(env, f"videos/{experiment_name}")


collide_counter = 0
reach_goal_counter = 0
oob_counter = 0
use_hj_counter = 0


# ALGO LOGIC: initialize agent here:
LOG_STD_MAX = 2
LOG_STD_MIN = -5


def layer_init(layer, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        if args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if args.bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)


class Policy(nn.Module):
    def __init__(self, input_shape, output_shape, env):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(
            input_shape, 256
        )  # Better result with slightly wider networks.
        self.fc2 = nn.Linear(256, 128)
        self.mean = nn.Linear(128, output_shape)
        self.logstd = nn.Linear(128, output_shape)
        # action rescaling
        self.action_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (env.action_space.high + env.action_space.low) / 2.0
        )
        self.apply(layer_init)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, device):
        mean, log_std = self.forward(x, device)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Policy, self).to(device)


class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, layer_init):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape + output_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.apply(layer_init)

    def forward(self, x, a, device):
        x = torch.Tensor(x).to(device)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer:
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

        return (
            np.array(s_lst),
            np.array(a_lst),
            np.array(r_lst),
            np.array(s_prime_lst),
            np.array(done_mask_lst),
        )


def save_checkpoint(exp_name, global_step, pg, qf1, qf2, ckpt_path=None):
    if not os.path.exists(f"checkpoints/{exp_name}/"):
        os.makedirs(f"checkpoints/{exp_name}/")
    if ckpt_path is None:
        ckpt_path = "checkpoints/{}/sac_checkpoint_{}".format(exp_name, global_step)
    print("Saving models to {}".format(ckpt_path))
    torch.save(
        {
            "pg_state_dict": pg.state_dict(),
            "qf1_state_dict": qf1.state_dict(),
            "qf2_state_dict": qf2.state_dict(),
        },
        ckpt_path,
    )


def load_checkpoint(ckpt_path, pg, qf1, qf2, evaluate=False):
    print("Loading checkpoint from {}".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    pg.load_state_dict(checkpoint["pg_state_dict"])
    qf1.load_state_dict(checkpoint["qf1_state_dict"])
    qf2.load_state_dict(checkpoint["qf2_state_dict"])

    if evaluate:
        pg.eval()


# USE_DEEPREACH = True
# if USE_DEEPREACH:
#     import sys
#     from torch.utils.data import DataLoader
#     import dataio, utils, training, loss_functions, modules, diff_operators
#     activation = 'sine'
#     model = modules.SingleBVPNet(in_features=4, out_features=1, type=activation, mode='mlp',
#                                 final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
#     checkpoint = torch.load(ckpt_path)
#     try:
#         model_weights = checkpoint['model']
#     except:
#         model_weights = checkpoint
#     model.load_state_dict(model_weights)
#     model.eval()
#     model.cuda()


# exit()

rb = ReplayBuffer(args.buffer_size)
unsafe_rb = ReplayBuffer(args.buffer_size)

pg = Policy(input_shape, output_shape, env).to(device)
qf1 = SoftQNetwork(input_shape, output_shape, layer_init).to(device)
qf2 = SoftQNetwork(input_shape, output_shape, layer_init).to(device)
qf1_target = SoftQNetwork(input_shape, output_shape, layer_init).to(device)
qf2_target = SoftQNetwork(input_shape, output_shape, layer_init).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
values_optimizer = optim.Adam(
    list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
)
policy_optimizer = optim.Adam(list(pg.parameters()), lr=args.policy_lr)
loss_fn = nn.MSELoss()

# Automatic entropy tuning
if args.autotune:
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha


def rollout(episodes):
    print("rollout")
    obs, done = env.reset(), False
    episode_reward, episode_length = 0.0, 0
    episode_count = 0

    for global_step in range(1, args.total_timesteps + 1):
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        env.render()
        obs = np.array(next_obs)

        if done:
            # Reseting what need to be
            obs, done = env.reset(), False
            print(f"{episode_reward=}")
            episode_reward, episode_length = 0.0, 0

            if info["collide_with_obs"]:
                collide_counter += 1
            writer.add_scalar(
                "eval/safety/collide_counter", collide_counter, global_step
            )

            if info["reach_goal"]:
                reach_goal_counter += 1
            writer.add_scalar("eval/sanity/goal", reach_goal_counter, global_step)

            if info["out_of_bounds"]:
                oob_counter += 1
            writer.add_scalar("eval/sanity/oob", oob_counter, global_step)

            episode_count += 1
            if episode_count == episodes:
                break

    writer.close()
    env.close()
    exit()


if args.checkpoint:
    load_checkpoint(args.checkpoint, pg, qf1, qf2, args.eval)
    rollout(1)

# TRY NOT TO MODIFY: start the game
global_episode = 0
obs, done = env.reset(), False
episode_reward, episode_length = 0.0, 0

safe_controller = SafeController()
for global_step in range(1, args.total_timesteps + 1):
    # ALGO LOGIC: put action logic here
    if args.use_hirl:
        unsafe_action = None

    if args.use_hj:
        action, _, _ = pg.get_action(np.array([obs]), device)
        action = action.tolist()[0]

        is_safe, safe_action = safe_controller.safety_layer(obs, action)

        if is_safe:
            if global_step < args.learning_starts:
                action = env.action_space.sample()
            else:
                action = safe_action
        else:
            if args.use_hirl:
                unsafe_action = action

            use_hj_counter += 1
            action = safe_action
    else:
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action, _, _ = pg.get_action(np.array([obs]), device)
            action = action.tolist()[0]

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, info = env.step(action)
    rb.put((obs, action, reward, next_obs, done))

    if args.use_hirl and unsafe_action:
        unsafe_rb.put((obs, unsafe_action, -args.hirl_penalty, next_obs, done))

    episode_reward += reward
    episode_length += 1
    obs = np.array(next_obs)

    # ALGO LOGIC: training.
    if (
        len(rb.buffer) > args.learning_starts
    ):  # starts update as soon as there is enough data.
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = pg.get_action(
                s_next_obses, device
            )
            qf1_next_target = qf1_target.forward(
                s_next_obses, next_state_actions, device
            )
            qf2_next_target = qf2_target.forward(
                s_next_obses, next_state_actions, device
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            )
            next_q_value = torch.Tensor(s_rewards).to(device) + (
                1 - torch.Tensor(s_dones).to(device)
            ) * args.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = qf1.forward(
            s_obs, torch.Tensor(s_actions).to(device), device
        ).view(-1)
        qf2_a_values = qf2.forward(
            s_obs, torch.Tensor(s_actions).to(device), device
        ).view(-1)
        qf1_loss = loss_fn(qf1_a_values, next_q_value)
        qf2_loss = loss_fn(qf2_a_values, next_q_value)
        qf_loss = (qf1_loss + qf2_loss) / 2

        values_optimizer.zero_grad()
        qf_loss.backward()
        values_optimizer.step()

        if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                args.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = pg.get_action(s_obs, device)
                qf1_pi = qf1.forward(s_obs, pi, device)
                qf2_pi = qf2.forward(s_obs, pi, device)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = pg.get_action(s_obs, device)
                    alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

    if len(unsafe_rb.buffer):  # starts update as soon as there is enough data.
        r_min = min(rb.buffer, key=itemgetter(2))[2]
        r_max = max(rb.buffer, key=itemgetter(2))[2]
        C = (r_max - r_min) / (args.gamma * 10) - r_max
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = unsafe_rb.sample(min(100, len(unsafe_rb.buffer)))

        s_rewards = np.ones_like(s_rewards) * C

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = pg.get_action(
                s_next_obses, device
            )
            qf1_next_target = qf1_target.forward(
                s_next_obses, next_state_actions, device
            )
            qf2_next_target = qf2_target.forward(
                s_next_obses, next_state_actions, device
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            )
            next_q_value = torch.Tensor(s_rewards).to(device) + (
                1 - torch.Tensor(s_dones).to(device)
            ) * args.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = qf1.forward(
            s_obs, torch.Tensor(s_actions).to(device), device
        ).view(-1)
        qf2_a_values = qf2.forward(
            s_obs, torch.Tensor(s_actions).to(device), device
        ).view(-1)
        qf1_loss = loss_fn(qf1_a_values, next_q_value)
        qf2_loss = loss_fn(qf2_a_values, next_q_value)
        qf_loss = (qf1_loss + qf2_loss) / 2

        values_optimizer.zero_grad()
        qf_loss.backward()
        values_optimizer.step()

        if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(
                args.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = pg.get_action(s_obs, device)
                qf1_pi = qf1.forward(s_obs, pi, device)
                qf2_pi = qf2.forward(s_obs, pi, device)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = pg.get_action(s_obs, device)
                    alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )
    if len(rb.buffer) > args.learning_starts and global_step % 100 == 0:
        writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
        writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
        writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/alpha", alpha, global_step)
        if args.autotune:
            writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if done:
        global_episode += 1  # Outside the loop already means the epsiode is done
        writer.add_scalar("charts/episodic_return", episode_reward, global_step)
        writer.add_scalar("charts/episode_length", episode_length, global_step)
        # Terminal verbosity
        if global_episode % 10 == 0:
            print(f"global_step={global_step}, episode_reward={episode_reward}")

        # Reseting what need to be
        obs, done = env.reset(), False
        episode_reward, episode_length = 0.0, 0

        writer.add_scalar("safety/use_hj", use_hj_counter, global_step)
        if info["collide_with_obs"]:
            collide_counter += 1
        writer.add_scalar("safety/collide_counter", collide_counter, global_step)

        if info["reach_goal"]:
            reach_goal_counter += 1
        writer.add_scalar("sanity/goal", reach_goal_counter, global_step)

        if info["out_of_bounds"]:
            oob_counter += 1
        writer.add_scalar("sanity/oob", oob_counter, global_step)

        writer.add_scalar("unsafe_buffer/len", len(unsafe_rb.buffer), global_step)


save_checkpoint(experiment_name, global_step, pg, qf1, qf2)
writer.close()
env.close()


# def imaginary_rollouts(policy, rb, device, safe_controller, init_state, rollouts=5, rollout_length=5):
#     for _ in range(rollouts):
#         obs = np.copy(init_state)
#         for _ in range(rollout_length):
#             with torch.no_grad():
