# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from stable_baselines3.common.buffers import ReplayBuffer
import atu
from atu.cost_buffer import CostReplayBuffer
from atu.wrappers import RecordEpisodeStatisticsWithCost

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--group-name", type=str,
        help="group name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="atu",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Safe-DubinsHallway-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=125_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=100_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--risk-threshold", type=float, default=0.5,
            help="Upperbound of Safety Constraint")
    parser.add_argument("--lambda-multiplier", type=float, default=100,
            help="Lagrange Multiplier For Constraint Violations")

    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, done_if_unsafe=True, use_disturbances=True)
        env = RecordEpisodeStatisticsWithCost(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.action_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (env.action_space.high + env.action_space.low) / 2.0
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
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
        return super().to(device)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=args.group_name,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    def soft_update(model, model_target):
        for param, target_param in zip(model.parameters(), model_target.parameters()):
            target_param.data.copy_(
                args.tau * param.data + (1 - args.tau) * target_param.data
            )

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )

    qf_safe_1 = SoftQNetwork(envs).to(device)
    qf_safe_2 = SoftQNetwork(envs).to(device)
    qf_safe_1_target = SoftQNetwork(envs).to(device)
    qf_safe_2_target = SoftQNetwork(envs).to(device)
    qf_safe_1_target.load_state_dict(qf_safe_1.state_dict())
    qf_safe_2_target.load_state_dict(qf_safe_2.state_dict())
    q_safe_optimizer = optim.Adam(
        list(qf_safe_1.parameters()) + list(qf_safe_2.parameters()), lr=args.q_lr
    )

    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    log_lambda = torch.tensor(
        np.log(args.lambda_multiplier, dtype=np.float32),
        requires_grad=True,
        device=device,
    )
    lambda_multiplier = log_lambda.exp().item()
    lambda_optimizer = optim.Adam(
        [log_lambda], lr=args.q_lr * 0.1
    )  # slow down lag multiplier

    envs.single_observation_space.dtype = np.float32
    rb = CostReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)
        costs = np.array([info.get("cost", 0) for info in infos])

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                writer.add_scalar(
                    "charts/percent_reach_goal", info["episode"]["prg"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, costs, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

                
                if args.RCPO:
                    # qsafe = torch.max(safety_critic(data.observation, data.action))
                    next_q_value = next_q_value - lambda_RCPO * qsafe



                qf_safe_1_next_target = qf_safe_1_target(
                    data.next_observations, next_state_actions
                )
                qf_safe_2_next_target = qf_safe_2_target(
                    data.next_observations, next_state_actions
                )
                # Be optimisitic for costs
                max_qf_safe_next_target = torch.max(
                    qf_safe_1_next_target, qf_safe_2_next_target
                )

                next_q_safe_value = data.costs.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (max_qf_safe_next_target).view(
                    -1
                )  # TODO create new arg for qf_safe

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            qf_safe_1_a_values = qf_safe_1(data.observations, data.actions).view(-1)
            qf_safe_2_a_values = qf_safe_2(data.observations, data.actions).view(-1)
            qf_safe_1_loss = F.mse_loss(qf_safe_1_a_values, next_q_safe_value)
            qf_safe_2_loss = F.mse_loss(qf_safe_2_a_values, next_q_safe_value)
            qf_safe_loss = qf_safe_1_loss + qf_safe_2_loss
            q_safe_optimizer.zero_grad()
            qf_safe_loss.backward()
            q_safe_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)

                    qf_safe_1_pi = qf_safe_1(data.observations, pi)
                    qf_safe_2_pi = qf_safe_2(data.observations, pi)
                    max_qf_safe_pi = torch.max(qf_safe_1_pi, qf_safe_2_pi)

                    writer.add_scalar(
                        "debug/max_qf-minus-risk",
                        (max_qf_safe_pi - args.risk_threshold).mean().item(),
                        global_step,
                    )

                    actor_loss = (
                        (alpha * log_pi)
                        + (lambda_multiplier * (max_qf_safe_pi - args.risk_threshold))
                        - min_qf_pi
                    ).mean()

                    # Objective: J(pi) s.t. Q_risk(s_t, a_t) <= epi_risk
                    # lagrangian relaxation: max_\lambda min_\pi -Q(s_t, a_t) + \lambda(Q_risk(s_t, a_t) - epi_risk)
                    # if Q_risk < epi_risk then Q_risk - epi_risk > 0 so lambda should be 0
                    # else if Q_risk > epi_risk then Q_risk - epi_risk > 0 so make lambda really big

                    # to adjust lambda do gradient decent on:
                    # max_\lambda:  lambda (Q_risk(s_t, a_t) - epi_risk) <=> min_\lambda  - lambda(Q_risk - epi_risk) = lambda(epi_risk - Q_risk)
                    # log_lambda  Q_risk -

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                    with torch.no_grad():
                        pi, _, _ = actor.get_action(data.observations)
                        qf_safe_1_pi = qf_safe_1(data.observations, pi)
                        qf_safe_2_pi = qf_safe_2(data.observations, pi)
                        max_qf_safe_pi = torch.max(qf_safe_1_pi, qf_safe_2_pi)
                    lambda_multiplier_loss = (
                        log_lambda * (args.risk_threshold - max_qf_safe_pi).mean()
                    )

                    lambda_optimizer.zero_grad()
                    lambda_multiplier_loss.backward()
                    lambda_optimizer.step()
                    lambda_multiplier = log_lambda.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                soft_update(qf1, qf1_target)
                soft_update(qf2, qf2_target)
                soft_update(qf_safe_1, qf_safe_1_target)
                soft_update(qf_safe_2, qf_safe_2_target)

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
                writer.add_scalar("losses/lambda", lambda_multiplier, global_step)
                writer.add_scalar(
                    "losses/lambda_loss", lambda_multiplier_loss.item(), global_step
                )

    envs.close()
    writer.close()
