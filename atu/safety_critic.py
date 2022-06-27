import argparse
import enum
import os
import copy
import random
import time
from distutils.util import strtobool
from tkinter.tix import Tree

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cost_buffer import CostReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from scipy.stats import norm
import atu
from atu.wrappers import RecordEpisodeStatisticsWithCost, SauteWrapper


def make_env(env_id, seed, idx, capture_video, run_name, saute, eval=False, args=None):
    def thunk():
        if "Pendulum" in env_id:
            env = gym.make(
                env_id, matlab_controller=controller, done_if_unsafe=args.done_if_unsafe
            )
        elif "Hallway" in env_id:
            env = gym.make(
                env_id,
                use_reach_avoid=args.ra,
                done_if_unsafe=args.done_if_unsafe,
                use_disturbances=args.dist,
                penalize_unsafe=args.basic,
            )
        else:
            env = gym.make(env_id)
        env = RecordEpisodeStatisticsWithCost(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: obs / env.world_boundary
        )
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        if args.scale_reward:
            env = gym.wrappers.TransformReward(env, lambda r: r / env.min_reward)

        if saute:
            env = SauteWrapper(
                env,
                unsafe_reward=args.saute_unsafe_reward,
                safety_budget=args.saute_safety_budget,
                saute_discount_factor=args.saute_discount_factor,
            )
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x))


class QNetwork(nn.Module):
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


def learn_safety_critic(args, run_name, writer, load_actor=None, load_critic=None):
    if load_actor:
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    args.seed,
                    0,
                    False,
                    run_name,
                    args.saute,
                    eval=False,
                    args=args,
                )
            ]
        )
        actor = Actor(envs)
        qf1 = QNetwork(envs)
        actor.load_state_dict(torch.load(load_actor))
        qf1.load_state_dict(torch.load(load_critic))
        return actor, qf1

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed,
                0,
                False,
                run_name,
                args.saute,
                eval=False,
                args=args,
            )
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    max_action = float(envs.single_action_space.high[0])
    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps_critic):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = actor(torch.Tensor(obs).to(device))
            actions = np.array(
                [
                    (
                        actions.tolist()[0]
                        + np.random.normal(
                            0,
                            max_action * args.exploration_noise,
                            size=envs.single_action_space.shape[0],
                        )
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)
        costs = np.array([info.get("cost", 0) for info in infos])

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episodic_cost={info['episode']['c']}"
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, costs, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = (target_actor(data.next_observations)).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma_risk * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_critic_loss", qf1_loss.item(), global_step
                )
                writer.add_scalar(
                    "losses/actor_critic_loss", actor_loss.item(), global_step
                )
                writer.add_scalar(
                    "losses/qf1_critic_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    # envs.close()

    return actor, qf1
