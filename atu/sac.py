# from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
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
from torch.utils.tensorboard import SummaryWriter

from scipy.stats import norm

# from helper_oc_controller import HelperOCController


from wrappers import RecordEpisodeStatisticsWithCost, SauteWrapper
import atu

# allow pygame to render headless


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
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
    parser.add_argument("--eval-every", type=int, default=10,
        help="Eval")
    parser.add_argument("--lagrange", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use Sac Lagrange") # https://github.com/AlgTUDelft/WCSAC/blob/main/wc_sac/sac/saclag.py
    parser.add_argument("--saute", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use Saute") 
    parser.add_argument("--saute-discount-factor", type=float, default=0.99, nargs="?", const=True,
        help="Use Saute") 
    parser.add_argument("--saute-unsafe-reward", type=float, default=1.0, nargs="?", const=True,
        help="Use Saute") 
    parser.add_argument("--saute-safety-budget", type=float, default=1.0, nargs="?", const=True,
        help="Use Saute") 
    parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Render")

    # Algorithm specific arguments
    # parser.add_argument("--env-id", type=str, default="Safe-Pendulum-v1",
    parser.add_argument("--env-id", type=str, default="Safe-DubinsHallway-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=0,
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
    parser.add_argument("--beta", type=float, default=0.1,
            help="Lagrangian Penalty Term")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--use-hj", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use safety controller")
    parser.add_argument("--always-use-hj", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use safety controller")
    parser.add_argument("--ra", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Reach Avoid")
    parser.add_argument("--sample-uniform", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="sampe hj uniform")
    parser.add_argument("--reward-hj", type=int, default=0,
        help="bonus for encouraging hj")
    parser.add_argument("--reward-shape", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="pentalty for bad actions")
    parser.add_argument("--done-if-unsafe", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Reset if unsafe, use min_reward / (1 - dicount facor")
    parser.add_argument("--use-min-reward", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Reset if unsafe, use min_reward")
    parser.add_argument("--unform_safe_action", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="sample action uniformally that are safe")
    parser.add_argument("--imagine-trajectory", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="sample action uniformally that are safe")
    args = parser.parse_args()
    # fmt: on
    return args


# controller = HelperOCController()


def make_env(env_id, seed, idx, capture_video, run_name, saute):
    def thunk():
        if "Pendulum" in env_id:
            env = gym.make(
                env_id, matlab_controller=controller, done_if_unsafe=args.done_if_unsafe
            )
        elif "Hallway" in env_id:
            env = gym.make(
                env_id, use_reach_avoid=args.ra, done_if_unsafe=args.done_if_unsafe
            )
        else:
            env = gym.make(env_id)
        env = RecordEpisodeStatisticsWithCost(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: obs / env.world_boundary
        )
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

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


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super(SoftQNetwork, self).__init__()
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
        super(Actor, self).__init__()
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
        return super(Actor, self).to(device)


def eval_policy(envs, actor, global_step):
    print("eval policy")
    actor.eval()
    total_episodic_return = 0
    total_episodic_cost = 0
    total_episodic_unsafe = 0
    EPISODES = 10
    for episode in range(EPISODES):
        obs = envs.reset()
        done = False
        while not done:
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

            next_obs, rewards, dones, infos = envs.step(actions)
            costs = [not info["safe"] for info in infos]

            obs = next_obs
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    total_episodic_return += info["episode"]["r"]
                    total_episodic_cost += info["episode"]["c"]
                    total_episodic_unsafe += info["episode"]["us"]
                    done = True
                    break

    writer.add_scalar("eval/return", total_episodic_return / EPISODES, global_step)
    writer.add_scalar(
        "eval/total_unsafe", total_episodic_unsafe / EPISODES, global_step
    )
    writer.add_scalar("eval/total_cost", total_episodic_cost / EPISODES, global_step)
    print(
        f"eval: average return: {total_episodic_return / EPISODES} average cost: {total_episodic_cost / EPISODES} average unsafe = {total_episodic_unsafe / EPISODES}"
    )

    actor.train()


if __name__ == "__main__":
    args = parse_args()
    if not args.render:
        import os

        os.environ["SDL_VIDEODRIVER"] = "dummy"

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
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name, args.saute)]
    )

    eval_envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + 1_000,
                0,
                args.capture_video,
                run_name,
                args.saute,
            )
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

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
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = CostReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
    )

    rb_near_brt = CostReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()

    unsafe_counter = 0
    hj_use_counter = 0

    for global_step in range(args.total_timesteps):
        if args.render:
            envs.envs[0].render()

        used_hj = False
        near_brt = False
        og_actions = []
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

            if args.use_hj:

                # for idx, env in enumerate(envs.envs):
                #     state = env.state
                #     opt_ctrl, value = controller.opt_ctrl_value(state)
                #     use_opt = False
                #     if value <= 0.1:
                #         # print(f"using hj {state=} {hj_use_counter=}")
                #         use_opt = True
                #         hj_use_counter += 1
                #         og_actions.append((idx, copy.deepcopy(actions[idx])))

                #         if args.sample_uniform:
                #             low, high = controller.safe_ctrl_bnds(state)
                #             opt_ctrl = [np.random.uniform(low, high)]

                #         actions[idx] = opt_ctrl

                #     if value <= 0.2:
                #         near_brt = True
                #     writer.add_scalar("sanity/use_hj", hj_use_counter, global_step)

                for idx, env in enumerate(envs.envs):
                    if env.use_opt_ctrl() or args.always_use_hj:
                        used_hj = True
                        opt_ctrl = env.opt_ctrl()
                        hj_use_counter += 1
                        og_actions.append((idx, copy.deepcopy(actions[idx])))
                        actions[idx] = opt_ctrl

                    writer.add_scalar("sanity/use_hj", hj_use_counter, global_step)

                    if args.imagine_trajectory:
                        imaginary_env = copy.deepcopy(env)
                        next_obs, reward, done, info = env.step(
                            og_action[-1][1]
                        )  # action policy took
                        cost = info.get("cost", 0)

        next_obs, rewards, dones, infos = envs.step(actions)
        costs = [info.get("cost", 0) for info in infos]

        if args.reward_shape:
            reward_shape_rewards = copy.deepcopy(rewards)
            reward_shape_actions = copy.deepcopy(actions)
            for idx, og_action in og_actions:
                reward_shape_rewards[idx] = envs.envs[0].min_reward
                reward_shape_actions[idx] = og_action
                print(f"replacing: {og_action} with {actions[idx]}")

        if args.use_hj and args.reward_hj != 0:
            for idx, og_action in og_actions:
                rewards[idx] = args.reward_hj

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "safe" in info.keys():
                unsafe_counter += not info["safe"]
                writer.add_scalar("sanity/unsafe_counter", unsafe_counter, global_step)

            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']} episodic_cost={info['episode']['c']} unsafe={info['episode']['us']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_cost", info["episode"]["c"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_unsafe", info["episode"]["us"], global_step
                )

                writer.add_scalar(
                    "charts/total_cost", info["episode"]["tc"], global_step
                )

                writer.add_scalar(
                    "charts/total_unsafe", info["episode"]["tus"], global_step
                )

                if envs.envs[0].episode_count % args.eval_every == 0:
                    eval_policy(eval_envs, actor, global_step)
                break

        if args.done_if_unsafe:
            for idx, info in enumerate(infos):
                if not info["safe"]:
                    if args.use_min_reward:
                        rewards[idx] = envs.envs.min_reward
                    else:
                        rewards[idx] = envs.envs.min_reward / (1 - args.gamma)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, costs, dones, infos)

        if args.use_hj and near_brt:
            rb_near_brt.add(obs, real_next_obs, actions, rewards, costs, dones, infos)

        if args.reward_shape:
            rb.add(
                obs,
                real_next_obs,
                reward_shape_actions,
                reward_shape_rewards,
                costs,
                dones,
                infos,
            )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            for idx, env in enumerate(envs.envs):
                info = infos[idx]
                if "safe" in info.keys():
                    if not info["safe"]:
                        print(f"unsafe state: {env.state}")

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

                if args.lagrange:
                    qfc_next_target = qfc_target(
                        data.next_observations, next_state_actions
                    )
                    next_qc_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * args.gamma * (qfc_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = (qf1_loss + qf2_loss) / 2

            q_optimizer.zero_grad()
            qf_loss.backward()
            nn.utils.clip_grad_norm_(
                list(qf1.parameters()) + list(qf2.parameters()), args.max_grad_norm
            )
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)

                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(actor.parameters()), args.max_grad_norm
                    )
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    eval_policy(eval_envs, actor, global_step)
    envs.close()
    eval_envs.close()
    writer.close()
