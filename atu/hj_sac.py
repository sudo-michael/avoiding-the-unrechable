# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
# commit: 15df5c0eeb3459cc5b3dadd722751a80261c4a5e
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool
import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import atu
from atu.wrappers import RecordEpisodeStatisticsWithCost


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--group-name", type=str, default="",
        help="group name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="atu2",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--eval-every", type=int, default=5_000,
        help="Eval every x steps")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Safe-DubinsHallway-v1",
    # parser.add_argument("--env-id", type=str, default="Safe-DubinsHallway4D-v0",
    # parser.add_argument("--env-id", type=str, default="Safe-SingleNarrowPassage-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=125_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=100_000,
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
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--use-hj", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use safety controller")
    parser.add_argument("--use-dist", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use disturbances")
    parser.add_argument("--reward-shape", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="pentalty for bad actions")
    parser.add_argument("--reward-shape-penalty", type=float, default=9.4069,
        help="reward pentalty for hj takeover")
    parser.add_argument("--reward-shape-gradv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use reward shaping based on gradVdotF")
    parser.add_argument("--reward-shape-gradv-takeover", type=float, default=-0.5,
        help="input for min(gradVdotF, x) (cost for using hj")
    parser.add_argument("--seperate-cost", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use reward shaping based on gradVdotF")
    parser.add_argument("--done-if-unsafe", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Reset if unsafe")
    parser.add_argument('--train-dist', type=lambda s: np.array([float(item) for item in s.split(',')]),
        help="Training Disturbance")
    parser.add_argument("--train-speed", type=float, default=1.0,
        help="Training Speed")
    parser.add_argument('--eval-dist', type=lambda s: np.array([float(item) for item in s.split(',')]),
        help="Eval Disturbance")
    parser.add_argument("--eval-speed", type=float, default=1.0,
        help="Eval Speed")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(args, seed, capture_video, idx, run_name, eval=False):
    def thunk():
        if eval:
            env = gym.make(
                args.env_id,
                done_if_unsafe=args.done_if_unsafe,
                use_disturbances=args.use_dist,
                dist=args.eval_dist,
                speed=args.eval_speed,
                eval=eval,
            )
        else:
            env = gym.make(
                args.env_id,
                done_if_unsafe=args.done_if_unsafe,
                use_disturbances=args.use_dist,
                dist=args.train_dist,
                speed=args.train_speed,
            )
        env = RecordEpisodeStatisticsWithCost(env)
        if capture_video:
            if idx == 0:
                if eval:
                    env = gym.wrappers.RecordVideo(env, f"videos_eval/{run_name}")
                else:
                    env = gym.wrappers.RecordVideo(env, f"eval/{run_name}")
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
        self.register_buffer(
            "action_scale",
            torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0),
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0),
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
        [
            make_env(
                args,
                seed=args.seed,
                capture_video=args.capture_video,
                idx=0,
                run_name=run_name,
            )
        ]
    )

    eval_envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args,
                seed=args.seed + 1_000,
                capture_video=args.capture_video,
                idx=0,
                run_name=run_name,
                eval=True,
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

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )

    start_time = time.time()

    def eval_policy(envs, actor, episodes=10):
        print("eval policy")
        actor.eval()
        total_episodic_return = 0
        total_episodic_cost = 0
        total_episodic_unsafe = 0
        total_reach_goal = 0
        total_hj_at_collision = 0
        collision_count = 0
        for episode in range(episodes):
            obs = envs.reset()
            done = False
            while not done:
                with torch.no_grad():
                    actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.cpu().numpy()
                next_obs, rewards, dones, infos = envs.step(actions)

                obs = next_obs
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                for info in infos:
                    if "episode" in info.keys():
                        total_episodic_return += info["episode"]["r"]
                        total_episodic_cost += info["episode"]["c"]
                        total_episodic_unsafe += info["episode"]["us"]
                        total_reach_goal += info.get("reach_goal", False)
                        done = True
                        if not info.get("reach_goal", False) and not info.get(
                            "TimeLimit.truncated", False
                        ):
                            total_hj_at_collision += info["hj_value"]
                            collision_count += 1

        print(
            f"eval: average return: {total_episodic_return / episodes} average cost: {total_episodic_cost / episodes} average unsafe = {total_episodic_unsafe / episodes} average reach goal = {total_reach_goal / episodes}"
        )
        actor.train()

        avg_hj = 0
        if collision_count > 0:
            avg_hj = total_hj_at_collision / collision_count
        return {
            "average_return": total_episodic_return / episodes,
            "average_cost": total_episodic_cost / episodes,
            "average_unsafe": total_episodic_unsafe / episodes,
            "average_reach_goal": total_reach_goal / episodes,
            "average_hj_at_collision": avg_hj,
        }

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    thj = 0  # wrapper didn't work
    for global_step in range(args.total_timesteps):
        # envs.envs[0].render(mode='human')
        used_hj = False
        if args.use_hj:
            og_actions = []
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))

            actions = actions.detach().cpu().numpy()

            if args.use_hj:
                for idx, env in enumerate(envs.envs):
                    if env.use_opt_ctrl():
                        opt_ctrl = env.opt_ctrl()
                        used_hj = True
                        og_actions.append((idx, copy.deepcopy(actions[idx])))
                        actions[idx] = opt_ctrl
                        thj += 1

            # print(obs, used_hj, actions, envs.envs[0].action_safe(obs[0], actions[0]))

        # TRY NOT TO MODIFY: execute the game and log data.
        # hack to add `used_hj` to info to calculate reward + track with wrapper
        # makes actions be of shape (1, 1), but it should really be (1,)
        next_obs, rewards, dones, infos = envs.step(
            [{"used_hj": used_hj, "actions": actions}]
        )

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
                    "charts/episodic_cost", info["episode"]["c"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_unsafe", info["episode"]["us"], global_step
                )
                writer.add_scalar(
                    "charts/percent_reach_goal", info["episode"]["prg"], global_step
                )
                writer.add_scalar(
                    "charts/total_reach_goal", info["episode"]["trg"], global_step
                )
                writer.add_scalar(
                    "charts/total_cost", info["episode"]["tc"], global_step
                )
                writer.add_scalar(
                    "charts/total_use_hj", info["episode"]["thj"], global_step
                )
                writer.add_scalar(
                    "charts/total_unsafe", info["episode"]["tus"], global_step
                )
                writer.add_scalar(
                    "charts/total_use_hj", info["episode"]["thj"], global_step
                )

                if (
                    not info.get("reach_goal", False)
                    and dones[0]
                    and not info.get("TimeLimit.truncated", False)
                ):
                    print(f"DEBUG collision: {obs=}")
                    print(f"DEBUG {info['hj_value']=}")
                    print(f"DEBUG {info['collision']=}")
                    writer.add_scalar(
                        "charts/hj_at_collision", info["hj_value"], global_step
                    )
                break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        if args.reward_shape and len(og_actions):
            reward_shape_actions = copy.deepcopy(actions)
            reward_shape_rewards = copy.deepcopy(rewards)
            for idx, og_action in og_actions:
                sim_next_obs, sim_rew, sim_done, sim_info = envs.envs[
                    idx
                ].simulate_step(np.copy(obs[idx]), og_action)
                # rescale obseration
                # sim_next_obs /= envs.envs[idx].world_boundary

                # # DEBUG
                # dbg_next_obs, dbg_rew, dbg_done, db_info = envs.envs[idx].simulate_step(
                #     np.copy(obs[idx]), actions[idx]
                # )
                # # # dbg_next_obs /= envs.envs[idx].world_boundary
                # if not np.allclose(dbg_next_obs, real_next_obs[idx]):
                #     print(dbg_next_obs)
                #     print(real_next_obs[idx])
                #     breakpoint()
                # assert np.all(dbg_next_obs == real_next_obs[idx]), print(
                #     dbg_next_obs, real_next_obs[idx]
                # )
                # # DEBUG

                reward_shape_actions[idx] = og_action
                if args.reward_shape_gradv and args.seperate_cost:
                    gradVdotFxu = envs.envs[idx].reward_penalty(obs[idx], og_action)
                    cost = -args.reward_shape_penalty
                    cost += min(gradVdotFxu, 0) * args.reward_shape_gradv_takeover
                    assert cost <= 0, f"{cost=} must be not positive: {gradVdotFxu=}"
                    writer.add_scalar("charts/reward_shape_cost", cost, global_step)
                    writer.add_scalar("charts/gradVdotFxu", gradVdotFxu, global_step)
                    reward_shape_rewards[idx] += cost
                elif args.reward_shape_gradv:
                    gradVdotFxu = envs.envs[idx].reward_penalty(obs[idx], og_action)
                    # min(gardVdotFxu, 0) since there shouldn't be a penalty for taking a safe action
                    # adding args.reward_shape_gradv_takeover since not penalizing for using hj
                    # isn't desincentivising agent from learning how to take unsafe action
                    cost = args.reward_shape_penalty * min(
                        gradVdotFxu, args.reward_shape_gradv_takeover
                    )

                    assert cost <= 0, f"{cost=} must be not positive: {gradVdotFxu=}"
                    writer.add_scalar("charts/reward_shape_cost", cost, global_step)
                    writer.add_scalar("charts/gradVdotFxu", gradVdotFxu, global_step)
                    reward_shape_rewards[idx] += cost
                else:
                    # previously had a bug of
                    # reward_shape_rewards[idx] -= envs.envs[idx].min_reward # bad since min_reward was negative
                    cost = -args.reward_shape_penalty
                    reward_shape_rewards[idx] += cost
                    writer.add_scalar("charts/reward_shape_cost", cost, global_step)

            # since only 1 instance, just wrapping everything in lists
            rb.add(
                obs,
                [sim_next_obs],
                reward_shape_actions,
                reward_shape_rewards,
                dones,
                infos,
            )

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
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

                writer.add_scalar("charts/thj", thj, global_step)

        if (global_step + 1) % args.eval_every == 0:
            stats = eval_policy(eval_envs, actor)
            writer.add_scalar("eval/return", stats["average_return"], global_step)
            writer.add_scalar("eval/total_unsafe", stats["average_cost"], global_step)
            writer.add_scalar("eval/total_cost", stats["average_unsafe"], global_step)
            writer.add_scalar(
                "eval/reach_goal", stats["average_reach_goal"], global_step
            )
            writer.add_scalar(
                "eval/hj_at_collision", stats["average_hj_at_collision"], global_step
            )

    stats = eval_policy(eval_envs, actor)
    writer.add_scalar("eval/return", stats["average_return"], global_step)
    writer.add_scalar("eval/total_unsafe", stats["average_cost"], global_step)
    writer.add_scalar("eval/total_cost", stats["average_unsafe"], global_step)
    writer.add_scalar("eval/reach_goal", stats["average_reach_goal"], global_step)
    writer.add_scalar(
        "eval/hj_at_collision", stats["average_hj_at_collision"], global_step
    )
    envs.close()
    eval_envs.close()
    writer.close()
