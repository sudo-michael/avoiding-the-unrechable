# https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ddpg_continuous_action.py
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
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import os
import copy

os.environ["SDL_VIDEODRIVER"] = "dummy"

from safe_pendulum import PendulumEnv


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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Safe-Pendulum-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=30_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--learning-rate-critic", type=float, default=3e-2,
        help="the learning rate of the optimizer for critic")
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
    parser.add_argument("--learning-starts", type=int, default=10_000,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--use-hj", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use safety controller")
    parser.add_argument("--use-bad-rb", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use safety controller")
    parser.add_argument("--reward-shape", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="pentalty for bad actions")
    args = parser.parse_args()
    # fmt: on
    return args


from helper_oc_controller import HelperOCController

controller = HelperOCController()


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if "Safe" in env_id:
            env = gym.make(env_id, matlab_controller=controller)
        else:
            env = gym.make(env_id)
        # env = PendulumEnv(matlab_controller=controller)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
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


class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x))


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
        [make_env(args.env_id, 0, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    # max_action =   # limit torque
    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
    )

    bad_rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
    )

    loss_fn = nn.MSELoss()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()

    import tqdm

    unsafe_counter = 0
    hj_use_counter = 0

    for global_step in tqdm.tqdm(range(args.total_timesteps)):
        # ALGO LOGIC: put action logic here
        used_hj = False
        original_actions = []
        if global_step < args.learning_starts:
            actions = envs.action_space.sample()
        else:
            actions = actor.forward(torch.Tensor(obs).to(device))
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

            if args.use_hj:
                used_hj = True

                for idx, env in enumerate(envs.envs):
                    state = env.state
                    opt_ctrl, value = controller.opt_ctrl_value(state)
                    use_opt = False
                    if value <= 0.1:
                        # print(f"using hj {state=} {hj_use_counter=}")
                        use_opt = True
                        hj_use_counter += 1
                        original_actions.append((idx, actions[idx]))
                        actions[idx] = opt_ctrl

                    writer.add_scalar("sanity/use_hj", hj_use_counter, global_step)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        if args.reward_shape:
            reward_shape_rewards = copy.deepcopy(rewards)
            reward_shape_actions = copy.deepcopy(actions)
            for idx, og_action in original_actions:
                reward_shape_rewards[idx] = -17  # min reward
                reward_shape_actions[idx] = og_action

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                break

            if "safe" in info.keys():
                unsafe_counter += not info["safe"]
                writer.add_scalar("sanity/unsafe_counter", unsafe_counter, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        if args.reward_shape:
            rb.add(
                obs,
                real_next_obs,
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
                next_state_actions = (
                    target_actor.forward(data.next_observations)
                ).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target.forward(
                    data.next_observations, next_state_actions
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1.forward(data.observations, data.actions).view(-1)
            qf1_loss = loss_fn(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            nn.utils.clip_grad_norm_(list(qf1.parameters()), args.max_grad_norm)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1.forward(
                    data.observations, actor.forward(data.observations)
                ).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()), args.max_grad_norm)
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

            if bad_rb.size() > 10 and args.use_bad_rb:
                data = bad_rb.sample(bad_rb.size() // 2)
                with torch.no_grad():
                    next_state_actions = (
                        target_actor.forward(data.next_observations)
                    ).clamp(
                        envs.single_action_space.low[0],
                        envs.single_action_space.high[0],
                    )
                    qf1_next_target = qf1_target.forward(
                        data.next_observations, next_state_actions
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * args.gamma * (qf1_next_target).view(-1)

                qf1_a_values = qf1.forward(data.observations, data.actions).view(-1)
                qf1_loss = loss_fn(qf1_a_values, next_q_value)

                # optimize the model
                q_optimizer.zero_grad()
                qf1_loss.backward()
                nn.utils.clip_grad_norm_(list(qf1.parameters()), args.max_grad_norm)
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    actor_loss = -qf1.forward(
                        data.observations, actor.forward(data.observations)
                    ).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(actor.parameters()), args.max_grad_norm
                    )
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
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

    envs.close()
    writer.close()
