import datetime
from collections import namedtuple
from pathlib import Path
from typing import List, Union

import gym
import pybullet_envs  # noqa: F401
import numpy as np
import torch
import torch.multiprocessing as mp

from models import MlpPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

Hyperparameters = namedtuple(
    "Hyperparameters",
    [
        "max_updates",
        "policy_hidden_dims",
        "value_fn_hidden_dims",
        "batch_size",
        "gamma",
        "rho_bar",
        "c_bar",
        # "policy_lr",
        # "value_fn_lr",
        "lr",
        "policy_loss_c",
        "v_loss_c",
        "entropy_c",
        "max_timesteps",
        "queue_lim",
        "max_norm",
        "n_actors",
        "env_name",
        "log_path",
        "save_every",
        "eval_every",
        "eval_eps",
        "verbose",
        "render",
    ],
)


class Trajectory:
    def __init__(
        self,
        id: int,
        observations: List[torch.Tensor] = [],
        actions: List[torch.Tensor] = [],
        rewards: List[torch.Tensor] = [],
        dones: List[torch.Tensor] = [],
        logits: List[torch.Tensor] = [],
    ):
        self.id = id
        self.obs = observations
        self.a = actions
        self.r = rewards
        self.d = dones
        self.logits = logits

    def add(
        self,
        obs: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        d: torch.Tensor,
        logits: torch.Tensor,
    ):
        self.obs.append(obs)
        self.a.append(a)
        self.r.append(r)
        self.d.append(d)
        self.logits.append(logits)


class Counter:
    def __init__(self, init_val: int = 0):
        self._val = mp.RawValue("i", init_val)
        self._lock = mp.Lock()

    def increment(self):
        with self._lock:
            self._val.value += 1

    @property
    def value(self):
        with self._lock:
            return self._val.value


def make_env(env_name: str):
    if "Bullet" in env_name:
        try:
            env = gym.make(env_name, isDiscrete=True)
        except TypeError:
            env = gym.make(env_name)
    else:
        env = gym.make(env_name)

    if env.action_space.__class__.__name__ != "Discrete":
        raise NotImplementedError("Continuous environments not supported yet")

    return env


def test_policy(
    policy: MlpPolicy,
    env: Union[gym.Env, str],
    episodes: int,
    deterministic: bool,
    max_episode_len: int,
    log_dir: Union[str, None] = None,
    verbose: bool = False,
):
    start_time = datetime.datetime.now()
    start_text = f"Started testing at {start_time:%d-%m-%Y %H:%M:%S}\n"

    if type(env) == str:
        env = make_env(env)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fpath = Path(log_dir).joinpath(f"test_log_{start_time:%d%m%Y%H%M%S}.txt")
        fpath.write_text(start_text)
    if verbose:
        print(start_text)
    policy.eval()
    rewards = []
    for e in range(episodes):
        obs = env.reset()
        obs = torch.tensor(obs, device=device, dtype=dtype)
        d = False
        ep_rewards = []
        for t in range(max_episode_len):
            action, _ = policy.select_action(obs, deterministic)
            obs, r, d, _ = env.step(action.item())
            obs = torch.tensor(obs, device=device, dtype=dtype)
            ep_rewards.append(r)
            if d:
                break
        rewards.append(sum(ep_rewards))
        ep_text = f"Episode {e+1}: Reward = {rewards[-1]:.2f}\n"
        if log_dir is not None:
            with open(fpath, mode="a") as f:
                f.write(ep_text)
        if verbose:
            print(ep_text)
    avg_reward = np.mean(rewards)
    std_dev = np.std(rewards)
    complete_text = (
        f"-----\n"
        f"Testing completed in "
        f"{(datetime.datetime.now() - start_time).seconds} seconds\n"
        f"Average Reward per episode: {avg_reward}"
    )
    if verbose:
        print(complete_text)
    if log_dir is not None:
        with open(fpath, mode="a") as f:
            f.write(complete_text)

    return avg_reward, std_dev
