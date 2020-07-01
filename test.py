import argparse

import torch

import utils
from models import MlpPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pp", "--policy_path", type=str, required=True, help="path to policy weights"
    )
    parser.add_argument(
        "-hd",
        "--policy_hidden_dim",
        type=int,
        required=True,
        help="dimension of hidden layer",
    )
    parser.add_argument(
        "-en",
        "--env_name",
        type=str,
        required=True,
        help="name of gym envrionment to test in",
    )
    parser.add_argument(
        "-ne",
        "--num_episodes",
        type=int,
        default=10,
        help="number of episodes to test for",
    )
    parser.add_argument(
        "-d", "--deterministic", help="use deterministic policy", action="store_true"
    )
    parser.add_argument(
        "-el",
        "--max_episode_len",
        type=int,
        default=1000,
        help="maximum number of steps per episode",
    )
    parser.add_argument(
        "-ld",
        "--log_dir",
        type=str,
        required=False,
        help="directory to store log file in",
    )
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )

    args = parser.parse_args()
    env = utils.make_env(args.env_name)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = MlpPolicy(observation_size, action_size, args.policy_hidden_dim)
    checkpoint = torch.load(args.policy_path)
    policy.load_state_dict(checkpoint["policy_state_dict"])

    utils.test_policy(
        policy,
        env,
        args.num_episodes,
        args.deterministic,
        args.max_episode_len,
        args.log_dir,
        args.verbose,
    )

    env.close()
