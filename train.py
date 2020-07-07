import datetime
from pathlib import Path

import torch.multiprocessing as mp

import utils
from actor import Actor
from learner import Learner
from models import MlpPolicy, MlpValueFn

hparams = utils.Hyperparameters(
    max_updates=50,
    policy_hidden_dims=128,
    value_fn_hidden_dims=128,
    batch_size=32,
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
    # policy_lr=1e-3,
    # value_fn_lr=1e-3,
    lr=1e-3,
    policy_loss_c=1,
    v_loss_c=0.5,
    entropy_c=0.0006,
    max_timesteps=1000,
    queue_lim=8,
    max_norm=10,
    n_actors=1,
    env_name="CartPole-v1",          # "RacecarBulletEnv-v0",
    log_path="./logs/",
    save_every=50,
    eval_every=2,
    eval_eps=20,
    verbose=1,
    render=False,
)

if __name__ == "__main__":

    start_time = datetime.datetime.now()

    mp.set_start_method("fork", force=True)

    print(f"[main] Start time: {start_time:%d-%m-%Y %H:%M:%S}")
    print(f"[main] {hparams}\n")

    if hparams.log_path is not None:
        log_path = Path(Path(hparams.log_path) / f"{start_time:%d%m%Y%H%M%S}")
        log_path.mkdir(parents=True, exist_ok=True)
        with open(Path(log_path / "hyperparameters.txt"), "w+") as f:
            f.write(f"{hparams}")
        if not hparams.save_every > 0:
            raise ValueError(
                f"save_every hyperparameter should be greater than 0, "
                f"got {hparams.save_every}"
            )
    else:
        log_path = None

    q = mp.Queue(maxsize=hparams.queue_lim)
    update_counter = utils.Counter(init_val=0)
    env = utils.make_env(hparams.env_name)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()
    policy = MlpPolicy(observation_size, action_size, hparams.policy_hidden_dims)
    policy.share_memory()
    value_fn = MlpValueFn(observation_size, hparams.value_fn_hidden_dims)
    learner = Learner(1, hparams, policy, value_fn, q, update_counter, log_path)

    actors = []
    for i in range(hparams.n_actors):
        policy = MlpPolicy(observation_size, action_size, hparams.policy_hidden_dims)
        actors.append(
            Actor(i + 1, hparams, policy, learner, q, update_counter, log_path)
        )

    print("[main] Initialized")

    for a in actors:
        a.start()
    learner.start()

    learner.completion.wait()
    for a in actors:
        a.completion.wait()

    learner.terminate()
    for a in actors:
        a.terminate()

    learner.join()
    for a in actors:
        a.join()

    print(
        f"[main] Completed in {(datetime.datetime.now() - start_time).seconds} seconds"
    )
