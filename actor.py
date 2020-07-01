import queue
from pathlib import Path
from typing import Union

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import utils
from learner import Learner
from models import MlpPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class Actor:
    def __init__(
        self,
        id: int,
        hparams: utils.Hyperparameters,
        policy: MlpPolicy,
        learner: Learner,
        q: mp.Queue,
        update_counter: utils.Counter,
        log_path: Union[Path, str, None] = None,
        timeout=10,
    ):
        self.id = id
        self.hp = hparams
        self.policy = policy
        for p in self.policy.parameters():
            p.requires_grad = False
        self.learner = learner
        self.timeout = timeout
        self.q = q
        self.update_counter = update_counter
        self.log_path = log_path
        if self.log_path is not None:
            self.log_path = Path(self.log_path) / Path(f"a{self.id}")
            self.log_path.mkdir(parents=True, exist_ok=False)

        self.completion = mp.Event()
        self.p = mp.Process(target=self._act, name=f"actor_{self.id}")
        print(f"[main] actor_{self.id} Initialized")

    def start(self):
        self.p.start()
        print(f"[main] Started actor_{self.id} with pid {self.p.pid}")

    def terminate(self):
        self.p.terminate()
        print(f"[main] Terminated actor_{self.id}")

    def join(self):
        self.p.join()

    def _act(self):
        try:

            if self.log_path is not None:
                writer = SummaryWriter(self.log_path)
                writer.add_text("hyperparameters", f"{self.hp}")

            env = utils.make_env(self.hp.env_name)
            traj_no = 0

            while not self.learner.completion.is_set():
                traj_no += 1
                self.policy.load_state_dict(self.learner.policy_weights)
                traj_id = (self.id, traj_no)
                traj = utils.Trajectory(traj_id, [], [], [], [], [])
                obs = env.reset()
                obs = torch.tensor(obs, device=device, dtype=dtype)
                traj.obs.append(obs)
                c = 0

                if self.hp.verbose >= 2:
                    print(f"[actor_{self.id}] Starting traj_{traj.id}")

                # record trajectory
                while c < self.hp.max_timesteps:
                    if self.hp.render:
                        env.render()
                    c += 1
                    a, logits = self.policy.select_action(obs)
                    # print(f"[actor_{self.id}] a_probs: {a_probs}")
                    obs, r, d, _ = env.step(a.item())
                    obs = torch.tensor(obs, device=device, dtype=dtype)
                    r = torch.tensor(r, device=device, dtype=dtype)
                    d = torch.tensor(d, device=device)
                    traj.add(obs, a, r, d, logits)

                    if d:
                        break

                if self.hp.verbose >= 2:
                    print(
                        f"[actor_{self.id}] traj_{traj.id} completed Reward = {sum(traj.r)}"
                    )
                if self.log_path is not None:
                    # action_one_hot = torch.zeros(env.action_space.n)
                    # action_one_hot[a] += 1
                    writer.add_histogram(
                        f"actor_{self.id}/actions/action_taken", a, traj_no
                    )
                    writer.add_histogram(
                        f"actor_{self.id}/actions/logits", logits.detach(), traj_no
                    )
                    writer.add_scalar(
                        f"actor_{self.id}/rewards/trajectory_reward",
                        sum(traj.r),
                        traj_no,
                    )

                while True:
                    try:
                        self.q.put(traj, timeout=self.timeout)
                        break
                    except queue.Full:
                        if self.learner.completion.is_set():
                            break
                        else:
                            continue

            if self.log_path is not None:
                writer.close()
            env.close()
            print(f"[actor_{self.id}] Finished acting")
            self.completion.set()
            return

        except KeyboardInterrupt:
            print(f"[actor_{self.id}] interrupted")
            if self.log_path is not None:
                writer.close()
            env.close()
            self.completion.set()
            return

        except Exception as e:
            if self.log_path is not None:
                writer.close()
            env.close()
            print(f"[actor_{self.id}] encoutered exception")
            raise e
