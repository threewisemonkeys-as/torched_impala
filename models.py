import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class MlpPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super(MlpPolicy, self).__init__()
        self.model = (
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Dropout(p=0.8),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            .to(device)
            .to(dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

    def select_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.forward(obs)
        if deterministic:
            action = torch.argmax(logits)
        else:
            action = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        return action, logits


class MlpValueFn(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super(MlpValueFn, self).__init__()
        self.model = (
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Dropout(p=0.8),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            .to(device)
            .to(dtype)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)
