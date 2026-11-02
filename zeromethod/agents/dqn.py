"""PyTorch Deep Q-Network agent for the 18-action cube environment.

PyTorch is imported only when this module is imported, keeping the core
ZeroMethod environment and tests dependency-free.
"""

from copy import deepcopy
from pathlib import Path
from random import Random

import torch
from torch import nn

from zeromethod.environment.actions import ACTION_COUNT
from .replay_buffer import ReplayBuffer
from .state_encoding import ENCODED_SIZE, encode_observation


class QNetwork(nn.Module):
    """MLP mapping a 324-feature sticker state to 18 action values."""

    def __init__(self, hidden_size=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(ENCODED_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ACTION_COUNT),
        )

    def forward(self, states):
        return self.layers(states)


class DQNAgent:
    """DQN with experience replay, a target network, and epsilon-greedy action selection."""

    def __init__(
        self, *, seed, device=None, hidden_size=512, learning_rate=1e-4,
        gamma=0.99, batch_size=64, replay_capacity=100_000,
        target_update_interval=1_000,
    ):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if batch_size <= 0 or target_update_interval <= 0:
            raise ValueError("batch_size and target_update_interval must be positive")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._rng = Random(seed)
        # Isolate model initialization from the caller's global Torch RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.online_network = QNetwork(hidden_size).to(self.device)
        self.target_network = deepcopy(self.online_network).to(self.device)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.replay_buffer = ReplayBuffer(replay_capacity, seed ^ 0x5A17)
        self.training_steps = 0

    def act(self, observation, *, epsilon=0.0):
        """Choose an action from the observation; epsilon=0 is greedy evaluation."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        if self._rng.random() < epsilon:
            return self._rng.randrange(ACTION_COUNT)
        features = torch.tensor(
            encode_observation(observation), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.online_network.eval()
        with torch.inference_mode():
            action = int(self.online_network(features).argmax(dim=1).item())
        self.online_network.train()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self):
        """Perform one Double-DQN-style? No: standard DQN target update."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(
            [encode_observation(item.state) for item in batch],
            dtype=torch.float32, device=self.device,
        )
        actions = torch.tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            [encode_observation(item.next_state) for item in batch],
            dtype=torch.float32, device=self.device,
        )
        dones = torch.tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device
        )

        predicted = self.online_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_values * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return float(loss.detach().cpu().item())

    def save(self, path, *, metadata=None):
        """Save model, optimizer, and reproducibility metadata to a checkpoint."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "metadata": dict(metadata or {}),
        }, destination)
        return destination
