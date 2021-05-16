from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, Generic, Iterable, List, Sequence, TypeVar

import numpy as np
import torch

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.utils.collate import collate_numpy, collate_torch
from asym_rlpo.utils.convert import numpy2torch
from asym_rlpo.utils.debugging import checkraise

Torch_S = TypeVar(
    'Torch_S',
    torch.Tensor,
    Dict[str, torch.Tensor],
)
Torch_O = TypeVar(
    'Torch_O',
    torch.Tensor,
    Dict[str, torch.Tensor],
)

S = TypeVar(
    'S',
    torch.Tensor,
    Dict[str, torch.Tensor],
    np.ndarray,
    Dict[str, np.ndarray],
)
O = TypeVar(
    'O',
    torch.Tensor,
    Dict[str, torch.Tensor],
    np.ndarray,
    Dict[str, np.ndarray],
)


class Interaction(Generic[S, O]):
    def __init__(
        self,
        *,
        state: S,
        observation: O,
        action: int,
        reward: float,
    ):
        self.state: S = state
        self.observation: O = observation
        self.action = action
        self.reward = reward


class Batch(Generic[S, O]):
    def __init__(
        self,
        *,
        states,
        observations,
        actions,
        rewards,
        next_states,
        next_observations,
        dones,
    ):
        self.states = states
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.next_observations = next_observations
        self.dones = dones

    def __len__(self):
        return len(self.dones)

    def torch(self) -> Batch:
        checkraise(
            isinstance(self.states, np.ndarray)
            and isinstance(self.observations, np.ndarray)
            and isinstance(self.actions, np.ndarray)
            and isinstance(self.rewards, np.ndarray)
            and isinstance(self.dones, np.ndarray),
            TypeError,
            'Batch is not numpy to begin with??',
        )
        return Batch(
            states=numpy2torch(self.states),
            observations=numpy2torch(self.observations),
            actions=numpy2torch(self.actions),
            rewards=numpy2torch(self.rewards),
            next_states=numpy2torch(self.next_states),
            next_observations=numpy2torch(self.next_observations),
            dones=numpy2torch(self.dones),
        )

    def to(self, device: torch.device) -> Batch:
        return Batch(
            states=gtorch.to(self.states, device),
            observations=gtorch.to(self.observations, device),
            actions=gtorch.to(self.actions, device),
            rewards=gtorch.to(self.rewards, device),
            next_states=gtorch.to(self.next_states, device),
            next_observations=gtorch.to(self.next_observations, device),
            dones=gtorch.to(self.dones, device),
        )


class Episode(Generic[S, O]):
    """Storage for collated episode data."""

    def __init__(self, *, states: S, observations: O, actions, rewards):
        self.states: S = states
        self.observations: O = observations
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index) -> Interaction[S, O]:
        return Interaction(
            state=(
                {k: v[index] for k, v in self.states.items()}
                if isinstance(self.states, dict)
                else self.states[index]
            ),
            observation=(
                {k: v[index] for k, v in self.observations.items()}
                if isinstance(self.observations, dict)
                else self.observations[index]
            ),
            action=self.actions[index],
            reward=self.rewards[index],
        )

    @staticmethod
    def from_interactions(interactions: Iterable[Interaction[S, O]]):
        states: S = collate_numpy(
            [interaction.state for interaction in interactions]
        )
        observations: O = collate_numpy(
            [interaction.observation for interaction in interactions]
        )
        actions = collate_numpy(
            [interaction.action for interaction in interactions]
        )
        rewards = collate_numpy(
            [interaction.reward for interaction in interactions]
        )
        return Episode(
            states=states,
            observations=observations,
            actions=actions,
            rewards=rewards,
        )

    def torch(self) -> Episode:
        checkraise(
            (
                isinstance(self.states, np.ndarray)
                or isinstance(self.states, dict)
                and all(isinstance(v, np.ndarray) for v in self.states.values())
            )
            and (
                isinstance(self.observations, np.ndarray)
                or isinstance(self.observations, dict)
                and all(
                    isinstance(v, np.ndarray)
                    for v in self.observations.values()
                )
            )
            and isinstance(self.actions, np.ndarray)
            and isinstance(self.rewards, np.ndarray),
            TypeError,
            'Episode is not numpy to begin with??',
        )
        return Episode(
            states=numpy2torch(self.states),
            observations=numpy2torch(self.observations),
            actions=numpy2torch(self.actions),
            rewards=numpy2torch(self.rewards),
        )

    def to(self, device: torch.device) -> Episode:
        return Episode(
            states=gtorch.to(self.states, device),
            observations=gtorch.to(self.observations, device),
            actions=gtorch.to(self.actions, device),
            rewards=gtorch.to(self.rewards, device),
        )


class EpisodeBuffer(Generic[S, O]):
    def __init__(self, max_timesteps: int):
        self.episodes: Deque[Episode[S, O]] = deque()
        self.max_timesteps = max_timesteps

    def num_interactions(self):
        return sum(len(episode) for episode in self.episodes)

    def num_episodes(self):
        return len(self.episodes)

    def _check_num_timesteps(self):
        num_timesteps = self.num_interactions()
        while num_timesteps > self.max_timesteps:
            episode = self.episodes.popleft()
            num_timesteps -= len(episode)

    def append_episode(self, episode: Episode[S, O]):
        self.episodes.append(episode)
        self._check_num_timesteps()

    def append_episodes(self, episodes: Sequence[Episode[S, O]]):
        for episode in episodes:
            self.episodes.append(episode)
        self._check_num_timesteps()

    def sample_episode(self) -> Episode[S, O]:
        return random.choice(self.episodes)

    def sample_episodes(
        self, *, num_samples: int, replacement: bool
    ) -> List[Episode[S, O]]:

        if replacement:
            episodes = random.choices(self.episodes, k=num_samples)

        else:
            indices = list(range(self.num_episodes()))
            random.shuffle(indices)
            episodes = [self.episodes[i] for i in indices[:num_samples]]

        return episodes

    def sample_batch(self, *, batch_size: int) -> Batch:
        states = []
        observations = []
        actions = []
        rewards = []
        next_states = []
        next_observations = []
        dones = []

        num_interactions = self.num_interactions()
        for _ in range(batch_size):
            interaction_id = random.randrange(num_interactions)
            lens = torch.tensor([len(episode) for episode in self.episodes])
            episode_id, cumlen = next(
                (i, cumlen)
                for i, cumlen in enumerate(torch.cumsum(lens, 0))
                if interaction_id < cumlen
            )

            episode = self.episodes[episode_id]
            in_episode_interaction_id = cumlen - len(episode) - interaction_id
            interaction = episode[in_episode_interaction_id]
            done = in_episode_interaction_id + 1 == len(episode)
            next_interaction = (
                None if done else episode[in_episode_interaction_id + 1]
            )

            states.append(interaction.state)
            observations.append(interaction.observation)
            actions.append(interaction.action)
            rewards.append(interaction.reward)
            next_states.append(
                gtorch.zeros_like(states[-1])
                if done
                else next_interaction.state
            )
            next_observations.append(
                gtorch.zeros_like(observations[-1])
                if done
                else next_interaction.observation
            )
            dones.append(done)

        return Batch(
            states=collate_torch(states),
            observations=collate_torch(observations),
            actions=collate_torch(actions),
            rewards=collate_torch(rewards),
            next_states=collate_torch(next_states),
            next_observations=collate_torch(next_observations),
            dones=collate_torch(dones),
        )
