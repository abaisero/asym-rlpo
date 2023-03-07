from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, Generic, Iterable, List, Sequence, TypeVar

import numpy as np
import torch

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.utils.collate import collate_numpy
from asym_rlpo.utils.convert import numpy2torch
from asym_rlpo.utils.debugging import checkraise

TorchObservation = TypeVar(
    'TorchObservation',
    torch.Tensor,
    Dict[str, torch.Tensor],
)
TorchLatent = TypeVar(
    'TorchLatent',
    torch.Tensor,
    Dict[str, torch.Tensor],
)

Observation = TypeVar(
    'Observation',
    torch.Tensor,
    Dict[str, torch.Tensor],
    np.ndarray,
    Dict[str, np.ndarray],
)
Latent = TypeVar(
    'Latent',
    torch.Tensor,
    Dict[str, torch.Tensor],
    np.ndarray,
    Dict[str, np.ndarray],
)


class Interaction(Generic[Observation, Latent]):
    def __init__(
        self,
        *,
        observation: Observation,
        latent: Latent,
        action: int,
        reward: float,
    ):
        self.observation: Observation = observation
        self.latent: Latent = latent
        self.action = action
        self.reward = reward


class Episode(Generic[Observation, Latent]):
    """Storage for collated episode data."""

    def __init__(
        self, *, observations: Observation, latents: Latent, actions, rewards
    ):
        self.observations: Observation = observations
        self.latents: Latent = latents
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index) -> Interaction[Observation, Latent]:
        return Interaction(
            observation=(
                {k: v[index] for k, v in self.observations.items()}
                if isinstance(self.observations, dict)
                else self.observations[index]
            ),
            latent=(
                {k: v[index] for k, v in self.latents.items()}
                if isinstance(self.latents, dict)
                else self.latents[index]
            ),
            action=self.actions[index],
            reward=self.rewards[index],
        )

    @staticmethod
    def from_interactions(
        interactions: Iterable[Interaction[Observation, Latent]]
    ) -> Episode[Observation, Latent]:
        observations: Observation = collate_numpy(
            [interaction.observation for interaction in interactions]
        )
        latents: Latent = collate_numpy(
            [interaction.latent for interaction in interactions]
        )
        actions = collate_numpy(
            [interaction.action for interaction in interactions]
        )
        rewards = collate_numpy(
            [interaction.reward for interaction in interactions]
        )
        return Episode(
            observations=observations,
            latents=latents,
            actions=actions,
            rewards=rewards,
        )

    def torch(self) -> Episode[TorchObservation, TorchLatent]:
        checkraise(
            (
                isinstance(self.observations, np.ndarray)
                or isinstance(self.observations, dict)
                and all(
                    isinstance(v, np.ndarray)
                    for v in self.observations.values()
                )
            )
            and (
                isinstance(self.latents, np.ndarray)
                or isinstance(self.latents, dict)
                and all(
                    isinstance(v, np.ndarray) for v in self.latents.values()
                )
            )
            and isinstance(self.actions, np.ndarray)
            and isinstance(self.rewards, np.ndarray),
            TypeError,
            'Episode is not numpy to begin with??',
        )
        return Episode(
            observations=numpy2torch(self.observations),
            latents=numpy2torch(self.latents),
            actions=numpy2torch(self.actions),
            rewards=numpy2torch(self.rewards),
        )

    def to(self, device: torch.device) -> Episode[Observation, Latent]:
        return Episode(
            observations=gtorch.to(self.observations, device),
            latents=gtorch.to(self.latents, device),
            actions=gtorch.to(self.actions, device),
            rewards=gtorch.to(self.rewards, device),
        )


class EpisodeBuffer(Generic[Observation, Latent]):
    def __init__(self, max_timesteps: int):
        self.episodes: Deque[Episode[Observation, Latent]] = deque()
        self.max_timesteps = max_timesteps
        self.__num_interactions = 0

    def num_episodes(self):
        return len(self.episodes)

    def num_interactions(self):
        return self.__num_interactions

    def _enforce_max_timesteps(self):
        while self.num_interactions() > self.max_timesteps:
            self.pop_episode()

    def append_episode(self, episode: Episode[Observation, Latent]):
        self.episodes.append(episode)
        self.__num_interactions += len(episode)
        self._enforce_max_timesteps()

    def append_episodes(self, episodes: Sequence[Episode[Observation, Latent]]):
        for episode in episodes:
            self.episodes.append(episode)
            self.__num_interactions += len(episode)
        self._enforce_max_timesteps()

    def pop_episode(self) -> Episode[Observation, Latent]:
        episode = self.episodes.popleft()
        self.__num_interactions -= len(episode)
        return episode

    def sample_episode(self) -> Episode[Observation, Latent]:
        return random.choice(self.episodes)

    def sample_episodes(
        self, num_samples: int, *, replacement: bool
    ) -> List[Episode[Observation, Latent]]:
        if replacement:
            return random.choices(self.episodes, k=num_samples)

        indices = list(range(self.num_episodes()))
        random.shuffle(indices)
        return [self.episodes[i] for i in indices[:num_samples]]
