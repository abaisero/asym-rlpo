from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
import torch

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.utils.collate import collate_numpy
from asym_rlpo.utils.convert import numpy2torch

logger = logging.getLogger(__name__)

TorchObservation = TypeVar(
    'TorchObservation',
    torch.Tensor,
    dict[str, torch.Tensor],
)
TorchLatent = TypeVar(
    'TorchLatent',
    torch.Tensor,
    dict[str, torch.Tensor],
)

Observation = TypeVar(
    'Observation',
    torch.Tensor,
    dict[str, torch.Tensor],
    np.ndarray,
    dict[str, np.ndarray],
)
Latent = TypeVar(
    'Latent',
    torch.Tensor,
    dict[str, torch.Tensor],
    np.ndarray,
    dict[str, np.ndarray],
)


@dataclass
class Interaction(Generic[Observation, Latent]):
    observation: Observation
    latent: Latent
    action: int
    reward: float
    info: dict


class Episode(Generic[Observation, Latent]):
    """Storage for collated episode data."""

    def __init__(
        self,
        *,
        observations: Observation,
        latents: Latent,
        actions,
        rewards,
        info: dict,
    ):
        self.observations: Observation = observations
        self.latents: Latent = latents
        self.actions = actions
        self.rewards = rewards
        self.info = info

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
        info = collate_numpy([interaction.info for interaction in interactions])
        return Episode(
            observations=observations,
            latents=latents,
            actions=actions,
            rewards=rewards,
            info=info,
        )

    def torch(self) -> Episode[TorchObservation, TorchLatent]:
        if not (
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
            and isinstance(self.rewards, np.ndarray)
        ):
            raise TypeError('Episode is not numpy to begin with??')

        return Episode(
            observations=numpy2torch(self.observations),
            latents=numpy2torch(self.latents),
            actions=numpy2torch(self.actions),
            rewards=numpy2torch(self.rewards),
            info=numpy2torch(self.info),
        )

    def to(self, device: torch.device) -> Episode[Observation, Latent]:
        return Episode(
            observations=gtorch.to(self.observations, device),
            latents=gtorch.to(self.latents, device),
            actions=gtorch.to(self.actions, device),
            rewards=gtorch.to(self.rewards, device),
            info=gtorch.to(self.info, device),
        )


class EpisodeBuilder(Generic[Observation, Latent]):
    def __init__(self):
        self.interactions: list[Interaction[Observation, Latent]] = []
        self.done = False

    def append(
        self,
        interaction: Interaction[Observation, Latent],
        done: bool,
    ):
        self.interactions.append(interaction)
        self.done |= done

    def build(self) -> Episode[Observation, Latent]:
        if not self.done:
            raise RuntimeError('Cannot build incomplete episode')

        return Episode.from_interactions(self.interactions)


class EpisodeBuffer(Generic[Observation, Latent]):
    def __init__(self, max_timesteps: int):
        self.episodes = deque()
        self.__max_timesteps = max_timesteps
        self.__num_interactions = 0

    def num_episodes(self) -> int:
        return len(self.episodes)

    def num_interactions(self) -> int:
        return self.__num_interactions

    def __getitem__(self, i) -> Episode[Observation, Latent]:
        return self.episodes[i]

    def _enforce_max_timesteps(self) -> bool:
        max_timesteps_exceeded = self.__max_timesteps < self.num_interactions()

        while self.__max_timesteps < self.num_interactions():
            self.pop_episode()

        return max_timesteps_exceeded

    def append_episode(self, episode: Episode[Observation, Latent]) -> bool:
        self.episodes.append(episode)
        self.__num_interactions += len(episode)

        return self._enforce_max_timesteps()

    def append_episodes(
        self,
        episodes: Sequence[Episode[Observation, Latent]],
    ) -> bool:
        for episode in episodes:
            self.episodes.append(episode)
            self.__num_interactions += len(episode)

        return self._enforce_max_timesteps()

    def pop_episode(self) -> Episode[Observation, Latent]:
        episode = self.episodes.popleft()
        self.__num_interactions -= len(episode)
        return episode


class EpisodeBufferSampler(Generic[Observation, Latent]):
    def __init__(self, episode_buffer: EpisodeBuffer[Observation, Latent]):
        super().__init__()
        self.episode_buffer = episode_buffer

    def sample_episode(self) -> Episode[Observation, Latent]:
        if self.episode_buffer.num_episodes == 0:
            raise ValueError('Cannot sample from empty episode buffer')

        i = random.randrange(self.episode_buffer.num_episodes())
        return self.episode_buffer[i]

    def sample_episodes(
        self,
        num_samples: int,
        *,
        replacement: bool,
    ) -> list[Episode[Observation, Latent]]:
        if self.episode_buffer.num_episodes == 0:
            raise ValueError('Cannot sample from empty episode buffer')

        if not replacement and num_samples > self.episode_buffer.num_episodes():
            raise ValueError(
                f'Cannot sample {num_samples} episodes from an episode buffer'
                f' that contains only {self.episode_buffer.num_episodes()} episodes'
            )

        indices = list(range(self.episode_buffer.num_episodes()))

        if replacement:
            indices = random.choices(indices, k=num_samples)

        else:
            random.shuffle(indices)
            indices = indices[:num_samples]

        return [self.episode_buffer[i] for i in indices]


class EpisodeFactory(Protocol, Generic[Observation, Latent]):
    def __call__(self) -> Episode[Observation, Latent]:
        ...


class EpisodesFactory(Protocol, Generic[Observation, Latent]):
    def __call__(self) -> list[Episode[Observation, Latent]]:
        ...


def populate_episode_buffer(
    episode_buffer: EpisodeBuffer[Observation, Latent],
    episode_factory: EpisodeFactory[Observation, Latent],
    *,
    timesteps: int,
):
    logger.info(f'populating episode buffer until {timesteps:_} timesteps...')
    while episode_buffer.num_interactions() < timesteps:
        episode = episode_factory().torch()
        if episode_buffer.append_episode(episode):
            break

        logger.debug(
            f'- episode buffer {episode_buffer.num_interactions():_} timesteps'
        )
    logger.info('populating episode buffer DONE')
