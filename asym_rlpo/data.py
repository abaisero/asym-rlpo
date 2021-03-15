import random
from collections import deque
from typing import Deque, Dict, Generic, List, Optional, Sequence, TypeVar

import numpy as np

from asym_rlpo.utils.collate import collate
from asym_rlpo.utils.debugging import checkraise

S = TypeVar('S', np.ndarray, Dict[str, np.ndarray])
O = TypeVar('O', np.ndarray, Dict[str, np.ndarray])


class Interaction(Generic[S, O]):
    def __init__(
        self,
        *,
        state: S,
        observation: O,
        action: int,
        reward: float,
        start: bool,
        done: bool,
    ):
        self.state: S = state
        self.observation: O = observation
        self.action = action
        self.reward = reward
        self.start = start
        self.done = done


class RawEpisode(Generic[S, O]):
    """Storage for non-collated episode data."""

    def __init__(
        self, interactions: Optional[Sequence[Interaction[S, O]]] = None
    ):
        self.interactions: Sequence[Interaction[S, O]] = (
            interactions if interactions is not None else []
        )
        self.check()

    def __len__(self):
        return len(self.interactions)

    def check(self):
        starts = [interaction.start for interaction in self.interactions]
        dones = [interaction.done for interaction in self.interactions]

        checkraise(starts[0], ValueError, 'first `start` must be True')
        checkraise(
            not any(starts[1:]), ValueError, 'non-first `start` must be False'
        )
        checkraise(
            not any(dones[:-1]), ValueError, 'non-last `done` must be False'
        )

        # TODO should this really ever assume that the episode has terminated
        # successfully?  I don't think we can make that assumption, due to
        # episodes which are truncated based on a max number of steps.
        # checkraise(dones[-1], ValueError, 'last `done` must be True')


class Episode(Generic[S, O]):
    """Storage for collated episode data."""

    def __init__(self, episode: RawEpisode[S, O]):
        self.states: S = collate(
            [interaction.state for interaction in episode.interactions]
        )
        self.observations: O = collate(
            [interaction.observation for interaction in episode.interactions]
        )
        self.actions = collate(
            [interaction.action for interaction in episode.interactions]
        )
        self.rewards = collate(
            [interaction.reward for interaction in episode.interactions]
        )
        self.starts = collate(
            [interaction.start for interaction in episode.interactions]
        )
        self.dones = collate(
            [interaction.done for interaction in episode.interactions]
        )

    def __len__(self):
        return len(self.dones)


# TODO how do we want to structure this?  I assume flat list of interactions?
# Do we always assume that we give full episodes into it?  Any scenario where
# we want to add individual interactions into it?  that would be a bit iffy..
# For now, I assume only full (but possibly truncated) episodes are appended.

# TODO actually flat list of interactions is more annoying than it needs to
# be..  especially if we're going to sample full episodes, we should directly
# use lists of episodes (as done in EpisodeBuffer2)
class RawEpisodeBuffer(Generic[S, O]):
    def __init__(self, maxlen: int):
        self.interactions: Deque[Interaction[S, O]] = deque(maxlen=maxlen)

    def num_interactions(self):
        return len(self.interactions)

    def num_episodes(self):
        return sum(interaction.start for interaction in self.interactions)

    def append_episode(self, episode: RawEpisode[S, O]):
        for interaction in episode.interactions:
            self.interactions.append(interaction)

    def append_episodes(self, episodes: Sequence[RawEpisode[S, O]]):
        for episode in episodes:
            self.append_episode(episode)

    def sample_episode(self) -> RawEpisode[S, O]:
        start_indices = [
            i
            for i, interaction in enumerate(self.interactions)
            if interaction.start
        ]

        # sampling start interaction
        i = random.randrange(len(start_indices))

        start_indices.append(len(self.interactions))
        index_start, index_end = start_indices[i], start_indices[i + 1]

        interactions = [
            self.interactions[i] for i in range(index_start, index_end)
        ]
        return RawEpisode(interactions)


class EpisodeBuffer(Generic[S, O]):
    def __init__(self, maxlen: int):
        self.episodes: Deque[Episode[S, O]] = deque(maxlen=maxlen)

    def num_interactions(self):
        return sum(len(episode) for episode in self.episodes)

    def num_episodes(self):
        return len(self.episodes)

    def append_episode(self, episode: Episode[S, O]):
        self.episodes.append(episode)

    def append_episodes(self, episodes: Sequence[Episode[S, O]]):
        for episode in episodes:
            self.append_episode(episode)

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
