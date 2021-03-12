import random
from collections import deque
from typing import Deque, Generic, Optional, Sequence, TypeVar

from asym_rlpo.utils.debugging import checkraise

S = TypeVar('S')
A = TypeVar('A')
O = TypeVar('O')


class Interaction(Generic[S, A, O]):
    def __init__(
        self,
        *,
        state: S,
        observation: O,
        action: A,
        reward: float,
        start: bool,
        done: bool,
    ):
        self.state = state
        self.observation = observation
        self.action = action
        self.reward = reward
        self.start = start
        self.done = done


class Episode(Generic[S, A, O]):
    def __init__(
        self, interactions: Optional[Sequence[Interaction[S, A, O]]] = None
    ):
        self.interactions = interactions if interactions is not None else []
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


# TODO how do we want to structure this?  I assume flat list of interactions?
# Do we always assume that we give full episodes into it?  Any scenario where
# we want to add individual interactions into it?  that would be a bit iffy..
# For now, I assume only full (but possibly truncated) episodes are appended.
class EpisodeBuffer(Generic[S, A, O]):
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.interactions: Deque[Interaction[S, A, O]] = deque(
            [], maxlen=maxlen
        )

    def num_interactions(self):
        return len(self.interactions)

    def num_episodes(self):
        return sum(interaction.start for interaction in self.interactions)

    def append_episode(self, episode: Episode[S, A, O]):
        for interaction in episode.interactions:
            self.interactions.append(interaction)

    def append_episodes(self, episodes: Sequence[Episode[S, A, O]]):
        for episode in episodes:
            self.append_episode(episode)

    def sample_episode(self) -> Episode[S, A, O]:
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
        return Episode(interactions)
