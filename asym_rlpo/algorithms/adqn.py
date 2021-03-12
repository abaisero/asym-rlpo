from typing import Sequence

from asym_rlpo.data import Episode, EpisodeBuffer

from .base import Algorithm


class ADQN(Algorithm):
    def __init__(self):
        super().__init__()
        self.episode_buffer = EpisodeBuffer(maxlen=1_000_000)

    def process(self, episodes: Sequence[Episode]):
        self.episode_buffer.append_episodes(episodes)

        print(
            f'buffer size: {self.episode_buffer.num_interactions()}'
            f'\t#episodes: {self.episode_buffer.num_episodes()}'
        )
