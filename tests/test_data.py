import numpy as np

from asym_rlpo.data import Episode, EpisodeBuffer


def make_episode(num_timesteps: int):
    return Episode(
        observations=np.zeros(num_timesteps),
        latents=np.zeros(num_timesteps),
        actions=np.zeros(num_timesteps, dtype=int),
        rewards=np.zeros(num_timesteps),
        info={},
    )


def test_episode_buffer_max_timesteps():
    episode_buffer = EpisodeBuffer(100)
    assert episode_buffer.num_interactions() == 0
    assert episode_buffer.num_episodes() == 0

    episode_buffer.append_episode(make_episode(101))
    assert episode_buffer.num_interactions() == 0
    assert episode_buffer.num_episodes() == 0

    episode_buffer.append_episode(make_episode(10))
    assert episode_buffer.num_interactions() == 10
    assert episode_buffer.num_episodes() == 1

    episode_buffer.append_episode(make_episode(20))
    assert episode_buffer.num_interactions() == 30
    assert episode_buffer.num_episodes() == 2

    episode_buffer.append_episode(make_episode(80))
    assert episode_buffer.num_interactions() == 100
    assert episode_buffer.num_episodes() == 2

    episode_buffer.append_episode(make_episode(1))
    assert episode_buffer.num_interactions() == 81
    assert episode_buffer.num_episodes() == 2

    episode_buffer.append_episodes(
        [
            make_episode(10),
            make_episode(9),
        ]
    )
    assert episode_buffer.num_interactions() == 100
    assert episode_buffer.num_episodes() == 4

    episode_buffer.append_episodes(
        [
            make_episode(50),
            make_episode(50),
        ]
    )
    assert episode_buffer.num_interactions() == 100
    assert episode_buffer.num_episodes() == 2
