import itertools as itt

import torch

from asym_rlpo.algorithms import make_a2c_algorithm
from asym_rlpo.env import make_env
from asym_rlpo.policies.random import RandomPolicy
from asym_rlpo.sampling import sample_episodes


def test_compute_history_features():
    # checks that full history features and reactive history features are
    # different

    max_episode_timesteps = 100
    env = make_env(
        'PO-pos-CartPole-v1', max_episode_timesteps=max_episode_timesteps
    )
    policy = RandomPolicy(env.action_space)
    (episode,) = sample_episodes(env, policy, num_episodes=1)
    episode = episode.torch()

    algos = {
        'full': make_a2c_algorithm(
            'a2c', env, truncated_histories=False, truncated_histories_n=-1
        ),
        'react-2': make_a2c_algorithm(
            'a2c', env, truncated_histories=True, truncated_histories_n=2
        ),
        'react-4': make_a2c_algorithm(
            'a2c', env, truncated_histories=True, truncated_histories_n=4
        ),
    }
    models = make_a2c_algorithm(
        'a2c', env, truncated_histories=False, truncated_histories_n=-1
    ).models

    assert not algos['full'].truncated_histories
    assert algos['full'].truncated_histories_n == -1
    assert algos['react-2'].truncated_histories
    assert algos['react-2'].truncated_histories_n == 2
    assert algos['react-4'].truncated_histories
    assert algos['react-4'].truncated_histories_n == 4

    with torch.no_grad():
        history_features = {
            k: v.compute_history_features(
                models.agent.action_model,
                models.agent.observation_model,
                models.agent.history_model,
                episode.actions,
                episode.observations,
            )
            for k, v in algos.items()
        }

    pairs = itt.combinations(history_features.values(), 2)
    for x, y in pairs:
        assert not torch.isclose(x, y).all()
