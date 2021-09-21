import itertools as itt

import torch

from asym_rlpo.algorithms import make_a2c_algorithm
from asym_rlpo.env import make_env
from asym_rlpo.features import (
    FullHistoryIntegrator,
    TruncatedHistoryIntegrator,
    make_history_integrator,
)
from asym_rlpo.policies.random import RandomPolicy
from asym_rlpo.sampling import sample_episodes


def test_history_integrators():
    # checks that full history integrator and reactive history integrator
    # features are different

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

    history_integrators = {
        k: make_history_integrator(
            models.agent.action_model,
            models.agent.observation_model,
            models.agent.history_model,
            truncated_histories=v.truncated_histories,
            truncated_histories_n=v.truncated_histories_n,
        )
        for k, v in algos.items()
    }

    assert isinstance(history_integrators['full'], FullHistoryIntegrator)
    assert isinstance(
        history_integrators['react-2'], TruncatedHistoryIntegrator
    )
    assert history_integrators['react-2'].n == 2
    assert isinstance(
        history_integrators['react-4'], TruncatedHistoryIntegrator
    )
    assert history_integrators['react-4'].n == 4

    for history_integrator in history_integrators.values():
        history_integrator.reset(episode.observations[0])

    history_features = (v.features for v in history_integrators.values())
    pairs = itt.combinations(history_features, 2)
    for x, y in pairs:
        assert not torch.isclose(x, y).all()

    for t in range(1, len(episode)):
        for history_integrator in history_integrators.values():
            history_integrator.step(
                episode.actions[t - 1], episode.observations[t]
            )

        # full and react-n are the same at the nth timestep
        if t not in (1, 3):
            history_features = (
                v.features for v in history_integrators.values()
            )
            pairs = itt.combinations(history_features, 2)
            for x, y in pairs:
                assert not torch.isclose(x, y).all()
