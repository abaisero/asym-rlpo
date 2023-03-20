import itertools as itt

import torch

from asym_rlpo.algorithms import make_a2c_algorithm
from asym_rlpo.envs import LatentType, make_env
from asym_rlpo.policies import RandomPolicy
from asym_rlpo.sampling import sample_episode


def test_compute_history_features():
    # checks that full history features and reactive history features are different

    max_episode_timesteps = 100
    env = make_env(
        'PO-pos-CartPole-v1',
        latent_type=LatentType.STATE,
        max_episode_timesteps=max_episode_timesteps,
    )
    policy = RandomPolicy(env.action_space)
    episode = sample_episode(env, policy).torch()

    algos = {
        'full': make_a2c_algorithm('a2c', env),
        'react-2': make_a2c_algorithm('a2c', env, truncated_histories_n=2),
        'react-4': make_a2c_algorithm('a2c', env, truncated_histories_n=4),
    }
    models = make_a2c_algorithm('a2c', env).models

    # this test uses implementation details (knows that
    # make_history_integrator) is build using a partial
    make_history_integrator = algos['full'].make_history_integrator
    assert make_history_integrator.keywords['truncated_histories_n'] is None
    make_history_integrator = algos['react-2'].make_history_integrator
    assert make_history_integrator.keywords['truncated_histories_n'] == 2
    make_history_integrator = algos['react-4'].make_history_integrator
    assert make_history_integrator.keywords['truncated_histories_n'] == 4

    with torch.no_grad():
        history_features = {
            k: v.compute_history_features(
                models.agent.interaction_model,
                models.agent.history_model,
                episode.actions,
                episode.observations,
            )
            for k, v in algos.items()
        }

    pairs = itt.combinations(history_features.values(), 2)
    for x, y in pairs:
        assert not torch.isclose(x, y).all()
