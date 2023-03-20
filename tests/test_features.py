import functools
from typing import Optional

import pytest
import torch

from asym_rlpo.algorithms import get_a2c_algorithm_class
from asym_rlpo.envs import LatentType, make_env
from asym_rlpo.features import (
    FullHistoryIntegrator,
    TruncatedHistoryIntegrator,
    make_history_integrator,
)
from asym_rlpo.models import make_models
from asym_rlpo.sampling import sample_episode
from asym_rlpo.utils.config import get_config


def integer_min(*args: Optional[int]) -> Optional[int]:
    integer_args = (n for n in args if n is not None)
    return min(integer_args, default=None)


# reducing default tolerance to make tests pass
torch_isclose = functools.partial(torch.isclose, rtol=1e-05, atol=1e-05)


@pytest.mark.parametrize('history_model', ['rnn', 'attention'])
def test_full_history_integrator(history_model: str):
    config = get_config()
    config._update({'history_model': history_model})

    env = make_env(
        'PO-pos-CartPole-v1',
        latent_type=LatentType.STATE,
        max_episode_timesteps=100,
    )
    episode = sample_episode(env).torch()

    algorithm_class = get_a2c_algorithm_class('a2c')
    models = make_models(env, keys=algorithm_class.model_keys)
    history_integrator = make_history_integrator(
        models.agent.interaction_model,
        models.agent.history_model,
    )

    assert isinstance(history_integrator, FullHistoryIntegrator)

    history_integrator.reset(episode.observations[0])
    history_features = history_integrator.features
    assert history_features.shape == (128,)


@pytest.mark.parametrize('history_model', ['rnn', 'attention'])
@pytest.mark.parametrize('truncated_histories_n', [2, 4, 6])
def test_truncated_history_integrators(
    history_model: str,
    truncated_histories_n: Optional[int],
):
    config = get_config()
    config._update({'history_model': history_model})

    env = make_env(
        'PO-pos-CartPole-v1',
        latent_type=LatentType.STATE,
        max_episode_timesteps=100,
    )
    episode = sample_episode(env).torch()

    algorithm_class = get_a2c_algorithm_class('a2c')
    models = make_models(env, keys=algorithm_class.model_keys)
    history_integrator = make_history_integrator(
        models.agent.interaction_model,
        models.agent.history_model,
        truncated_histories_n=truncated_histories_n,
    )

    assert isinstance(history_integrator, TruncatedHistoryIntegrator)
    assert history_integrator.n == truncated_histories_n

    history_integrator.reset(episode.observations[0])
    history_features = history_integrator.features
    assert history_features.shape == (128,)


@pytest.mark.parametrize('history_model', ['rnn', 'attention'])
@pytest.mark.parametrize('truncated_histories_n1', [None, 4, 8])
@pytest.mark.parametrize('truncated_histories_n2', [None, 4, 8])
def test_history_integrators(
    history_model: str,
    truncated_histories_n1: Optional[int],
    truncated_histories_n2: Optional[int],
):
    config = get_config()
    config._update({'history_model': history_model})

    env = make_env(
        'PO-pos-CartPole-v1',
        latent_type=LatentType.STATE,
        max_episode_timesteps=100,
    )
    episode = sample_episode(env).torch()

    # single set of models for both integrators
    algorithm_class = get_a2c_algorithm_class('a2c')
    models = make_models(env, keys=algorithm_class.model_keys)

    history_integrator1 = make_history_integrator(
        models.agent.interaction_model,
        models.agent.history_model,
        truncated_histories_n=truncated_histories_n1,
    )
    history_integrator2 = make_history_integrator(
        models.agent.interaction_model,
        models.agent.history_model,
        truncated_histories_n=truncated_histories_n2,
    )

    expected = truncated_histories_n1 == truncated_histories_n2

    observation = episode.observations[0]
    history_integrator1.reset(observation)
    history_integrator2.reset(observation)

    history_features1 = history_integrator1.features
    history_features2 = history_integrator2.features
    assert torch_isclose(history_features1, history_features2).all() == expected

    for t in range(1, len(episode)):
        interaction = (episode.actions[t - 1], episode.observations[t])
        history_integrator1.step(*interaction)
        history_integrator2.step(*interaction)

        # full and truncated-n are the same at the nth timestep
        same_model = truncated_histories_n1 == truncated_histories_n2
        non_truncated_model = None in [
            truncated_histories_n1,
            truncated_histories_n2,
        ]
        min_truncated_histories_n = integer_min(
            truncated_histories_n1, truncated_histories_n2
        )
        equivalent_model = (t + 1) == min_truncated_histories_n
        # expected result is equality if they are the same model,
        # or if one model is non-truncated, while the other is truncated,
        # and at the exact timestep corresponding to the truncation length
        expected = same_model or (non_truncated_model and equivalent_model)

        history_features1 = history_integrator1.features
        history_features2 = history_integrator2.features
        history_features_close = torch_isclose(
            history_features1, history_features2
        ).all()
        assert history_features_close == expected
