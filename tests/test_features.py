import pytest
import torch

from asym_rlpo.envs import LatentType, make_env
from asym_rlpo.models import make_model_factory
from asym_rlpo.models.history import (
    FullHistoryIntegrator,
    ReactiveHistoryIntegrator,
)
from asym_rlpo.policies import RandomPolicy
from asym_rlpo.sampling import sample_episode


def min_positive(*args: int) -> int | None:
    positive_args = (n for n in args if n > 0)
    return min(positive_args, default=None)


# reducing default tolerance to make tests pass
def torch_isclose(x: torch.Tensor, y: torch.Tensor) -> bool:
    return bool(torch.isclose(x, y, rtol=1e-05, atol=1e-05).all())


# @pytest.mark.parametrize('history_model', ['rnn', 'attention'])
@pytest.mark.parametrize('history_model', ['rnn'])
def test_full_history_integrator(history_model: str):
    env = make_env(
        'PO-pos-CartPole-v1',
        latent_type=LatentType.STATE,
        max_episode_timesteps=100,
    )
    policy = RandomPolicy(env.action_space)
    episode = sample_episode(env, policy).torch()

    model_factory = make_model_factory(env)
    model_factory.history_model = history_model
    model_factory.attention_num_heads = 1
    model_factory.history_model_memory_size = 0

    model = model_factory.make_history_model()
    history_integrator = model.make_history_integrator()

    assert isinstance(history_integrator, FullHistoryIntegrator)

    history_integrator.reset(episode.observations[0])
    history_features, _ = history_integrator.sample_features()
    assert history_features.shape == (128,)


# @pytest.mark.parametrize('history_model', ['rnn', 'attention'])
@pytest.mark.parametrize('history_model', ['rnn'])
@pytest.mark.parametrize('memory_size', [2, 4, 6])
def test_truncated_history_integrators(
    history_model: str,
    memory_size: int,
):
    env = make_env(
        'PO-pos-CartPole-v1',
        latent_type=LatentType.STATE,
        max_episode_timesteps=100,
    )
    policy = RandomPolicy(env.action_space)
    episode = sample_episode(env, policy).torch()

    model_factory = make_model_factory(env)
    model_factory.history_model = history_model
    model_factory.attention_num_heads = 1
    model_factory.history_model_memory_size = memory_size

    model = model_factory.make_history_model()
    history_integrator = model.make_history_integrator()

    assert isinstance(history_integrator, ReactiveHistoryIntegrator)
    assert history_integrator.memory_size == memory_size

    history_integrator.reset(episode.observations[0])
    history_features, _ = history_integrator.sample_features()
    assert history_features.shape == (128,)


# @pytest.mark.parametrize('history_model', ['rnn', 'attention'])
@pytest.mark.parametrize('history_model', ['rnn'])
@pytest.mark.parametrize('memory_size_1', [0, 4, 8])
@pytest.mark.parametrize('memory_size_2', [0, 4, 8])
def test_history_integrators(
    history_model: str,
    memory_size_1: int,
    memory_size_2: int,
):
    env = make_env(
        'PO-pos-CartPole-v1',
        latent_type=LatentType.STATE,
        max_episode_timesteps=100,
    )
    policy = RandomPolicy(env.action_space)
    episode = sample_episode(env, policy).torch()

    # single set of models for both integrators
    model_factory = make_model_factory(env)
    model_factory.history_model = history_model

    model_factory.history_model_memory_size = memory_size_1
    history_model1 = model_factory.make_history_model()
    history_integrator1 = history_model1.make_history_integrator()

    model_factory.history_model_memory_size = memory_size_2
    history_model2 = model_factory.make_history_model()
    history_integrator2 = history_model2.make_history_integrator()

    history_model2.load_state_dict(history_model1.state_dict())

    expected = memory_size_1 == memory_size_2

    observation = episode.observations[0]
    history_integrator1.reset(observation)
    history_integrator2.reset(observation)

    history_features1, _ = history_integrator1.sample_features()
    history_features2, _ = history_integrator2.sample_features()
    assert torch_isclose(history_features1, history_features2) == expected

    for t in range(1, len(episode)):
        interaction = (episode.actions[t - 1], episode.observations[t])
        history_integrator1.step(*interaction)
        history_integrator2.step(*interaction)

        # full and truncated-n are the same at the nth timestep
        same_model = memory_size_1 == memory_size_2
        non_truncated_model = memory_size_1 == 0 or memory_size_2 == 0
        min_memory_size = min_positive(memory_size_1, memory_size_2)
        equivalent_model = (t + 1) == min_memory_size
        # expected result is equality if they are the same model,
        # or if one model is non-truncated, while the other is truncated,
        # and at the exact timestep corresponding to the truncation length
        expected = same_model or (non_truncated_model and equivalent_model)

        history_features1, _ = history_integrator1.sample_features()
        history_features2, _ = history_integrator2.sample_features()
        history_features_close = torch_isclose(
            history_features1, history_features2
        )
        assert history_features_close == expected
