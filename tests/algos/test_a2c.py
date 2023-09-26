import itertools as itt

import torch

from asym_rlpo.envs import make_env
from asym_rlpo.models import make_model_factory
from asym_rlpo.policies import RandomPolicy
from asym_rlpo.sampling import sample_episode


def test_compute_history_features():
    # checks that full history features and reactive history features are different

    max_episode_timesteps = 100
    env = make_env(
        'PO-pos-CartPole-v1',
        latent_type='state',
        max_episode_timesteps=max_episode_timesteps,
    )
    policy = RandomPolicy(env.action_space)
    episode = sample_episode(env, policy).torch()

    model_factory = make_model_factory(env)
    model_factory.history_model = 'rnn'

    history_models = {}

    model_factory.history_model_memory_size = -1
    history_models['full'] = model_factory.make_history_model()

    model_factory.history_model_memory_size = 2
    history_models['react-2'] = model_factory.make_history_model()

    model_factory.history_model_memory_size = 4
    history_models['react-4'] = model_factory.make_history_model()

    history_models['react-2'].load_state_dict(
        history_models['full'].state_dict()
    )
    history_models['react-4'].load_state_dict(
        history_models['full'].state_dict()
    )

    with torch.no_grad():
        history_features = {
            k: history_model.episodic(episode)
            for k, history_model in history_models.items()
        }

    pairs = itt.combinations(history_features.values(), 2)
    for x, y in pairs:
        assert not torch.isclose(x, y).all()
