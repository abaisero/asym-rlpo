import re
from typing import Iterable, Optional

import gym
import gym_gridverse as gv
import torch.nn as nn

from asym_rlpo.modules import make_module
from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.representations.gv import (
    GV_ObservationRepresentation,
    GV_StateRepresentation,
)
from asym_rlpo.representations.history import RNNHistoryRepresentation
from asym_rlpo.representations.identity import IdentityRepresentation
from asym_rlpo.representations.mlp import MLPRepresentation
from asym_rlpo.representations.onehot import OneHotRepresentation
from asym_rlpo.utils.debugging import checkraise


def make_models(
    env: gym.Env, *, keys: Optional[Iterable[str]] = None
) -> nn.ModuleDict:

    if isinstance(env, gv.gym.GymEnvironment):
        models = make_models_gv(env)

    elif (
        re.fullmatch(r'CartPole-v\d+', env.spec.id)
        or re.fullmatch(r'Acrobot-v\d+', env.spec.id)
        or re.fullmatch(r'LunarLander-v\d+', env.spec.id)
    ):
        models = make_models_openai(env)

    else:
        raise NotImplementedError

    if keys is None:
        return models

    keys = set(keys)
    missing_keys = keys - models.keys()
    checkraise(
        len(missing_keys) == 0,
        ValueError,
        'models dictionary does not contains keys {}',
        missing_keys,
    )

    return nn.ModuleDict(
        {key: model for key, model in models.items() if key in keys}
    )


def make_models_openai(env: gym.Env) -> nn.ModuleDict:
    (input_dim,) = env.state_space.shape
    q_model = nn.Sequential(
        make_module('linear', 'leaky_relu', input_dim, 512),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, env.action_space.n),
    )

    # action_model = EmbeddingRepresentation(env.action_space.n, 128)
    # observation_model = MLPRepresentation(env.observation_space, 128)

    action_model = OneHotRepresentation(env.action_space)
    observation_model = IdentityRepresentation(env.observation_space)
    state_model = IdentityRepresentation(env.state_space)

    history_model = RNNHistoryRepresentation(
        action_model,
        observation_model,
        hidden_size=128,
        nonlinearity='tanh',
    )
    qh_model = nn.Sequential(
        make_module('linear', 'leaky_relu', history_model.dim, 512),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, env.action_space.n),
    )
    qhs_model = nn.Sequential(
        make_module(
            'linear',
            'leaky_relu',
            history_model.dim + state_model.dim,
            512,
        ),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, env.action_space.n),
    )
    return nn.ModuleDict(
        {
            'action_model': action_model,
            'observation_model': observation_model,
            'state_model': state_model,
            'history_model': history_model,
            'q_model': q_model,
            'qh_model': qh_model,
            'qhs_model': qhs_model,
        }
    )


def make_models_gv(env: gym.Env) -> nn.ModuleDict:
    action_model = EmbeddingRepresentation(env.action_space.n, 64)
    observation_model = GV_ObservationRepresentation(env.observation_space)
    state_model = GV_StateRepresentation(env.state_space)

    history_model = RNNHistoryRepresentation(
        action_model,
        observation_model,
        hidden_size=128,
    )
    q_model = nn.Sequential(
        nn.Linear(state_model.dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n),
    )
    return nn.ModuleDict(
        {
            'action_model': action_model,
            'observation_model': observation_model,
            'state_model': state_model,
            'history_model': history_model,
            'q_model': q_model,
        }
    )


# poe-dqn
# def make_models_gv(env: gym.Env) -> nn.ModuleDict:
#     action_model = EmbeddingRepresentation(env.action_space.n, 64)
#     observation_model = GV_ObservationRepresentation(env.observation_space)
#     history_model = RNNHistoryRepresentation(
#         action_model,
#         observation_model,
#         hidden_size=128,
#     )
#     q_model = nn.Sequential(
#         nn.Linear(history_model.dim, 128),
#         nn.ReLU(),
#         nn.Linear(128, 128),
#         nn.ReLU(),
#         nn.Linear(128, env.action_space.n),
#     )
#     models = nn.ModuleDict(
#         {
#             'action_model': action_model,
#             'observation_model': observation_model,
#             'history_model': history_model,
#             'q_model': q_model,
#         }
#     )
