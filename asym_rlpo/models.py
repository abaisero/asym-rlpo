import re
from typing import Iterable, Optional

import gym
import gym_gridverse as gv
import gym_pomdps
import torch.nn as nn

from asym_rlpo.modules import make_module
from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.representations.gv import (
    GV_ObservationRepresentation,
    GV_StateRepresentation,
)
from asym_rlpo.representations.history import GRUHistoryRepresentation
from asym_rlpo.representations.identity import IdentityRepresentation
from asym_rlpo.representations.mlp import MLPRepresentation
from asym_rlpo.representations.onehot import OneHotRepresentation
from asym_rlpo.utils.debugging import checkraise


def make_models(
    env: gym.Env, *, keys: Optional[Iterable[str]] = None
) -> nn.ModuleDict:

    if isinstance(env.unwrapped, gv.gym.GymEnvironment):
        models = make_models_gv(env)

    elif (
        re.fullmatch(r'CartPole-v\d+', env.spec.id)
        or re.fullmatch(r'Acrobot-v\d+', env.spec.id)
        or re.fullmatch(r'LunarLander-v\d+', env.spec.id)
    ):
        models = make_models_openai(env)

    elif isinstance(env.unwrapped, gym_pomdps.POMDP):
        models = make_models_flat(env)

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


def make_models_flat(env: gym.Env) -> nn.ModuleDict:
    # gen purpose models
    state_model = EmbeddingRepresentation(env.state_space.n, 64)
    action_model = EmbeddingRepresentation(env.action_space.n, 64)
    observation_model = EmbeddingRepresentation(
        env.observation_space.n, 64, padding_idx=-1
    )
    history_model = GRUHistoryRepresentation(
        action_model,
        observation_model,
        hidden_size=128,
    )

    # DQN models
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
    qs_model = nn.Sequential(
        make_module('linear', 'leaky_relu', state_model.dim, 512),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, env.action_space.n),
    )

    # A2C models
    policy_model = nn.Sequential(
        make_module('linear', 'leaky_relu', history_model.dim, 512),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, env.action_space.n),
        nn.LogSoftmax(dim=-1),
    )
    vh_model = nn.Sequential(
        make_module('linear', 'leaky_relu', history_model.dim, 512),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, 1),
    )
    vhs_model = nn.Sequential(
        make_module(
            'linear', 'leaky_relu', history_model.dim + state_model.dim, 512
        ),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, 1),
    )

    return nn.ModuleDict(
        {
            # GENERIC
            'state_model': state_model,
            'action_model': action_model,
            'observation_model': observation_model,
            'history_model': history_model,
            # DQN
            'qs_model': qs_model,
            'qh_model': qh_model,
            'qhs_model': qhs_model,
            # A2C
            'policy_model': policy_model,
            'vh_model': vh_model,
            'vhs_model': vhs_model,
        }
    )


def make_models_openai(env: gym.Env) -> nn.ModuleDict:
    # gen purpose models
    state_model = IdentityRepresentation(env.state_space)
    action_model = OneHotRepresentation(env.action_space)
    observation_model = IdentityRepresentation(env.observation_space)
    history_model = GRUHistoryRepresentation(
        action_model,
        observation_model,
        hidden_size=128,
    )

    # DQN models
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
    qs_model = nn.Sequential(
        make_module('linear', 'leaky_relu', state_model.dim, 512),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, env.action_space.n),
    )

    # A2C models
    policy_model = nn.Sequential(
        make_module('linear', 'leaky_relu', history_model.dim, 512),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, env.action_space.n),
        nn.LogSoftmax(dim=-1),
    )
    vh_model = nn.Sequential(
        make_module('linear', 'leaky_relu', history_model.dim, 512),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, 1),
    )
    vhs_model = nn.Sequential(
        make_module(
            'linear', 'leaky_relu', history_model.dim + state_model.dim, 512
        ),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, 1),
    )

    return nn.ModuleDict(
        {
            # GENERIC
            'state_model': state_model,
            'action_model': action_model,
            'observation_model': observation_model,
            'history_model': history_model,
            # DQN
            'qs_model': qs_model,
            'qh_model': qh_model,
            'qhs_model': qhs_model,
            # A2C
            'policy_model': policy_model,
            'vh_model': vh_model,
            'vhs_model': vhs_model,
        }
    )


def make_models_gv(env: gym.Env) -> nn.ModuleDict:
    # gen purpose models
    state_model = GV_StateRepresentation(env.state_space)
    action_model = EmbeddingRepresentation(env.action_space.n, 1)
    observation_model = GV_ObservationRepresentation(env.observation_space)
    history_model = GRUHistoryRepresentation(
        action_model,
        observation_model,
        hidden_size=64,
    )

    # DQN models
    def make_q_model(in_size):
        return nn.Sequential(
            make_module('linear', 'relu', in_size, 512),
            nn.ReLU(),
            make_module('linear', 'linear', 512, env.action_space.n),
        )

    qh_model = make_q_model(history_model.dim)
    qhs_model = make_q_model(history_model.dim + state_model.dim)
    qs_model = make_q_model(state_model.dim)

    # A2C models
    policy_model = nn.Sequential(
        make_module('linear', 'leaky_relu', history_model.dim, 128),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 128, 128),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 128, env.action_space.n),
        nn.LogSoftmax(dim=-1),
    )
    vh_model = nn.Sequential(
        make_module('linear', 'leaky_relu', history_model.dim, 128),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 128, 128),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 128, 1),
    )
    vhs_model = nn.Sequential(
        make_module(
            'linear', 'leaky_relu', history_model.dim + state_model.dim, 128
        ),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 128, 128),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 128, 1),
    )

    return nn.ModuleDict(
        {
            # GENERIC
            'state_model': state_model,
            'action_model': action_model,
            'observation_model': observation_model,
            'history_model': history_model,
            # DQN
            'qs_model': qs_model,
            'qh_model': qh_model,
            'qhs_model': qhs_model,
            # A2C
            'policy_model': policy_model,
            'vh_model': vh_model,
            'vhs_model': vhs_model,
        }
    )
