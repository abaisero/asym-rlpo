import gym
import torch.nn as nn

from asym_rlpo.modules import make_module
from asym_rlpo.representations.history import GRUHistoryRepresentation
from asym_rlpo.representations.identity import IdentityRepresentation
from asym_rlpo.representations.normalization import NormalizationRepresentation
from asym_rlpo.representations.onehot import OneHotRepresentation
from asym_rlpo.representations.resize import ResizeRepresentation
from asym_rlpo.utils.config import get_config


def _make_q_model(in_size, out_size):
    return nn.Sequential(
        make_module('linear', 'relu', in_size, 512),
        nn.ReLU(),
        make_module('linear', 'relu', 512, 256),
        nn.ReLU(),
        make_module('linear', 'linear', 256, out_size),
    )


def _make_v_model(in_size):
    return nn.Sequential(
        make_module('linear', 'relu', in_size, 512),
        nn.ReLU(),
        make_module('linear', 'relu', 512, 256),
        nn.ReLU(),
        make_module('linear', 'linear', 256, 1),
    )


def _make_policy_model(in_size, out_size):
    return nn.Sequential(
        make_module('linear', 'relu', in_size, 512),
        nn.ReLU(),
        make_module('linear', 'relu', 512, 256),
        nn.ReLU(),
        make_module('linear', 'linear', 256, out_size),
        nn.LogSoftmax(dim=-1),
    )


def _make_representation_models(env: gym.Env) -> nn.ModuleDict:
    config = get_config()
    hs_features_dim: int = config.hs_features_dim
    normalize_hs_features: bool = config.normalize_hs_features

    # agent
    state_model = IdentityRepresentation(env.state_space)
    action_model = OneHotRepresentation(env.action_space)
    observation_model = IdentityRepresentation(env.observation_space)
    history_model = GRUHistoryRepresentation(
        action_model,
        observation_model,
        hidden_size=128,
    )

    # resize history and state models
    if hs_features_dim:
        history_model = ResizeRepresentation(history_model, hs_features_dim)
        state_model = ResizeRepresentation(state_model, hs_features_dim)

    # normalize history and state models
    if normalize_hs_features:
        history_model = NormalizationRepresentation(history_model)
        state_model = NormalizationRepresentation(state_model)

    return nn.ModuleDict(
        {
            'state_model': state_model,
            'action_model': action_model,
            'observation_model': observation_model,
            'history_model': history_model,
        }
    )


def make_models(  # pylint: disable=too-many-locals
    env: gym.Env,
) -> nn.ModuleDict:

    models = nn.ModuleDict(
        {
            'agent': _make_representation_models(env),
            'critic': _make_representation_models(env),
        }
    )

    # DQN models
    models.agent.update(
        {
            'qh_model': _make_q_model(
                models.agent.history_model.dim, env.action_space.n
            ),
            'qhs_model': _make_q_model(
                models.agent.history_model.dim + models.agent.state_model.dim,
                env.action_space.n,
            ),
            'qs_model': _make_q_model(
                models.agent.state_model.dim, env.action_space.n
            ),
        }
    )

    # A2C models
    models.agent.update(
        {
            'policy_model': _make_policy_model(
                models.agent.history_model.dim, env.action_space.n
            )
        }
    )
    models.critic.update(
        {
            'vh_model': _make_v_model(models.critic.history_model.dim),
            'vhs_model': _make_v_model(
                models.critic.history_model.dim + models.critic.state_model.dim
            ),
            'vs_model': _make_v_model(models.critic.state_model.dim),
        }
    )

    return models
