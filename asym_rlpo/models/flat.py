import torch.nn as nn

from asym_rlpo.envs import Environment
from asym_rlpo.modules import make_module
from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.representations.history import GRUHistoryRepresentation
from asym_rlpo.representations.interaction import InteractionRepresentation
from asym_rlpo.representations.normalization import NormalizationRepresentation
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


def _make_representation_models(env: Environment) -> nn.ModuleDict:
    config = get_config()
    hs_features_dim: int = config.hs_features_dim
    normalize_hs_features: bool = config.normalize_hs_features

    # agent
    action_model = EmbeddingRepresentation(env.action_space.n, 64)
    observation_model = EmbeddingRepresentation(
        env.observation_space.n, 64, padding_idx=-1
    )
    latent_model = EmbeddingRepresentation(env.latent_space.n, 64)
    interaction_model = InteractionRepresentation(
        action_model, observation_model
    )
    history_model = GRUHistoryRepresentation(interaction_model, hidden_size=128)

    # resize history and state models
    if hs_features_dim:
        history_model = ResizeRepresentation(history_model, hs_features_dim)
        latent_model = ResizeRepresentation(latent_model, hs_features_dim)

    # normalize history and state models
    if normalize_hs_features:
        history_model = NormalizationRepresentation(history_model)
        latent_model = NormalizationRepresentation(latent_model)

    return nn.ModuleDict(
        {
            'latent_model': latent_model,
            'action_model': action_model,
            'observation_model': observation_model,
            'history_model': history_model,
        }
    )


def make_models(env: Environment) -> nn.ModuleDict:

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
            'qhz_model': _make_q_model(
                models.agent.history_model.dim + models.agent.latent_model.dim,
                env.action_space.n,
            ),
            'qz_model': _make_q_model(
                models.agent.latent_model.dim, env.action_space.n
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
            'vhz_model': _make_v_model(
                models.critic.history_model.dim + models.critic.latent_model.dim
            ),
            'vz_model': _make_v_model(models.critic.latent_model.dim),
        }
    )

    return models
