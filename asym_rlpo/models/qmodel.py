import abc

import gym.spaces
import torch

from asym_rlpo.data import Episode
from asym_rlpo.models.history import HistoryModel
from asym_rlpo.models.model import Model
from asym_rlpo.models.types import QModule
from asym_rlpo.policies import EpsilonGreedyHistoryPolicy, GreedyHistoryPolicy
from asym_rlpo.types import ActionValueFunction


class QModel(Model):
    def __init__(self, action_space: gym.spaces.Discrete):
        super().__init__()
        self.action_space = action_space

    @property
    def dim(self):
        return 1

    @abc.abstractmethod
    def values(self, episode: Episode) -> torch.Tensor:
        assert False


class QhaModel(QModel):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        history_model: HistoryModel,
        value_module: QModule,
    ):
        super().__init__(action_space)
        self.history_model = history_model
        self.value_module = value_module

    def values(self, episode: Episode) -> torch.Tensor:
        history_features = self.history_model.episodic(episode)
        return self.value_module(history_features).squeeze(-1)

    def value_function(self) -> ActionValueFunction:
        return self.value_module

    def policy(self) -> GreedyHistoryPolicy:
        return GreedyHistoryPolicy(
            self.history_model.make_history_integrator(),
            self.value_function(),
        )

    def epsilon_greedy_policy(self) -> EpsilonGreedyHistoryPolicy:
        return EpsilonGreedyHistoryPolicy(
            self.history_model.make_history_integrator(),
            self.value_function(),
            self.action_space,
        )


class QzaModel(QModel):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        latent_model: Model,
        value_module: QModule,
    ):
        super().__init__(action_space)
        self.latent_model = latent_model
        self.value_module = value_module

    def values(self, episode: Episode) -> torch.Tensor:
        latent_features = self.latent_model(episode.latents)
        return self.value_module(latent_features).squeeze(-1)


class QhzaModel(QModel):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        history_model: HistoryModel,
        latent_model: Model,
        value_module: QModule,
    ):
        super().__init__(action_space)
        self.history_model = history_model
        self.latent_model = latent_model
        self.value_module = value_module

    def values(self, episode: Episode) -> torch.Tensor:
        history_features = self.history_model.episodic(episode)
        latent_features = self.latent_model(episode.latents)
        input_features = torch.cat([history_features, latent_features], dim=-1)
        return self.value_module(input_features).squeeze(-1)
