import abc

import torch

from asym_rlpo.data import Episode
from asym_rlpo.models.history import HistoryModel
from asym_rlpo.models.memory import MemoryModel
from asym_rlpo.models.model import Model
from asym_rlpo.models.types import UModule, VModule
from asym_rlpo.types import Features, Values


class CriticModel(Model):
    @property
    def dim(self):
        return 1

    @abc.abstractmethod
    def values(self, episode: Episode) -> Values:
        assert False


class H_CriticModel(CriticModel):
    def __init__(
        self,
        history_model: HistoryModel,
        value_module: VModule,
    ):
        super().__init__()
        self.history_model = history_model
        self.value_module = value_module

    def values(self, episode: Episode) -> Values:
        history_features = self.history_model.episodic(episode)
        return self.value_module(history_features).squeeze(-1)


class Z_CriticModel(CriticModel):
    def __init__(
        self,
        latent_model: Model,
        value_module: VModule,
    ):
        super().__init__()
        self.latent_model = latent_model
        self.value_module = value_module

    def values(self, episode: Episode) -> Values:
        latent_features = self.latent_model(episode.latents)
        return self.value_module(latent_features).squeeze(-1)


class HZ_CriticModel(CriticModel):
    def __init__(
        self,
        history_model: HistoryModel,
        latent_model: Model,
        value_module: VModule,
    ):
        super().__init__()
        self.history_model = history_model
        self.latent_model = latent_model
        self.value_module = value_module

    def values(self, episode: Episode) -> Values:
        history_features = self.history_model.episodic(episode)
        latent_features = self.latent_model(episode.latents)
        input_features = torch.cat([history_features, latent_features], dim=-1)
        return self.value_module(input_features).squeeze(-1)


class HM_CriticModel(CriticModel):
    def __init__(
        self,
        history_model: HistoryModel,
        memory_model: MemoryModel,
        value_module: UModule,
    ):
        super().__init__()
        self.history_model = history_model
        self.memory_model = memory_model
        self.value_module = value_module

    def values(self, episode: Episode) -> Values:
        history_features = self.history_model.episodic(episode)
        memory_features = self.memory_model.episodic(
            episode,
            select_memories=True,
        )
        return self(history_features, memory_features)

    def max_memory_values(self, episode: Episode) -> Values:
        history_features = self.history_model.episodic(episode)
        memory_features = self.memory_model.episodic(
            episode,
            select_memories=False,
        )

        length = history_features.size(0)
        expanded_shape = (length, length, -1)
        history_features = history_features.unsqueeze(0).expand(expanded_shape)
        memory_features = memory_features.unsqueeze(1).expand(expanded_shape)
        history_memory_features = torch.cat(
            [history_features, memory_features],
            dim=-1,
        )

        values = self.value_module(history_memory_features).squeeze(-1)
        condition = torch.ones_like(values, dtype=torch.bool).tril()
        return torch.where(condition, values, float('-inf')).max(1).values

    def forward(
        self,
        history_features: Features,
        memory_features: Features,
    ) -> Values:
        assert history_features.ndim in [1, 2]
        assert memory_features.ndim == 2

        # broadcasting case
        if history_features.ndim == 1:
            batch_size = memory_features.size(0)
            history_features = history_features.expand(batch_size, -1)

        history_memory_features = torch.cat(
            [history_features, memory_features],
            dim=-1,
        )
        return self.value_module(history_memory_features).squeeze(-1)
