from typing import Any

import torch

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import TorchObservation
from asym_rlpo.models.history.integrator import HistoryIntegrator
from asym_rlpo.models.interaction import InteractionModel
from asym_rlpo.models.sequence import SequenceModel
from asym_rlpo.types import Features


def compute_full_history_features(
    sequence_model: SequenceModel,
    interaction_features: Features,
) -> Features:
    history_features, _ = sequence_model(interaction_features.unsqueeze(0))
    history_features = history_features.squeeze(0)

    return history_features


class FullHistoryIntegrator(HistoryIntegrator):
    def __init__(
        self,
        interaction_model: InteractionModel,
        sequence_model: SequenceModel,
    ):
        super().__init__()
        self.interaction_model = interaction_model
        self.sequence_model = sequence_model
        self.__features: Features
        self.__hidden: Any

    def reset(self, observation: TorchObservation):
        interaction_features = self.interaction_model(
            None,
            gtorch.unsqueeze(observation, 0),
        ).unsqueeze(1)
        self.__features, self.__hidden = self.sequence_model(
            interaction_features
        )
        self.__features = self.__features.squeeze(0).squeeze(0)

    def step(self, action: torch.Tensor, observation: TorchObservation):
        interaction_features = self.interaction_model(
            action.unsqueeze(0),
            gtorch.unsqueeze(observation, 0),
        ).unsqueeze(1)
        self.__features, self.__hidden = self.sequence_model(
            interaction_features, hidden=self.__hidden
        )
        self.__features = self.__features.squeeze(0).squeeze(0)

    def sample_features(self) -> tuple[Features, dict]:
        info = {}
        return self.__features, info
