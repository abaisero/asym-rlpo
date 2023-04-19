from collections import deque

import torch

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import TorchObservation
from asym_rlpo.models.history.integrator import HistoryIntegrator
from asym_rlpo.models.interaction import InteractionModel
from asym_rlpo.models.sequence import SequenceModel
from asym_rlpo.types import Features


def compute_reactive_history_features(
    sequence_model: SequenceModel,
    interaction_features: Features,
    *,
    memory_size: int,
) -> Features:
    if memory_size <= 0:
        raise ValueError(f'invalid {memory_size=}')

    padding = torch.zeros_like(
        interaction_features[0].expand(memory_size - 1, -1)
    )
    interaction_features = torch.cat(
        [padding, interaction_features],
        dim=0,
    )
    interaction_features = interaction_features.unfold(0, memory_size, 1)
    interaction_features = interaction_features.swapaxes(-2, -1)
    history_features, _ = sequence_model(interaction_features)
    history_features = history_features[:, -1]

    return history_features


class ReactiveHistoryIntegrator(HistoryIntegrator):
    def __init__(
        self,
        interaction_model: InteractionModel,
        sequence_model: SequenceModel,
        *,
        memory_size: int,
    ):
        super().__init__()
        self.interaction_model = interaction_model
        self.sequence_model = sequence_model
        self.memory_size = memory_size
        self._interaction_features_deque = deque(maxlen=memory_size)

    def reset(self, observation: TorchObservation):
        interaction_features = self.interaction_model(
            None,
            gtorch.unsqueeze(observation, 0),
        ).squeeze(0)

        self._interaction_features_deque.clear()
        self._interaction_features_deque.extend(
            torch.zeros(interaction_features.size(-1))
            for _ in range(self.memory_size - 1)
        )

        self._interaction_features_deque.append(interaction_features)

    def step(self, action: torch.Tensor, observation: TorchObservation):
        interaction_features = self.interaction_model(
            action.unsqueeze(0),
            gtorch.unsqueeze(observation, 0),
        ).squeeze(0)
        self._interaction_features_deque.append(interaction_features)

    def sample_features(self) -> tuple[Features, dict]:
        assert len(self._interaction_features_deque) == self.memory_size

        interaction_tensors = tuple(self._interaction_features_deque)
        interaction_features = torch.stack(interaction_tensors).unsqueeze(0)
        history_features, _ = self.sequence_model(interaction_features)
        history_features = history_features.squeeze(0)[-1]

        info = {}
        return history_features, info
