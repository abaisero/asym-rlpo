import torch

from asym_rlpo.data import TorchObservation
from asym_rlpo.models.history import ReactiveHistoryIntegrator
from asym_rlpo.models.history.integrator import HistoryIntegrator
from asym_rlpo.models.memory_reactive import MemoryPolicy
from asym_rlpo.models.sequence import SequenceModel
from asym_rlpo.types import Features


def compute_memory_reactive_history_features(
    sequence_model: SequenceModel,
    interaction_features: Features,
    *,
    memory_size: int,
) -> Features:
    if memory_size <= 0:
        raise ValueError(f'invalid truncation {memory_size=}')

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


# class MemoryReactiveHistoryIntegrator(HistoryIntegrator):
#     def __init__(
#         self,
#         memory_policy: MemoryPolicy,
#         reactive_history_integrator: ReactiveHistoryIntegrator,
#     ):
#         super().__init__()
#         self.memory_policy = memory_policy
#         self.reactive_history_integrator = reactive_history_integrator

#     @property
#     def epsilon(self) -> float:
#         return self.memory_policy.epsilon

#     @epsilon.setter
#     def epsilon(self, value: float):
#         self.memory_policy.epsilon = value

#     def reset(self, observation: TorchObservation):
#         self.memory_policy.reset(observation)
#         self.reactive_history_integrator.reset(observation)

#     def step(self, action: torch.Tensor, observation: TorchObservation):
#         self.memory_policy.step(action, observation)
#         self.reactive_history_integrator.step(action, observation)

#     def sample_features(self) -> tuple[Features, dict]:
#         memory, info = self.memory_policy.sample_memory()
#         memory_features = info['memory_features']
#         info = {'memory': memory}

#         (
#             reactive_features,
#             _,
#         ) = self.reactive_history_integrator.sample_features()

#         history_features = torch.cat(
#             [memory_features, reactive_features],
#             dim=-1,
#         )
#         return history_features, info
