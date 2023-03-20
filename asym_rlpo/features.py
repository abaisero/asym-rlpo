import abc
from collections import deque
from typing import Callable, Deque, Optional

import torch

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import TorchObservation
from asym_rlpo.representations.history import HistoryRepresentation
from asym_rlpo.representations.interaction import InteractionRepresentation


def compute_interaction_features(
    interaction_model: InteractionRepresentation,
    action: Optional[torch.Tensor],
    observation: TorchObservation,
    *,
    device: torch.device,
) -> torch.Tensor:
    action_model = interaction_model.action_model
    observation_model = interaction_model.observation_model

    observation_features = observation_model(gtorch.to(observation, device))
    batch_shape = observation_features.shape[:-1]
    action_features = (
        torch.zeros(batch_shape + (action_model.dim,), device=device)
        if action is None
        else action_model(action.to(device))
    )
    interaction_features = torch.cat(
        [action_features, observation_features], dim=-1
    )

    return interaction_features


def compute_full_history_features(
    interaction_model: InteractionRepresentation,
    history_model: HistoryRepresentation,
    actions: torch.Tensor,
    observations: TorchObservation,
) -> torch.Tensor:
    action_features = interaction_model.action_model(actions)
    action_features = action_features.roll(1, 0)
    action_features[0, :] = 0.0
    observation_features = interaction_model.observation_model(observations)

    interaction_features = torch.cat(
        [action_features, observation_features],
        dim=-1,
    )
    history_features, _ = history_model(interaction_features.unsqueeze(0))
    history_features = history_features.squeeze(0)

    return history_features


def compute_truncated_history_features(
    interaction_model: InteractionRepresentation,
    history_model: HistoryRepresentation,
    actions: torch.Tensor,
    observations: TorchObservation,
    *,
    n: int,
) -> torch.Tensor:
    if n <= 0:
        raise ValueError(f'invalid truncation value n={n}')

    action_features = interaction_model.action_model(actions)
    action_features = action_features.roll(1, 0)
    action_features[0, :] = 0.0
    observation_features = interaction_model.observation_model(observations)

    interaction_features = torch.cat(
        [action_features, observation_features],
        dim=-1,
    )
    padding = torch.zeros_like(interaction_features[0].expand(n - 1, -1))
    interaction_features = torch.cat(
        [padding, interaction_features],
        dim=0,
    ).unfold(0, n, 1)
    interaction_features = interaction_features.swapaxes(-2, -1)
    history_features, _ = history_model(interaction_features)
    history_features = history_features[:, -1]

    return history_features


def compute_history_features(
    interaction_model: InteractionRepresentation,
    history_model: HistoryRepresentation,
    actions: torch.Tensor,
    observations: TorchObservation,
    *,
    n: Optional[int] = None,
) -> torch.Tensor:
    return (
        compute_full_history_features(
            interaction_model,
            history_model,
            actions,
            observations,
        )
        if n is None
        else compute_truncated_history_features(
            interaction_model,
            history_model,
            actions,
            observations,
            n=n,
        )
    )


class HistoryIntegrator(metaclass=abc.ABCMeta):
    def __init__(
        self,
        interaction_model: InteractionRepresentation,
        history_model: HistoryRepresentation,
    ):
        self.interaction_model = interaction_model
        self.history_model = history_model

    def compute_interaction_features(
        self, action: Optional[torch.Tensor], observation: TorchObservation
    ) -> torch.Tensor:
        # the history model is the only one guaranteed to have parameters
        device = next(self.history_model.parameters()).device
        return compute_interaction_features(
            self.interaction_model,
            action,
            observation,
            device=device,
        )

    @abc.abstractmethod
    def reset(self, observation):
        assert False

    @abc.abstractmethod
    def step(self, action, observation):
        assert False

    @property
    @abc.abstractmethod
    def features(self) -> torch.Tensor:
        assert False


class FullHistoryIntegrator(HistoryIntegrator):
    def __init__(
        self,
        interaction_model: InteractionRepresentation,
        history_model: HistoryRepresentation,
    ):
        super().__init__(
            interaction_model,
            history_model,
        )
        self.__features: torch.Tensor
        self.__hidden: torch.Tensor

    def reset(self, observation):
        interaction_features = self.compute_interaction_features(
            None,
            gtorch.unsqueeze(observation, 0),
        ).unsqueeze(1)
        self.__features, self.__hidden = self.history_model(
            interaction_features
        )
        self.__features = self.__features.squeeze(0).squeeze(0)

    def step(self, action, observation):
        interaction_features = self.compute_interaction_features(
            action.unsqueeze(0),
            gtorch.unsqueeze(observation, 0),
        ).unsqueeze(1)
        self.__features, self.__hidden = self.history_model(
            interaction_features, hidden=self.__hidden
        )
        self.__features = self.__features.squeeze(0).squeeze(0)

    @property
    def features(self) -> torch.Tensor:
        return self.__features


class TruncatedHistoryIntegrator(HistoryIntegrator):
    def __init__(
        self,
        interaction_model: InteractionRepresentation,
        history_model: HistoryRepresentation,
        *,
        n: int,
    ):
        super().__init__(
            interaction_model,
            history_model,
        )
        self.n = n
        self._interaction_features_deque: Deque[torch.Tensor] = deque(maxlen=n)

    def reset(self, observation):
        interaction_features = self.compute_interaction_features(
            None,
            gtorch.unsqueeze(observation, 0),
        ).squeeze(0)

        self._interaction_features_deque.clear()
        self._interaction_features_deque.extend(
            torch.zeros(interaction_features.size(-1))
            for _ in range(self.n - 1)
        )

        self._interaction_features_deque.append(interaction_features)

    def step(self, action, observation):
        interaction_features = self.compute_interaction_features(
            action.unsqueeze(0),
            gtorch.unsqueeze(observation, 0),
        ).squeeze(0)
        self._interaction_features_deque.append(interaction_features)

    @property
    def features(self) -> torch.Tensor:
        assert len(self._interaction_features_deque) == self.n

        interaction_tensors = tuple(self._interaction_features_deque)
        interaction_features = torch.stack(interaction_tensors).unsqueeze(0)
        history_features, _ = self.history_model(interaction_features)
        history_features = history_features.squeeze(0)[-1]

        return history_features


def make_history_integrator(
    interaction_model: InteractionRepresentation,
    history_model: HistoryRepresentation,
    *,
    truncated_histories_n: Optional[int] = None,
) -> HistoryIntegrator:
    return (
        FullHistoryIntegrator(
            interaction_model,
            history_model,
        )
        if truncated_histories_n is None
        else TruncatedHistoryIntegrator(
            interaction_model,
            history_model,
            n=truncated_histories_n,
        )
    )


HistoryIntegratorMaker = Callable[
    [InteractionRepresentation, HistoryRepresentation],
    HistoryIntegrator,
]
HistoryFeaturesComputer = Callable[
    [
        InteractionRepresentation,
        HistoryRepresentation,
        torch.Tensor,
        TorchObservation,
    ],
    torch.Tensor,
]
