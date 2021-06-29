import abc
from collections import deque
from typing import Deque, Optional

import torch
import torch.nn as nn

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import Torch_O


def compute_input_features(
    action_model: nn.Module,
    observation_model: nn.Module,
    action: Optional[torch.Tensor],
    observation: Torch_O,
    *,
    device: torch.device,
) -> torch.Tensor:

    observation_features = observation_model(gtorch.to(observation, device))
    batch_shape = observation_features.shape[:-1]
    action_features = (
        torch.zeros(batch_shape + (action_model.dim,), device=device)
        if action is None
        else action_model(action.to(device))
    )
    input_features = torch.cat([action_features, observation_features], dim=-1)

    return input_features


def compute_full_history_features(
    action_model: nn.Module,
    observation_model: nn.Module,
    history_model: nn.Module,
    actions: torch.Tensor,
    observations: Torch_O,
) -> torch.Tensor:

    action_features = action_model(actions)
    action_features = action_features.roll(1, 0)
    action_features[0, :] = 0.0
    observation_features = observation_model(observations)

    inputs = torch.cat([action_features, observation_features], dim=-1)
    history_features, _ = history_model(inputs.unsqueeze(0))
    history_features = history_features.squeeze(0)

    return history_features


def compute_truncated_history_features(
    action_model: nn.Module,
    observation_model: nn.Module,
    history_model: nn.Module,
    actions: torch.Tensor,
    observations: Torch_O,
    *,
    n: int,
) -> torch.Tensor:

    action_features = action_model(actions)
    action_features = action_features.roll(1, 0)
    action_features[0, :] = 0.0
    observation_features = observation_model(observations)

    inputs = torch.cat([action_features, observation_features], dim=-1)
    padding = torch.zeros_like(inputs[0].expand(n - 1, -1))
    inputs = torch.cat([padding, inputs], dim=0).unfold(0, n, 1)
    inputs = inputs.swapaxes(-2, -1)
    history_features, _ = history_model(inputs)
    history_features = history_features[:, -1]

    return history_features


def compute_history_features(
    action_model: nn.Module,
    observation_model: nn.Module,
    history_model: nn.Module,
    actions: torch.Tensor,
    observations: Torch_O,
    *,
    truncated: bool,
    n: int,
) -> torch.Tensor:

    return (
        compute_truncated_history_features(
            action_model,
            observation_model,
            history_model,
            actions,
            observations,
            n=n,
        )
        if truncated
        else compute_full_history_features(
            action_model,
            observation_model,
            history_model,
            actions,
            observations,
        )
    )


class HistoryIntegrator(metaclass=abc.ABCMeta):
    def __init__(
        self,
        action_model: nn.Module,
        observation_model: nn.Module,
        history_model: nn.Module,
    ):
        self.action_model = action_model
        self.observation_model = observation_model
        self.history_model = history_model

    def compute_input_features(
        self, action: Optional[torch.Tensor], observation: Torch_O
    ) -> torch.Tensor:

        # the history model is the only one guaranteed to have parameters
        device = next(self.history_model.parameters()).device
        return compute_input_features(
            self.action_model,
            self.observation_model,
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
        action_model: nn.Module,
        observation_model: nn.Module,
        history_model: nn.Module,
    ):
        super().__init__(
            action_model,
            observation_model,
            history_model,
        )
        self._features: torch.Tensor = None
        self._hidden = None

    def reset(self, observation):
        self._hidden = None

        input_features = self.compute_input_features(
            None,
            gtorch.unsqueeze(observation, 0),
        ).unsqueeze(1)
        self._features, self._hidden = self.history_model(
            input_features, hidden=self._hidden
        )
        self._features = self._features.squeeze(0).squeeze(0)

    def step(self, action, observation):
        input_features = self.compute_input_features(
            action.unsqueeze(0),
            gtorch.unsqueeze(observation, 0),
        ).unsqueeze(1)
        self._features, self._hidden = self.history_model(
            input_features, hidden=self._hidden
        )
        self._features = self._features.squeeze(0).squeeze(0)

    @property
    def features(self) -> torch.Tensor:
        return self._features


class TruncatedHistoryIntegrator(HistoryIntegrator):
    def __init__(
        self,
        action_model: nn.Module,
        observation_model: nn.Module,
        history_model: nn.Module,
        *,
        n: int,
    ):
        super().__init__(
            action_model,
            observation_model,
            history_model,
        )
        self.n = n
        self._input_features_deque: Deque[torch.Tensor] = deque(maxlen=n)

    def reset(self, observation):
        input_features = self.compute_input_features(
            None,
            gtorch.unsqueeze(observation, 0),
        ).squeeze(0)

        self._input_features_deque.clear()
        self._input_features_deque.extend(
            torch.zeros(input_features.size(-1)) for _ in range(self.n - 1)
        )

        self._input_features_deque.append(input_features)

    def step(self, action, observation):
        input_features = self.compute_input_features(
            action.unsqueeze(0),
            gtorch.unsqueeze(observation, 0),
        ).squeeze(0)
        self._input_features_deque.append(input_features)

    @property
    def features(self) -> torch.Tensor:
        assert len(self._input_features_deque) == self.n

        input_features = torch.stack(
            tuple(self._input_features_deque)
        ).unsqueeze(0)

        history_features, _ = self.history_model(input_features)
        history_features = history_features.squeeze(0)[-1]

        return history_features


def make_history_integrator(
    action_model: nn.Module,
    observation_model: nn.Module,
    history_model: nn.Module,
    *,
    truncated_histories: bool,
    truncated_histories_n: int,
) -> HistoryIntegrator:
    return (
        TruncatedHistoryIntegrator(
            action_model,
            observation_model,
            history_model,
            n=truncated_histories_n,
        )
        if truncated_histories
        else FullHistoryIntegrator(
            action_model,
            observation_model,
            history_model,
        )
    )
