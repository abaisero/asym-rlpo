from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode, Torch_O

from .base import PO_EpisodicDQN_ABC


class DQN(PO_EpisodicDQN_ABC):
    model_keys = [
        'action_model',
        'observation_model',
        'history_model',
        'qh_model',
    ]

    def compute_q_values(
        self,
        models: nn.ModuleDict,
        actions: torch.Tensor,
        observations: Torch_O,
    ) -> torch.Tensor:

        history_features = self.compute_history_features(
            models.action_model,
            models.observation_model,
            models.history_model,
            actions,
            observations,
        )
        qh_values = models.qh_model(history_features)
        return qh_values

    def episodic_loss(
        self, episodes: Sequence[Episode], *, discount: float
    ) -> torch.Tensor:

        losses = []
        for episode in episodes:

            q_values = self.compute_q_values(
                self.models, episode.actions, episode.observations
            )
            with torch.no_grad():
                target_q_values = self.compute_q_values(
                    self.target_models, episode.actions, episode.observations
                )

            q_values = q_values.gather(
                1, episode.actions.unsqueeze(-1)
            ).squeeze(-1)
            q_values_bootstrap = target_q_values.max(-1).values.roll(-1, 0)
            q_values_bootstrap[-1] = 0.0
            loss = F.mse_loss(
                q_values,
                episode.rewards + discount * q_values_bootstrap,
            )
            losses.append(loss)

        return sum(losses) / len(losses)  # type: ignore
