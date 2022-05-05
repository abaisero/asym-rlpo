from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode, Torch_O, Torch_S

from .base import EpisodicDQN_ABC


class ADQN_State(EpisodicDQN_ABC):
    model_keys = {
        'agent': [
            'action_model',
            'observation_model',
            'history_model',
            'qh_model',
            'state_model',
            'qs_model',
        ]
    }

    def compute_q_values(
        self,
        models: nn.ModuleDict,
        actions: torch.Tensor,
        observations: Torch_O,
        states: Torch_S,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        history_features = self.compute_history_features(
            models.agent.action_model,
            models.agent.observation_model,
            models.agent.history_model,
            actions,
            observations,
        )
        qh_values = models.agent.qh_model(history_features)

        state_features = models.agent.state_model(states)
        qs_values = models.agent.qs_model(state_features)

        return qh_values, qs_values

    def qs_loss(
        self,
        episode: Episode,
        qh_values: torch.Tensor,
        qs_values: torch.Tensor,
        target_qh_values: torch.Tensor,
        target_qs_values: torch.Tensor,
        *,
        discount: float,
    ) -> torch.Tensor:

        qs_values = qs_values.gather(1, episode.actions.unsqueeze(-1)).squeeze(
            -1
        )
        qs_values_bootstrap = (
            target_qs_values.gather(
                1, target_qh_values.argmax(-1).unsqueeze(-1)
            )
            .squeeze(-1)
            .roll(-1, 0)
        )
        qs_values_bootstrap[-1] = 0.0

        loss = F.mse_loss(
            qs_values,
            episode.rewards + discount * qs_values_bootstrap,
        )
        return loss

    def qh_loss(
        self,
        episode: Episode,
        qh_values: torch.Tensor,
        qs_values: torch.Tensor,
        target_qh_values: torch.Tensor,
        target_qs_values: torch.Tensor,
        *,
        discount: float,
    ) -> torch.Tensor:

        loss = F.mse_loss(
            qh_values,
            target_qs_values,
        )
        return loss

    def episodic_loss(
        self, episodes: Sequence[Episode], *, discount: float
    ) -> torch.Tensor:
        losses = []
        for episode in episodes:

            qh_values, qs_values = self.compute_q_values(
                self.models,
                episode.actions,
                episode.observations,
                episode.states,
            )
            with torch.no_grad():
                target_qh_values, target_qs_values = self.compute_q_values(
                    self.target_models,
                    episode.actions,
                    episode.observations,
                    episode.states,
                )

            qs_loss = self.qs_loss(
                episode,
                qh_values,
                qs_values,
                target_qh_values,
                target_qs_values,
                discount=discount,
            )
            qh_loss = self.qh_loss(
                episode,
                qh_values,
                qs_values,
                target_qh_values,
                target_qs_values,
                discount=discount,
            )
            loss = (qs_loss + qh_loss) / 2
            losses.append(loss)

        return sum(losses) / len(losses)  # type: ignore


class ADQN_State_Bootstrap(ADQN_State):
    def qh_loss(
        self,
        episode: Episode,
        qh_values: torch.Tensor,
        qs_values: torch.Tensor,
        target_qh_values: torch.Tensor,
        target_qs_values: torch.Tensor,
        *,
        discount: float,
    ) -> torch.Tensor:

        qh_values = qh_values.gather(1, episode.actions.unsqueeze(-1)).squeeze(
            -1
        )
        qs_values_bootstrap = (
            target_qs_values.gather(
                1, target_qh_values.argmax(-1).unsqueeze(-1)
            )
            .squeeze(-1)
            .roll(-1, 0)
        )
        qs_values_bootstrap[-1] = 0.0

        loss = F.mse_loss(
            qh_values,
            episode.rewards + discount * qs_values_bootstrap,
        )
        return loss
