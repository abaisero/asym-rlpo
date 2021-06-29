from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode, Torch_O, Torch_S

from .base import PO_EpisodicDQN_ABC


class ADQN(PO_EpisodicDQN_ABC):
    model_keys = [
        'action_model',
        'observation_model',
        'history_model',
        'qh_model',
        'state_model',
        'qhs_model',
    ]

    def compute_q_values(
        self,
        models: nn.ModuleDict,
        actions: torch.Tensor,
        observations: Torch_O,
        states: Torch_S,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        history_features = self.compute_history_features(
            models.action_model,
            models.observation_model,
            models.history_model,
            actions,
            observations,
        )
        qh_values = models.qh_model(history_features)

        state_features = models.state_model(states)
        inputs = torch.cat([history_features, state_features], dim=-1)
        qhs_values = models.qhs_model(inputs)

        return qh_values, qhs_values

    def qhs_loss(  # pylint: disable=no-self-use
        self,
        episode: Episode,
        qh_values: torch.Tensor,  # pylint: disable=unused-argument
        qhs_values: torch.Tensor,
        target_qh_values: torch.Tensor,
        target_qhs_values: torch.Tensor,
        *,
        discount: float,
    ) -> torch.Tensor:

        qhs_values = qhs_values.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        qhs_values_bootstrap = (
            target_qhs_values.gather(
                1, target_qh_values.argmax(-1).unsqueeze(-1)
            )
            .squeeze(-1)
            .roll(-1, 0)
        )
        qhs_values_bootstrap[-1] = 0.0

        loss = F.mse_loss(
            qhs_values,
            episode.rewards + discount * qhs_values_bootstrap,
        )
        return loss

    def qh_loss(  # pylint: disable=no-self-use
        self,
        episode: Episode,  # pylint: disable=unused-argument
        qh_values: torch.Tensor,
        qhs_values: torch.Tensor,  # pylint: disable=unused-argument
        target_qh_values: torch.Tensor,  # pylint: disable=unused-argument
        target_qhs_values: torch.Tensor,
        *,
        discount: float,  # pylint: disable=unused-argument
    ) -> torch.Tensor:

        loss = F.mse_loss(
            qh_values,
            target_qhs_values,
        )
        return loss

    def episodic_loss(
        self, episodes: Sequence[Episode], *, discount: float
    ) -> torch.Tensor:
        losses = []
        for episode in episodes:

            qh_values, qhs_values = self.compute_q_values(
                self.models,
                episode.actions,
                episode.observations,
                episode.states,
            )
            with torch.no_grad():
                target_qh_values, target_qhs_values = self.compute_q_values(
                    self.target_models,
                    episode.actions,
                    episode.observations,
                    episode.states,
                )

            qhs_loss = self.qhs_loss(
                episode,
                qh_values,
                qhs_values,
                target_qh_values,
                target_qhs_values,
                discount=discount,
            )
            qh_loss = self.qh_loss(
                episode,
                qh_values,
                qhs_values,
                target_qh_values,
                target_qhs_values,
                discount=discount,
            )
            loss = (qhs_loss + qh_loss) / 2
            losses.append(loss)

        return sum(losses) / len(losses)  # type: ignore


class ADQN_Bootstrap(ADQN):
    def qh_loss(
        self,
        episode: Episode,
        qh_values: torch.Tensor,
        qhs_values: torch.Tensor,
        target_qh_values: torch.Tensor,
        target_qhs_values: torch.Tensor,
        *,
        discount: float,
    ) -> torch.Tensor:

        qh_values = qh_values.gather(1, episode.actions.unsqueeze(-1)).squeeze(
            -1
        )
        qhs_values_bootstrap = (
            target_qhs_values.gather(
                1, target_qh_values.argmax(-1).unsqueeze(-1)
            )
            .squeeze(-1)
            .roll(-1, 0)
        )
        qhs_values_bootstrap[-1] = 0.0

        loss = F.mse_loss(
            qh_values,
            episode.rewards + discount * qhs_values_bootstrap,
        )
        return loss
