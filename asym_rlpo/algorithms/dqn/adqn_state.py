from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode, TorchLatent, TorchObservation

from .base import DQN_ABC


class ADQN_State(DQN_ABC):
    model_keys = {
        'agent': [
            'action_model',
            'observation_model',
            'history_model',
            'qh_model',
            'latent_model',
            'qz_model',
        ]
    }

    def compute_q_values(
        self,
        models: nn.ModuleDict,
        actions: torch.Tensor,
        observations: TorchObservation,
        latents: TorchLatent,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        history_features = self.compute_history_features(
            models.agent.action_model,
            models.agent.observation_model,
            models.agent.history_model,
            actions,
            observations,
        )
        qh_values = models.agent.qh_model(history_features)

        latent_features = models.agent.latent_model(latents)
        qz_values = models.agent.qz_model(latent_features)

        return qh_values, qz_values

    def qz_loss(
        self,
        episode: Episode,
        qh_values: torch.Tensor,
        qz_values: torch.Tensor,
        target_qh_values: torch.Tensor,
        target_qz_values: torch.Tensor,
        *,
        discount: float,
    ) -> torch.Tensor:

        qz_values = qz_values.gather(1, episode.actions.unsqueeze(-1)).squeeze(
            -1
        )
        qz_values_bootstrap = (
            target_qz_values.gather(
                1, target_qh_values.argmax(-1).unsqueeze(-1)
            )
            .squeeze(-1)
            .roll(-1, 0)
        )
        qz_values_bootstrap[-1] = 0.0

        loss = F.mse_loss(
            qz_values,
            episode.rewards + discount * qz_values_bootstrap,
        )
        return loss

    def qh_loss(
        self,
        episode: Episode,
        qh_values: torch.Tensor,
        qz_values: torch.Tensor,
        target_qh_values: torch.Tensor,
        target_qz_values: torch.Tensor,
        *,
        discount: float,
    ) -> torch.Tensor:

        loss = F.mse_loss(
            qh_values,
            target_qz_values,
        )
        return loss

    def episodic_loss(
        self, episodes: Sequence[Episode], *, discount: float
    ) -> torch.Tensor:
        losses = []
        for episode in episodes:

            qh_values, qz_values = self.compute_q_values(
                self.models,
                episode.actions,
                episode.observations,
                episode.latents,
            )
            with torch.no_grad():
                target_qh_values, target_qz_values = self.compute_q_values(
                    self.target_models,
                    episode.actions,
                    episode.observations,
                    episode.latents,
                )

            qz_loss = self.qz_loss(
                episode,
                qh_values,
                qz_values,
                target_qh_values,
                target_qz_values,
                discount=discount,
            )
            qh_loss = self.qh_loss(
                episode,
                qh_values,
                qz_values,
                target_qh_values,
                target_qz_values,
                discount=discount,
            )
            loss = (qz_loss + qh_loss) / 2
            losses.append(loss)

        return sum(losses) / len(losses)  # type: ignore


class ADQN_State_Bootstrap(ADQN_State):
    def qh_loss(
        self,
        episode: Episode,
        qh_values: torch.Tensor,
        qz_values: torch.Tensor,
        target_qh_values: torch.Tensor,
        target_qz_values: torch.Tensor,
        *,
        discount: float,
    ) -> torch.Tensor:

        qh_values = qh_values.gather(1, episode.actions.unsqueeze(-1)).squeeze(
            -1
        )
        qz_values_bootstrap = (
            target_qz_values.gather(
                1, target_qh_values.argmax(-1).unsqueeze(-1)
            )
            .squeeze(-1)
            .roll(-1, 0)
        )
        qz_values_bootstrap[-1] = 0.0

        loss = F.mse_loss(
            qh_values,
            episode.rewards + discount * qz_values_bootstrap,
        )
        return loss
