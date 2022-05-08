from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode, Torch_L, Torch_O

from .base import DQN_ABC


class ADQN_Short(DQN_ABC):
    model_keys = {
        'agent': [
            'action_model',
            'observation_model',
            'history_model',
            'qh_model',
            'latent_model',
            'qhz_model',
        ]
    }

    def compute_q_values(
        self,
        models: nn.ModuleDict,
        actions: torch.Tensor,
        observations: Torch_O,
        latents: Torch_L,
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
        inputs = torch.cat([history_features, latent_features], dim=-1)
        qhz_values = models.agent.qhz_model(inputs)

        return qh_values, qhz_values

    def qhz_loss(  # pylint: disable=no-self-use
        self,
        episode: Episode,
        qh_values: torch.Tensor,  # pylint: disable=unused-argument
        qhz_values: torch.Tensor,
        target_qh_values: torch.Tensor,
        target_qhz_values: torch.Tensor,
        *,
        discount: float,
    ) -> torch.Tensor:

        qhz_values = qhz_values.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        qhz_values_bootstrap = target_qh_values.max(-1).values.roll(-1, 0)
        qhz_values_bootstrap[-1] = 0.0

        loss = F.mse_loss(
            qhz_values,
            episode.rewards + discount * qhz_values_bootstrap,
        )
        return loss

    def qh_loss(  # pylint: disable=no-self-use
        self,
        episode: Episode,  # pylint: disable=unused-argument
        qh_values: torch.Tensor,
        qhz_values: torch.Tensor,  # pylint: disable=unused-argument
        target_qh_values: torch.Tensor,  # pylint: disable=unused-argument
        target_qhz_values: torch.Tensor,
        *,
        discount: float,  # pylint: disable=unused-argument
    ) -> torch.Tensor:

        loss = F.mse_loss(
            qh_values,
            target_qhz_values,
        )
        return loss

    def episodic_loss(
        self, episodes: Sequence[Episode], *, discount: float
    ) -> torch.Tensor:
        losses = []
        for episode in episodes:

            qh_values, qhz_values = self.compute_q_values(
                self.models,
                episode.actions,
                episode.observations,
                episode.latents,
            )
            with torch.no_grad():
                target_qh_values, target_qhz_values = self.compute_q_values(
                    self.target_models,
                    episode.actions,
                    episode.observations,
                    episode.latents,
                )

            qhz_loss = self.qhz_loss(
                episode,
                qh_values,
                qhz_values,
                target_qh_values,
                target_qhz_values,
                discount=discount,
            )
            qh_loss = self.qh_loss(
                episode,
                qh_values,
                qhz_values,
                target_qh_values,
                target_qhz_values,
                discount=discount,
            )
            loss = (qhz_loss + qh_loss) / 2
            losses.append(loss)

        return sum(losses) / len(losses)  # type: ignore
