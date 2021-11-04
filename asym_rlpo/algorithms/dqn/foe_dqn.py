from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from asym_rlpo.data import Episode

from .base import FO_EpisodicDQN_ABC


class FOE_DQN(FO_EpisodicDQN_ABC):
    model_keys = {
        'agent': [
            'state_model',
            'qs_model',
        ]
    }

    def episodic_loss(
        self, episodes: Sequence[Episode], *, discount: float
    ) -> torch.Tensor:

        losses = []
        for episode in episodes:

            qs_values = self.models.agent.qs_model(
                self.models.agent.state_model(episode.states)
            )
            with torch.no_grad():
                target_qs_values = self.target_models.agent.qs_model(
                    self.models.agent.state_model(episode.states)
                )

            qs_values = qs_values.gather(
                1, episode.actions.unsqueeze(-1)
            ).squeeze(-1)
            qs_values_bootstrap = target_qs_values.max(-1).values.roll(-1, 0)
            qs_values_bootstrap[-1] = 0.0

            loss = F.mse_loss(
                qs_values,
                episode.rewards + discount * qs_values_bootstrap,
            )
            losses.append(loss)

        return sum(losses) / len(losses)  # type: ignore
