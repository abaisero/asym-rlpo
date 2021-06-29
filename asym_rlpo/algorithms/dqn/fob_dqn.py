from __future__ import annotations

import torch
import torch.nn.functional as F

from asym_rlpo.data import Batch

from .base import FO_BatchedDQN_ABC


class FOB_DQN(FO_BatchedDQN_ABC):
    model_keys = ['state_model', 'qs_model']

    def batched_loss(self, batch: Batch, *, discount: float) -> torch.Tensor:

        qs_values = self.models.qs_model(self.models.state_model(batch.states))
        with torch.no_grad():
            target_qs_values = self.target_models.qs_model(
                self.models.state_model(batch.next_states)
            )

        qs_values = qs_values.gather(1, batch.actions.unsqueeze(-1)).squeeze(-1)
        qs_values_bootstrap = target_qs_values.max(-1).values
        qs_values_bootstrap[batch.dones] = 0.0

        loss = F.mse_loss(
            qs_values,
            batch.rewards + discount * qs_values_bootstrap,
        )
        return loss
