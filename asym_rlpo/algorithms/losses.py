from __future__ import annotations

import torch
import torch.nn.functional as F


def dqn_loss_action_values(
    values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    discount: float,
    target_values: torch.Tensor,
) -> torch.Tensor:
    values = values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_values = target_values.max(-1).values.roll(-1, 0)
    next_values[-1] = 0.0

    return F.mse_loss(values, rewards + discount * next_values)


def dqn_loss_all_values(
    values: torch.Tensor,
    target_values: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(values, target_values)


def dqn_loss_bootstrap(
    values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    discount: float,
    target_values: torch.Tensor,
    target_action_selector: torch.Tensor,
) -> torch.Tensor:
    values = values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_values = (
        target_values.gather(1, target_action_selector.argmax(-1).unsqueeze(-1))
        .squeeze(-1)
        .roll(-1, 0)
    )
    next_values[-1] = 0.0

    return F.mse_loss(values, rewards + discount * next_values)
