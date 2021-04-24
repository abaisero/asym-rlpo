from functools import partial
from typing import Optional, Protocol

import torch

from asym_rlpo.utils.debugging import checkraise


class TargetFunction(Protocol):
    def __call__(
        self, rewards: torch.Tensor, values: torch.Tensor, *, discount: float
    ) -> torch.Tensor:
        ...


def target_function_factory(
    name: str,
    *,
    n: Optional[int] = None,
    lambda_: Optional[float] = None,
) -> TargetFunction:
    if name == 'mc':
        return mc_target

    if name == 'td0':
        return partial(td0_target)

    if name == 'td-n':
        return partial(td_n_target, n=n)

    if name == 'td-lambda':
        return partial(td_lambda_target, lambda_=lambda_)

    raise ValueError('invalid target name `{name}`')


def mc_target(
    rewards: torch.Tensor,
    values: torch.Tensor,  # pylint: disable=unused-argument
    *,
    discount: float,
) -> torch.Tensor:
    checkraise(rewards.ndim == 1, ValueError, '`rewards` must have 1 dimension')

    size = rewards.size(-1)
    indices = torch.arange(size)
    exponents = indices.unsqueeze(0) - indices.unsqueeze(-1)
    discounts = (discount ** exponents).triu()
    return discounts @ rewards


def td0_target(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    discount: float,
) -> torch.Tensor:
    checkraise(rewards.ndim == 1, ValueError, '`rewards` must have 1 dimension')
    checkraise(values.ndim == 1, ValueError, '`values` must have 1 dimension')
    checkraise(
        rewards.shape == values.shape,
        ValueError,
        '`rewards` and `values` must have the same shape',
    )

    values = values.roll(-1)
    values[-1] = 0.0
    return rewards + discount * values


def td_n_target(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    discount: float,
    n: int,
) -> torch.Tensor:
    checkraise(rewards.ndim == 1, ValueError, '`rewards` must have 1 dimension')
    checkraise(values.ndim == 1, ValueError, '`values` must have 1 dimension')
    checkraise(
        rewards.shape == values.shape,
        ValueError,
        '`rewards` and `values` must have the same shape',
    )

    size = rewards.size(-1)
    indices = torch.arange(size)
    exponents = indices.unsqueeze(0) - indices.unsqueeze(-1)
    discounts = (discount ** exponents).triu().tril(n - 1)
    values = values.roll(-n)
    values[-n:] = 0.0
    return discounts @ rewards + (discount ** n) * values


def td_lambda_target(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    discount: float,
    lambda_: float,
) -> torch.Tensor:
    checkraise(rewards.ndim == 1, ValueError, '`rewards` must have 1 dimension')
    checkraise(values.ndim == 1, ValueError, '`values` must have 1 dimension')
    checkraise(
        rewards.shape == values.shape,
        ValueError,
        '`rewards` and `values` must have the same shape',
    )

    size = rewards.size(-1)
    indices = torch.arange(size)
    exponents = indices.unsqueeze(0) - indices.unsqueeze(-1)
    discounts = ((discount * lambda_) ** exponents).triu()
    values = values.roll(-1)
    values[-1] = 0.0
    return discounts @ (rewards + discount * (1 - lambda_) * values)
