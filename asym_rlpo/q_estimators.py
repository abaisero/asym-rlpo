from functools import partial
from typing import Optional, Protocol

import torch

from asym_rlpo.utils.debugging import checkraise


class Q_Estimator(Protocol):
    def __call__(
        self, rewards: torch.Tensor, values: torch.Tensor, *, discount: float
    ) -> torch.Tensor:
        ...


def q_estimator_factory(
    name: str,
    *,
    n: Optional[int] = None,
    lambda_: Optional[float] = None,
) -> Q_Estimator:
    if name == 'mc':
        return mc_q_estimator

    if name == 'td0':
        return partial(td0_q_estimator)

    if name == 'td-n':
        return partial(tdn_q_estimator, n=n)

    if name == 'td-lambda':
        return partial(tdlambda_q_estimator, lambda_=lambda_)

    raise ValueError('invalid estimator name `{name}`')


def mc_q_estimator(
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


def td0_q_estimator(
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


def tdn_q_estimator(
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


def tdlambda_q_estimator(
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
