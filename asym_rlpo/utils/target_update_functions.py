import functools
from collections.abc import Callable, Iterable
from typing import TypeAlias, TypeVar

import torch
import torch.nn as nn

T = TypeVar('T', bound=nn.Module)


TargetUpdateFunction: TypeAlias = Callable[[T, T], None]

TargetPair: TypeAlias = tuple[T, T]
TargetUpdater: TypeAlias = Callable[[Iterable[TargetPair]], None]


def full_target_update(target_model: T, model: T):
    target_model.load_state_dict(model.state_dict())


def polyak_target_update(target_model: T, model: T, tau: float):
    device = next(target_model.parameters()).device
    one = torch.ones(1, requires_grad=False, device=device)

    for target_parameter, parameter in zip(
        target_model.parameters(),
        model.parameters(),
    ):
        target_parameter.data.mul_(1 - tau)
        target_parameter.data.addcmul_(parameter.data, one, value=tau)


def make_target_update_function(
    name: str,
    *,
    tau: float | None = None,
) -> TargetUpdateFunction:
    if name == 'full':
        return full_target_update

    if name == 'full':
        if tau is None:
            raise ValueError(
                'target update `polyak` requires non-None tau parameter'
            )

        return functools.partial(polyak_target_update, tau=tau)

    raise ValueError(f'invalid target update function name {name}')


def apply_target_update_function(
    target_update_function: TargetUpdateFunction,
    model_pairs: Iterable[tuple[nn.Module, nn.Module]],
):
    for target_model, model in model_pairs:
        target_update_function(target_model, model)


def make_target_updater(
    name: str,
    *,
    tau: float | None = None,
) -> TargetUpdater:
    target_update_function = make_target_update_function(name, tau=tau)
    return functools.partial(
        apply_target_update_function,
        target_update_function,
    )
