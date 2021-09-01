import functools
from typing import Callable, Optional

from asym_rlpo.utils.debugging import checkraise

Schedule = Callable[[int], float]


def constant_schedule(
    step: int, *, const: float  # pylint: disable=unused-argument
) -> float:
    return const


def linear_schedule(
    step: int, *, value_from: float, value_to: float, nsteps: int
) -> float:
    # normalize step between 0.0 and 1.0
    t = min(max(0.0, step / (nsteps - 1)), 1.0)
    return value_from * (1.0 - t) + value_to * t


def exponential_schedule(
    step: int, *, value_from: float, halflife: int
) -> float:
    return value_from * 0.5 ** (step / halflife)


def make_schedule(
    name: str,
    *,
    const: Optional[int] = None,
    value_from: Optional[float] = None,
    value_to: Optional[float] = None,
    nsteps: Optional[int] = None,
    halflife: Optional[int] = None,
) -> Schedule:

    if name == 'constant':
        checkraise(
            const is not None,
            ValueError,
            f'invalid arguments {const}',
        )
        return functools.partial(constant_schedule, const=const)

    if name == 'linear':
        checkraise(
            None not in [value_from, value_to, nsteps],
            ValueError,
            f'invalid arguments {value_from} {value_to} {nsteps}',
        )
        return functools.partial(
            linear_schedule,
            value_from=value_from,
            value_to=value_to,
            nsteps=nsteps,
        )

    if name == 'exponential':
        checkraise(
            None not in [value_from, halflife],
            ValueError,
            f'invalid arguments {value_from} {halflife}',
        )
        return functools.partial(
            exponential_schedule,
            value_from=value_from,
            halflife=halflife,
        )

    raise ValueError(f'invalid schedule name {name}')
