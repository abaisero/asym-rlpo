import functools
from collections.abc import Callable

Schedule = Callable[[int], float]


def constant_schedule(step: int, *, const: float) -> float:
    return const


def linear_schedule(
    step: int,
    *,
    value_from: float,
    value_to: float,
    nsteps: int,
) -> float:
    # normalize step between 0.0 and 1.0
    t = min(max(0.0, step / (nsteps - 1)), 1.0)
    return value_from * (1.0 - t) + value_to * t


def exponential_schedule(
    step: int,
    *,
    value_from: float,
    halflife: int,
) -> float:
    return value_from * 0.5 ** (step / halflife)


def make_schedule(
    name: str,
    *,
    const: int | None = None,
    value_from: float | None = None,
    value_to: float | None = None,
    nsteps: int | None = None,
    halflife: int | None = None,
) -> Schedule:
    if name == 'constant':
        if const is None:
            raise ValueError(f'invalid arguments {const}')

        return functools.partial(constant_schedule, const=const)

    if name == 'linear':
        if value_from is None or value_to is None or nsteps is None:
            raise ValueError(
                f'invalid arguments {value_from} {value_to} {nsteps}'
            )

        return functools.partial(
            linear_schedule,
            value_from=value_from,
            value_to=value_to,
            nsteps=nsteps,
        )

    if name == 'exponential':
        if value_from is None or halflife is None:
            raise ValueError(f'invalid arguments {value_from} {halflife}')

        return functools.partial(
            exponential_schedule,
            value_from=value_from,
            halflife=halflife,
        )

    raise ValueError(f'invalid schedule name {name}')
