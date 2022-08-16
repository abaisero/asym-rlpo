from typing import List, Sequence

import more_itertools as mitt
import torch.nn as nn

from .init import init_linear_module

# NOTE: the possible values for `nonlinearity` change between some of these
# functions

def make_perceptron_modules(
    in_features: int,
    out_features: int,
    nonlinearity: str,
    *args,
    **kwargs,
) -> List[nn.Module]:
    """creates and initializes a linear module with a nonlinearity"""

    module = nn.Linear(in_features, out_features, *args, **kwargs)

    if nonlinearity == 'identity':
        init_linear_module(module, 'linear')
        return [module]

    if nonlinearity == 'relu':
        init_linear_module(module, 'relu')
        return [module, nn.ReLU()]

    if nonlinearity == 'logsoftmax':
        init_linear_module(module, 'linear')
        return [module, nn.LogSoftmax(dim=-1)]

    raise ValueError(f'invalid nonlinearity {nonlinearity}')


def make_mlp_modules(
    sizes: Sequence[int],
    nonlinearities: Sequence[str],
    *args,
    **kwargs,
) -> List[nn.Module]:
    """creates and initializes multiple layers of linear modules and nonlinearities"""

    if len(sizes) < 2:
        raise ValueError(f'requires at least 2 sizes, given {len(sizes)}')

    if len(nonlinearities) != len(sizes) - 1:
        raise ValueError(
            f'incompatible number of sizes and nonlinearities, '
            f'given {len(sizes)} and {len(nonlinearities)}'
        )

    modules = mitt.flatten(
        make_perceptron_modules(
            in_size,
            out_size,
            nonlinearity,
            *args,
            **kwargs,
        )
        for (in_size, out_size), nonlinearity in zip(
            mitt.pairwise(sizes), nonlinearities
        )
    )
    return list(modules)


def make_mlp(
    sizes: Sequence[int],
    nonlinearities: Sequence[str],
    *args,
    **kwargs,
) -> nn.Sequential:
    """creates and initializes multiple layers of linear modules and nonlinearities"""

    modules = make_mlp_modules(sizes, nonlinearities, *args, **kwargs)
    return nn.Sequential(*modules)
