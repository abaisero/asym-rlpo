import torch.nn as nn


def make_module(name: str, nonlinearity: str, *args, **kwargs) -> nn.Module:
    """make_module.

    Utility function to create and initialize pytorch Modules.  Primarily
    necessary for nn.Linear.

    Args:
        name (str):
        nonlinearity (str):
        args:
        kwargs:

    Returns:
        nn.Module:
    """
    if name == 'linear':
        module = nn.Linear(*args, **kwargs)

        gain = nn.init.calculate_gain(nonlinearity)
        nn.init.xavier_normal_(module.weight, gain)
        try:
            # the linear model might not have a bias
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

        return module

    raise ValueError(f'invalid module name {name}')
