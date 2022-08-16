import torch.nn as nn

# NOTE: the possible values for `nonlinearity` change between some of these
# functions


def init_linear_module(module: nn.Linear, nonlinearity: str):
    """initializes a linear module according to the nonlinearity"""
    gain = nn.init.calculate_gain(nonlinearity)
    nn.init.xavier_normal_(module.weight, gain)

    if module.bias is not None:
        nn.init.zeros_(module.bias)

    return module
