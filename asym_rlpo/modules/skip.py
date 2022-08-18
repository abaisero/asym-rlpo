import torch
import torch.nn as nn


class SkipSequential(nn.Sequential):
    """Skip-connection version of nn.Sequential.

    Assumes last module is a nonlinearity.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # self iterates over the nn.Sequential modules
        *first_modules, last_module = self

        output = input
        for module in first_modules:
            output = module(input)

        return last_module(input + output)
