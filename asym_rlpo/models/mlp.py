from collections.abc import Sequence

import torch

from asym_rlpo.models.model import FeatureModel
from asym_rlpo.modules.mlp import make_mlp


class MLP_Model(FeatureModel):
    def __init__(
        self,
        sizes: Sequence[int],
        nonlinearities: Sequence[str],
        *args,
        skip_last: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.sizes = sizes
        self.model = make_mlp(
            sizes,
            nonlinearities,
            *args,
            skip_last=skip_last,
            **kwargs,
        )

    @property
    def dim(self):
        return self.sizes[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def zeros_like(self, device: torch.device | None = None):
        return torch.zeros(self.sizes[-1], device=device)
