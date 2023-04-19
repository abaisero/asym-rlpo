import torch

from asym_rlpo.models.model import FeatureModel


class EmptyModel(FeatureModel):
    @property
    def dim(self):
        return 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(inputs.shape + (0,))

    def zeros_like(self, device: torch.device | None = None):
        return torch.tensor([], device=device)
