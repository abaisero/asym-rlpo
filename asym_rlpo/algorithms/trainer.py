from __future__ import annotations

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from asym_rlpo.types import (
    GradientNormDict,
    LossDict,
    OptimizerDict,
    OptimizerFactory,
    ParametersDict,
)


class Trainer:
    def __init__(
        self,
        optimizers: OptimizerDict,
        parameters: ParametersDict,
        *,
        max_gradient_norm: float,
    ):
        super().__init__()
        self.optimizers = optimizers
        self.parameters = parameters
        self.max_gradient_norm = max_gradient_norm

    @staticmethod
    def from_factories(
        optimizer_factories: dict[str, OptimizerFactory],
        parameters: ParametersDict,
        max_gradient_norm: float,
    ) -> Trainer:
        optimizers = {
            k: optimizer_factory(parameters[k])
            for k, optimizer_factory in optimizer_factories.items()
        }
        return Trainer(
            optimizers,
            parameters,
            max_gradient_norm=max_gradient_norm,
        )

    def __getitem__(self, key: str) -> torch.optim.Optimizer:
        return self.optimizers[key]

    def __getattr__(self, name: str) -> torch.optim.Optimizer:
        return self.optimizers[name]

    def state_dict(self) -> dict:
        return {
            k: optimizer.state_dict()
            for k, optimizer in self.optimizers.items()
        }

    def load_state_dict(self, state_dict: dict):
        if state_dict.keys() != self.optimizers.keys():
            raise ValueError(f'incompatible keys `{state_dict.keys()}`')

        for k, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict[k])

    def gradient_step(self, losses: LossDict) -> GradientNormDict:
        if losses.keys() != self.optimizers.keys():
            raise ValueError(f'incompatible keys `{losses.keys()}`')

        gradient_norms = {}

        for k, optimizer in self.optimizers.items():
            optimizer.zero_grad()
            losses[k].backward()
            gradient_norms[k] = clip_grad_norm_(
                self.parameters[k],
                max_norm=self.max_gradient_norm,
            )
            optimizer.step()

        return gradient_norms
