from __future__ import annotations

from torch.nn.utils.clip_grad import clip_grad_norm_

from asym_rlpo.types import (
    GradientNormDict,
    LossDict,
    OptimizerDict,
    OptimizerFactory,
    ParametersGeneratorsDict,
)


class Trainer:
    def __init__(
        self,
        optimizers: OptimizerDict,
        parameters_generators: ParametersGeneratorsDict,
        *,
        max_gradient_norm: float,
    ):
        super().__init__()
        self.optimizers = optimizers
        self.parameters_generators = parameters_generators
        self.max_gradient_norm = max_gradient_norm

    @staticmethod
    def from_factories(
        optimizer_factories: dict[str, OptimizerFactory],
        parameters_generators: ParametersGeneratorsDict,
        max_gradient_norm: float,
    ) -> Trainer:
        optimizers = {
            k: optimizer_factory(parameters_generators[k]())
            for k, optimizer_factory in optimizer_factories.items()
        }
        return Trainer(
            optimizers,
            parameters_generators,
            max_gradient_norm=max_gradient_norm,
        )

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
            parameters_generator = self.parameters_generators[k]
            parameters = parameters_generator()

            optimizer.zero_grad()
            losses[k].backward()
            gradient_norms[k] = clip_grad_norm_(
                parameters,
                max_norm=self.max_gradient_norm,
            )
            optimizer.step()

        return gradient_norms
