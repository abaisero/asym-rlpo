from __future__ import annotations

import abc
import functools
from typing import Protocol

from asym_rlpo.data import Episode
from asym_rlpo.models.history.full import (
    FullHistoryIntegrator,
    compute_full_history_features,
)
from asym_rlpo.models.history.integrator import HistoryIntegrator
from asym_rlpo.models.history.reactive import (
    ReactiveHistoryIntegrator,
    compute_reactive_history_features,
)
from asym_rlpo.models.interaction import InteractionModel
from asym_rlpo.models.model import Model
from asym_rlpo.models.sequence import SequenceModel
from asym_rlpo.types import Features


class HistoryFeaturesFunction(Protocol):
    def __call__(
        self,
        sequence_model: SequenceModel,
        interaction_features: Features,
    ) -> Features:
        ...


def make_history_features_function(memory_size: int) -> HistoryFeaturesFunction:
    if memory_size > 0:
        return functools.partial(
            compute_reactive_history_features,
            memory_size=memory_size,
        )

    else:
        return compute_full_history_features


class HistoryModel(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        assert False

    @abc.abstractmethod
    def episodic(self, episode: Episode) -> Features:
        assert False

    @abc.abstractmethod
    def make_history_integrator(self) -> HistoryIntegrator:
        assert False


# TODO rename this
class ConcreteHistoryModel(Model):
    def __init__(
        self,
        interaction_model: InteractionModel,
        sequence_model: SequenceModel,
        history_features_function: HistoryFeaturesFunction,
        history_integrator_factory: HistoryIntegratorFactory,
    ):
        super().__init__()
        self.interaction_model = interaction_model
        self.sequence_model = sequence_model
        self.history_features_function = history_features_function
        self.history_integrator_factory = history_integrator_factory

    @property
    def dim(self):
        return self.sequence_model.dim

    def episodic(self, episode: Episode) -> Features:
        interaction_features = self.interaction_model.episodic(episode)
        return self.history_features_function(
            self.sequence_model,
            interaction_features,
        )

    def make_history_integrator(self) -> HistoryIntegrator:
        return self.history_integrator_factory(
            self.interaction_model,
            self.sequence_model,
        )


def make_history_model(
    interaction_model: InteractionModel,
    sequence_model: SequenceModel,
    *,
    memory_size: int,
) -> HistoryModel:
    history_features_function = make_history_features_function(memory_size)
    history_integrator_factory = functools.partial(
        make_history_integrator,
        memory_size=memory_size,
    )

    return ConcreteHistoryModel(
        interaction_model,
        sequence_model,
        history_features_function,
        history_integrator_factory,
    )


class HistoryIntegratorFactory(Protocol):
    def __call__(
        self,
        interaction_model: InteractionModel,
        sequence_model: SequenceModel,
    ) -> HistoryIntegrator:
        ...


def make_history_integrator(
    interaction_model: InteractionModel,
    sequence_model: SequenceModel,
    *,
    memory_size: int,
) -> HistoryIntegrator:
    if memory_size > 0:
        return ReactiveHistoryIntegrator(
            interaction_model,
            sequence_model,
            memory_size=memory_size,
        )

    return FullHistoryIntegrator(interaction_model, sequence_model)
