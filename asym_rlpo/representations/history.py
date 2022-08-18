from __future__ import annotations

from .base import Representation
from .sequence import SequenceRepresentation, make_sequence_model


class HistoryRepresentation(Representation):
    # TODO where is interaction_model used...?

    def __init__(
        self,
        sequence_model: SequenceRepresentation,
    ):
        super().__init__()
        self.sequence_model = sequence_model

    @property
    def dim(self):
        return self.sequence_model.dim

    def forward(self, inputs, *, hidden=None):
        return self.sequence_model(inputs, hidden)


def make_history_representation(
    name: str,
    interaction_model: Representation,
    dim: int,
    **kwargs,
) -> HistoryRepresentation:
    """makes a history representation"""

    sequence_model = make_sequence_model(
        name,
        interaction_model.dim,
        dim,
        **kwargs,
    )
    return HistoryRepresentation(sequence_model)
