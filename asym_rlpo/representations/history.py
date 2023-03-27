from __future__ import annotations

from typing import Optional
from .base import Representation
from .sequence import SequenceRepresentation, make_sequence_model


class HistoryRepresentation(Representation):
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
        return self.sequence_model(inputs, hidden=hidden)


def make_history_representation(
    name: str,
    interaction_model: Representation,
    dim: int,
    *,
    num_heads: Optional[int] = None,
    **kwargs,
) -> HistoryRepresentation:
    """makes a history representation"""

    sequence_model = make_sequence_model(
        name,
        interaction_model.dim,
        dim,
        num_heads = num_heads,
        **kwargs,
    )
    return HistoryRepresentation(sequence_model)
