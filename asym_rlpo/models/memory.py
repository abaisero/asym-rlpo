from asym_rlpo.data import Episode
from asym_rlpo.models.history import compute_reactive_history_features
from asym_rlpo.models.history.reactive import ReactiveHistoryIntegrator
from asym_rlpo.models.interaction import InteractionModel
from asym_rlpo.models.model import Model
from asym_rlpo.models.sequence import SequenceModel
from asym_rlpo.types import Features


# TODO how to handle this better?
class ReactiveMemoryIntegrator(ReactiveHistoryIntegrator):
    pass


class MemoryModel(Model):
    def __init__(
        self,
        interaction_model: InteractionModel,
        sequence_model: SequenceModel,
        *,
        memory_size: int,
    ):
        if memory_size <= 0:
            raise ValueError(f'invalid {memory_size=}')

        super().__init__()
        self.interaction_model = interaction_model
        self.sequence_model = sequence_model
        self.memory_size = memory_size

    @property
    def dim(self):
        return self.sequence_model.dim

    def episodic(self, episode: Episode, *, select_memories: bool):
        interaction_features = self.interaction_model.episodic(episode)
        memory_features = self(interaction_features)

        if not select_memories:
            return memory_features

        try:
            memories = episode.info['memory']
        except KeyError as exception:
            raise RuntimeError(
                'MemoryModel requires `memory` info field when `select_memories`=True'
            ) from exception

        return memory_features[memories]

    def forward(self, interaction_features: Features) -> Features:
        return compute_reactive_history_features(
            self.sequence_model,
            interaction_features,
            memory_size=self.memory_size,
        )

    def make_memory_integrator(self) -> ReactiveMemoryIntegrator:
        return ReactiveMemoryIntegrator(
            self.interaction_model,
            self.sequence_model,
            memory_size=self.memory_size,
        )
