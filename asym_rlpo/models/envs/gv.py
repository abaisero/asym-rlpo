import gym.spaces

from asym_rlpo.envs import LatentType
from asym_rlpo.models.embedding import EmbeddingModel
from asym_rlpo.models.empty import EmptyModel
from asym_rlpo.models.factory import ModelFactory
from asym_rlpo.models.gv import GV_Memory_Model, GV_Model
from asym_rlpo.models.history import HistoryModel, make_history_model
from asym_rlpo.models.model import Model
from asym_rlpo.models.sequence import make_sequence_model
from asym_rlpo.models.types import PolicyModule, QModule, VModule
from asym_rlpo.modules.mlp import make_mlp
from asym_rlpo.utils.config import get_config


def _make_qmodule(in_size, out_size) -> QModule:
    return make_mlp([in_size, 512, out_size], ['relu', 'identity'])


def _make_vmodule(in_size) -> VModule:
    return make_mlp([in_size, 512, 1], ['relu', 'identity'])


def _make_policymodule(in_size, out_size) -> PolicyModule:
    return make_mlp([in_size, 512, out_size], ['relu', 'logsoftmax'])


class GVModelFactory(ModelFactory):
    def make_latent_model(self) -> Model:
        config = get_config()

        if self.env.latent_type is LatentType.GV_MEMORY:
            assert isinstance(self.env.latent_space, gym.spaces.Box)
            return GV_Memory_Model(self.env.latent_space, embedding_size=64)

        assert isinstance(self.env.latent_space, gym.spaces.Dict)
        assert config.gv_state_submodels is not None and (
            len(config.gv_state_submodels)
            == len(set(config.gv_state_submodels))
            > 0
        )

        return GV_Model(
            self.env.latent_space,
            config.gv_state_submodels,
            embedding_size=4,
            layers=[64] * config.gv_state_representation_layers,
        )

    def make_action_model(self) -> Model:
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        return EmbeddingModel(self.env.action_space.n, 64)

    def make_observation_model(self) -> Model:
        config = get_config()

        assert isinstance(self.env.observation_space, gym.spaces.Dict)
        assert config.gv_observation_submodels is not None and (
            len(config.gv_observation_submodels)
            == len(set(config.gv_observation_submodels))
            > 0
        )

        return GV_Model(
            self.env.observation_space,
            config.gv_observation_submodels,
            embedding_size=4,
            layers=[64] * config.gv_observation_representation_layers,
        )

    def make_history_model(self) -> HistoryModel:
        interaction_model = self.make_interaction_model()
        sequence_model = make_sequence_model(
            self.history_model,
            interaction_model.dim,
            128,
            attention_num_heads=self.attention_num_heads,
        )
        return make_history_model(
            interaction_model,
            sequence_model,
            memory_size=self.history_model_memory_size,
        )

    def make_policymodule(
        self,
        history_model: HistoryModel,
    ) -> PolicyModule:
        return _make_policymodule(history_model.dim, self.env.action_space.n)

    def make_h_vmodule(self, history_model: HistoryModel) -> VModule:
        return _make_vmodule(history_model.dim)

    def make_hz_vmodule(
        self,
        history_model: HistoryModel,
        latent_model: Model,
    ) -> VModule:
        return _make_vmodule(history_model.dim + latent_model.dim)

    def make_z_vmodule(self, latent_model: Model) -> VModule:
        return _make_vmodule(latent_model.dim)

    def make_ha_qmodule(self, history_model: HistoryModel) -> QModule:
        return _make_qmodule(history_model.dim, self.env.action_space.n)

    def make_hza_qmodule(
        self,
        history_model: HistoryModel,
        latent_model: Model,
    ) -> VModule:
        return _make_qmodule(
            history_model.dim + latent_model.dim,
            self.env.action_space.n,
        )

    def make_za_qmodule(self, latent_model: Model) -> QModule:
        return _make_qmodule(latent_model.dim, self.env.action_space.n)
