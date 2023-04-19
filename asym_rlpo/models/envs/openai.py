import gym.spaces

from asym_rlpo.models.factory import ModelFactory
from asym_rlpo.models.history import HistoryModel, make_history_model
from asym_rlpo.models.identity import IdentityModel
from asym_rlpo.models.model import Model
from asym_rlpo.models.onehot import OneHotModel
from asym_rlpo.models.sequence import make_sequence_model
from asym_rlpo.models.types import PolicyModule, QModule, VModule
from asym_rlpo.modules.mlp import make_mlp


def _make_qmodule(in_size, out_size) -> QModule:
    return make_mlp(
        [in_size, 512, 256, out_size],
        ['relu', 'relu', 'identity'],
    )


def _make_vmodule(in_size) -> VModule:
    return make_mlp(
        [in_size, 512, 256, 1],
        ['relu', 'relu', 'identity'],
    )


def _make_policymodule(in_size, out_size) -> PolicyModule:
    return make_mlp(
        [in_size, 512, 256, out_size],
        ['relu', 'relu', 'logsoftmax'],
    )


class OpenAIModelFactory(ModelFactory):
    def make_latent_model(self) -> Model:
        assert isinstance(self.env.latent_space, gym.spaces.Box)
        return IdentityModel(self.env.latent_space)

    def make_action_model(self) -> Model:
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        return OneHotModel(self.env.action_space)

    def make_observation_model(self) -> Model:
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        return IdentityModel(self.env.observation_space)

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
