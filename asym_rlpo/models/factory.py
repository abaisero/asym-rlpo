import abc

from asym_rlpo.envs import Environment
from asym_rlpo.models.actor import ActorModel
from asym_rlpo.models.critic import (
    CriticModel,
    H_CriticModel,
    HZ_CriticModel,
    Z_CriticModel,
)
from asym_rlpo.models.history import HistoryModel
from asym_rlpo.models.interaction import InteractionModel
from asym_rlpo.models.model import FeatureModel, Model
from asym_rlpo.models.qmodel import QhaModel, QhzaModel, QModel, QzaModel
from asym_rlpo.models.types import (
    CriticType,
    PolicyModule,
    QModelType,
    QModule,
    VModule,
)


class ModelFactory(metaclass=abc.ABCMeta):
    def __init__(self, env: Environment):
        self.env = env

        self.history_model: str
        self.attention_num_heads: int | None = None
        self.history_model_memory_size: int

    @abc.abstractmethod
    def make_latent_model(self) -> FeatureModel:
        assert False

    @abc.abstractmethod
    def make_action_model(self) -> FeatureModel:
        assert False

    @abc.abstractmethod
    def make_observation_model(self) -> FeatureModel:
        assert False

    def make_interaction_model(self) -> InteractionModel:
        return InteractionModel(
            self.make_action_model(),
            self.make_observation_model(),
        )

    def make_actor_model(self) -> ActorModel:
        history_model = self.make_history_model()
        policy_module = self.make_policymodule(history_model)
        return ActorModel(history_model, policy_module)

    def make_critic_model(self, critic_type: CriticType) -> CriticModel:
        if critic_type is CriticType.H:
            history_model = self.make_history_model()
            vmodule = self.make_h_vmodule(history_model)
            return H_CriticModel(history_model, vmodule)

        if critic_type is CriticType.HZ:
            history_model = self.make_history_model()
            latent_model = self.make_latent_model()
            vmodule = self.make_hz_vmodule(history_model, latent_model)
            return HZ_CriticModel(history_model, latent_model, vmodule)

        if critic_type is CriticType.Z:
            latent_model = self.make_latent_model()
            vmodule = self.make_z_vmodule(latent_model)
            return Z_CriticModel(latent_model, vmodule)

        raise ValueError(f'invalid critic type `{critic_type}`')

    @abc.abstractmethod
    def make_history_model(self) -> HistoryModel:
        assert False

    @abc.abstractmethod
    def make_policymodule(
        self,
        history_model: HistoryModel,
    ) -> PolicyModule:
        assert False

    @abc.abstractmethod
    def make_h_vmodule(self, history_model: HistoryModel) -> VModule:
        assert False

    @abc.abstractmethod
    def make_hz_vmodule(
        self,
        history_model: HistoryModel,
        latent_model: Model,
    ) -> VModule:
        assert False

    @abc.abstractmethod
    def make_z_vmodule(self, latent_model: Model) -> VModule:
        assert False

    def make_qmodel(self, qmodel_type: QModelType) -> QModel:
        if qmodel_type is QModelType.H:
            return self.make_qha_model()

        if qmodel_type is QModelType.HZ:
            return self.make_qhza_model()

        if qmodel_type is QModelType.Z:
            return self.make_qza_model()

        raise ValueError(f'invalid qmodel type `{qmodel_type}`')

    def make_qha_model(self) -> QhaModel:
        history_model = self.make_history_model()
        qmodule = self.make_ha_qmodule(history_model)
        return QhaModel(self.env.action_space, history_model, qmodule)

    def make_qhza_model(self) -> QhzaModel:
        history_model = self.make_history_model()
        latent_model = self.make_latent_model()
        qmodule = self.make_hza_qmodule(history_model, latent_model)
        return QhzaModel(
            self.env.action_space,
            history_model,
            latent_model,
            qmodule,
        )

    def make_qza_model(self) -> QzaModel:
        latent_model = self.make_latent_model()
        qmodule = self.make_za_qmodule(latent_model)
        return QzaModel(self.env.action_space, latent_model, qmodule)

    @abc.abstractmethod
    def make_ha_qmodule(self, history_model: HistoryModel) -> QModule:
        assert False

    @abc.abstractmethod
    def make_hza_qmodule(
        self,
        history_model: HistoryModel,
        latent_model: Model,
    ) -> VModule:
        assert False

    @abc.abstractmethod
    def make_za_qmodule(self, latent_model: Model) -> QModule:
        assert False
