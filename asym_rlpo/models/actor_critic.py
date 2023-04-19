import torch.nn as nn

from asym_rlpo.models.actor import ActorModel, MemoryReactive_ActorModel
from asym_rlpo.models.critic import CriticModel, HM_CriticModel


class ActorCriticModel(nn.Module):
    def __init__(
        self,
        actor_model: ActorModel,
        critic_model: CriticModel,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.critic_model = critic_model


class MemoryReactive_ActorCriticModel(nn.Module):
    def __init__(
        self,
        actor_model: MemoryReactive_ActorModel,
        critic_model: HM_CriticModel,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.critic_model = critic_model
