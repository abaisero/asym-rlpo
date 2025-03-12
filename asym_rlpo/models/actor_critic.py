import torch.nn as nn

from asym_rlpo.models.actor import ActorModel, MemoryReactive_ActorModel
from asym_rlpo.models.critic import (
    CriticModel,
    H_CriticModel,
    HZ_CriticModel,
    HM_CriticModel,
)


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


class NoisyActorCriticModel(nn.Module):
    def __init__(
        self,
        actor_model: ActorModel,
        h_critic_model: H_CriticModel,
        hz_critic_model: HZ_CriticModel,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.h_critic_model = h_critic_model
        self.hz_critic_model = hz_critic_model
