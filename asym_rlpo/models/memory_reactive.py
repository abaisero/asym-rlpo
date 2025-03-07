from __future__ import annotations

import random

import torch

from asym_rlpo.data import Episode, TorchObservation
from asym_rlpo.models.critic import HM_CriticModel
from asym_rlpo.models.history import HistoryIntegrator, HistoryModel
from asym_rlpo.models.memory import MemoryModel
from asym_rlpo.policies import StochasticHistoryPolicy
from asym_rlpo.types import Features, Memory


class MemoryPolicy:
    def __init__(self, hm_critic_model: HM_CriticModel):
        super().__init__()
        self.hm_critic_model = hm_critic_model
        self.history_integrator = (
            hm_critic_model.history_model.make_history_integrator()
        )
        self.memory_integrator = hm_critic_model.memory_model.make_memory_integrator()

        self.memory_features: list[Features] = []
        self.epsilon: float = 1.0

    def reset(self, observation):
        self.history_integrator.reset(observation)
        self.memory_integrator.reset(observation)

        memory_features, _ = self.memory_integrator.sample_features()
        self.memory_features = [memory_features]

    def step(self, action, observation):
        self.history_integrator.step(action, observation)
        self.memory_integrator.step(action, observation)

        memory_features, _ = self.memory_integrator.sample_features()
        self.memory_features.append(memory_features)

    def sample_memory(self) -> tuple[Memory, dict]:
        num_memories = len(self.memory_features)

        if num_memories == 1:
            memory = 0
        elif random.random() < self.epsilon:
            memory = int(torch.randint(num_memories, (1,)).item())
        else:
            history_features, _ = self.history_integrator.sample_features()
            memory_features = torch.stack(self.memory_features)

            values = self.hm_critic_model(
                history_features,
                memory_features,
            )
            memory = values.argmax().item()

        info = {'memory_features': self.memory_features[memory]}
        return memory, info


class MemoryReactiveHistoryModel(HistoryModel):
    def __init__(
        self,
        memory_model: MemoryModel,
        memory_policy: MemoryPolicy,
    ):
        super().__init__()
        self.memory_model = memory_model
        self.memory_policy = memory_policy

    @property
    def dim(self):
        return 2 * self.memory_model.dim

    def episodic(self, episode: Episode) -> Features:
        memory_features = self.memory_model.episodic(
            episode,
            select_memories=False,
        )

        try:
            memories = episode.info['memory']
        except KeyError as exception:
            raise RuntimeError(
                'MemoryModel requires `memory` info field'
            ) from exception

        selected_memory_features = memory_features[memories]
        return torch.cat([selected_memory_features, memory_features], dim=-1)

    def make_history_integrator(self) -> MemoryReactiveHistoryIntegrator:
        return MemoryReactiveHistoryIntegrator(
            self.memory_model,
            self.memory_policy,
        )


class MemoryReactiveHistoryIntegrator(HistoryIntegrator):
    def __init__(
        self,
        memory_model: MemoryModel,
        memory_policy: MemoryPolicy,
    ):
        super().__init__()
        self.memory_model = memory_model
        self.memory_policy = memory_policy
        self.memory_integrator = memory_model.make_memory_integrator()

    def reset(self, observation: TorchObservation):
        self.memory_policy.reset(observation)
        self.memory_integrator.reset(observation)

    def step(self, action: torch.Tensor, observation: TorchObservation):
        self.memory_policy.step(action, observation)
        self.memory_integrator.step(action, observation)

    def sample_features(self) -> tuple[Features, dict]:
        memory, info = self.memory_policy.sample_memory()
        memory_features = info['memory_features']
        info = {'memory': memory}

        (
            reactive_memory_features,
            _,
        ) = self.memory_integrator.sample_features()

        history_features = torch.cat(
            [memory_features, reactive_memory_features],
            dim=-1,
        )
        return history_features, info


class MemoryReactiveHistoryPolicy(StochasticHistoryPolicy):
    history_integrator: MemoryReactiveHistoryIntegrator

    # def __init__(
    #     self,
    #     history_integrator: MemoryReactiveHistoryIntegrator,
    #     policy_function: PolicyFunction,
    # ):
    #     super().__init__(history_integrator)
    #     self.policy_function = policy_function

    # def sample_action(self) -> tuple[Action, dict]:
    #     (
    #         history_features,
    #         info,
    #     ) = self.history_integrator.sample_features()
    #     action_logits = self.policy_function(history_features)
    #     action_dist = torch.distributions.Categorical(logits=action_logits)
    #     action = int(action_dist.sample().item())
    #     info = {**info}
    #     return action, info


# class MemoryReactive_ActorModel(ActorModel):
#     history_model: MemoryReactiveHistoryModel

#     # def __init__(
#     #     self,
#     #     history_model: MemoryReactiveHistoryModel,
#     #     policy_module: PolicyModule,
#     # ):
#     #     super().__init__(history_model, policy_module)
#     #     # self.__history_model = history_model
#     #     # self.__policy_module = policy_module

#     # # def action_logits(self, episode: Episode) -> ActionLogits:
#     # #     history_features = self.__history_model.episodic(episode)
#     # #     return self.__policy_module(history_features)

#     # # def policy_function(self) -> PolicyFunction:
#     # #     return self.__policy_module

#     # # def policy(self) -> MemoryReactiveHistoryPolicy:
#     # #     return MemoryReactiveHistoryPolicy(
#     # #         self.__history_model.make_history_integrator(),
#     # #         self.__policy_module,
#     # #     )
