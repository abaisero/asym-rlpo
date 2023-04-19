import torch.nn as nn

from asym_rlpo.data import Episode
from asym_rlpo.models.history import HistoryModel
from asym_rlpo.models.types import PolicyModule
from asym_rlpo.models.memory_reactive import MemoryReactiveHistoryModel
from asym_rlpo.policies import HistoryPolicy, StochasticHistoryPolicy
from asym_rlpo.types import ActionLogits, PolicyFunction


class ActorModel(nn.Module):
    def __init__(
        self,
        history_model: HistoryModel,
        policy_module: PolicyModule,
    ):
        super().__init__()
        self.history_model = history_model
        self.policy_module = policy_module

    def action_logits(self, episode: Episode) -> ActionLogits:
        history_features = self.history_model.episodic(episode)
        return self.policy_module(history_features)

    def policy_function(self) -> PolicyFunction:
        return self.policy_module

    def policy(self) -> HistoryPolicy:
        return StochasticHistoryPolicy(
            self.history_model.make_history_integrator(),
            self.policy_function(),
        )


class MemoryReactive_ActorModel(ActorModel):
    history_model: MemoryReactiveHistoryModel
