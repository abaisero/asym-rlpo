import abc
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode
from asym_rlpo.features import make_history_integrator
from asym_rlpo.policies.base import PartiallyObservablePolicy
from asym_rlpo.q_estimators import Q_Estimator, td0_q_estimator

from ..base import PO_Algorithm_ABC


class PO_A2C_ABC(PO_Algorithm_ABC):
    def behavior_policy(self) -> PartiallyObservablePolicy:
        return BehaviorPolicy(
            self.models,
            truncated_histories=self.truncated_histories,
            truncated_histories_n=self.truncated_histories_n,
        )

    def evaluation_policy(self) -> PartiallyObservablePolicy:
        return EvaluationPolicy(
            self.models,
            truncated_histories=self.truncated_histories,
            truncated_histories_n=self.truncated_histories_n,
        )

    def compute_action_logits(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        history_features = self.compute_history_features(
            models.action_model,
            models.observation_model,
            models.history_model,
            episode.actions,
            episode.observations,
        )
        action_logits = models.policy_model(history_features)
        return action_logits

    @abc.abstractmethod
    def compute_v_values(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:
        assert False

    def actor_losses(  # pylint: disable=too-many-locals
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Optional[Q_Estimator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if q_estimator is None:
            q_estimator = td0_q_estimator

        action_logits = self.compute_action_logits(self.models, episode)
        device = action_logits.device

        with torch.no_grad():
            v_values = self.compute_v_values(self.models, episode)
            q_values = q_estimator(episode.rewards, v_values, discount=discount)

        discounts = discount ** torch.arange(len(episode), device=device)
        action_nlls = -action_logits.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        advantages = q_values.detach() - v_values.detach()
        actor_loss = (discounts * advantages * action_nlls).sum()

        action_dists = torch.distributions.Categorical(logits=action_logits)
        negentropy_loss = -action_dists.entropy().sum()

        return actor_loss, negentropy_loss

    def critic_loss(  # pylint: disable=too-many-locals
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Optional[Q_Estimator] = None,
    ) -> torch.Tensor:

        if q_estimator is None:
            q_estimator = td0_q_estimator

        v_values = self.compute_v_values(self.models, episode)

        with torch.no_grad():
            target_v_values = self.compute_v_values(self.target_models, episode)
            target_q_values = q_estimator(
                episode.rewards, target_v_values, discount=discount
            )

        critic_loss = F.mse_loss(v_values, target_q_values, reduction='sum')

        return critic_loss


class BehaviorPolicy(PartiallyObservablePolicy):
    def __init__(
        self,
        models: nn.ModuleDict,
        *,
        truncated_histories: bool,
        truncated_histories_n: int,
    ):
        super().__init__()
        self.models = models
        self.history_integrator = make_history_integrator(
            models.action_model,
            models.observation_model,
            models.history_model,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    def reset(self, observation):
        self.history_integrator.reset(observation)

    def step(self, action, observation):
        self.history_integrator.step(action, observation)

    def action_logits(self):
        return self.models.policy_model(self.history_integrator.features)

    def po_sample_action(self):
        action_dist = torch.distributions.Categorical(
            logits=self.action_logits()
        )
        return action_dist.sample().item()


class EvaluationPolicy(PartiallyObservablePolicy):
    def __init__(
        self,
        models: nn.ModuleDict,
        *,
        truncated_histories: bool,
        truncated_histories_n: int,
    ):
        super().__init__()
        self.models = models
        self.behavior_policy = BehaviorPolicy(
            models,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )
        self.epsilon: float

    def reset(self, observation):
        self.behavior_policy.reset(observation)

    def step(self, action, observation):
        self.behavior_policy.step(action, observation)

    def po_sample_action(self):
        action_logits = self.behavior_policy.action_logits()
        return (
            torch.distributions.Categorical(logits=action_logits).sample()
            if random.random() < self.epsilon
            else action_logits.argmax()
        ).item()
