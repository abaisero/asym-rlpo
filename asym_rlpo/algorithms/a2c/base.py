from __future__ import annotations

import abc
import random
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Episode
from asym_rlpo.features import HistoryIntegrator
from asym_rlpo.policies import HistoryPolicy
from asym_rlpo.q_estimators import Q_Estimator, td0_q_estimator

from ..base import Algorithm_ABC


class A2C_ABC(Algorithm_ABC):
    def behavior_policy(self) -> ModelPolicy:
        history_integrator = self.make_history_integrator(
            self.models.agent.action_model,
            self.models.agent.observation_model,
            self.models.agent.history_model,
        )
        return ModelPolicy(
            history_integrator,
            self.models.agent.policy_model,
        )

    def evaluation_policy(self) -> EpsilonGreedyModelPolicy:
        history_integrator = self.make_history_integrator(
            self.models.agent.action_model,
            self.models.agent.observation_model,
            self.models.agent.history_model,
        )
        return EpsilonGreedyModelPolicy(
            history_integrator,
            self.models.agent.policy_model,
        )

    def compute_action_logits(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        history_features = self.compute_history_features(
            models.agent.action_model,
            models.agent.observation_model,
            models.agent.history_model,
            episode.actions,
            episode.observations,
        )
        action_logits = models.agent.policy_model(history_features)
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


# policies


# function which maps history features to action-logits
PolicyFunction = Callable[[torch.Tensor], torch.Tensor]


class ModelPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        policy_function: PolicyFunction,
    ):
        super().__init__(history_integrator)
        self.policy_function = policy_function

    def sample_action(self):
        action_logits = self.policy_function(self.history_integrator.features)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        return action_dist.sample().item()


class EpsilonGreedyModelPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        policy_function: PolicyFunction,
    ):
        super().__init__(history_integrator)
        self.policy_function = policy_function

    def sample_action(self):
        action_logits = self.policy_function(self.history_integrator.features)

        if random.random() < self.epsilon:
            action_dist = torch.distributions.Categorical(logits=action_logits)
            return action_dist.sample().item()
        return action_logits.argmax().item()
