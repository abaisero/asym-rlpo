from asym_rlpo.features import HistoryIntegrator
from asym_rlpo.policies.policy import Policy


class HistoryPolicy(Policy):
    def __init__(self, history_integrator: HistoryIntegrator):
        super().__init__()
        self.history_integrator = history_integrator

    def reset(self, observation):
        self.history_integrator.reset(observation)

    def step(self, action, observation):
        self.history_integrator.step(action, observation)
