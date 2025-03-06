import logging

from asym_rlpo.policies import Policy

logger = logging.getLogger(__name__)


class Shopping_HardcodedPolicy(Policy):
    def __init__(self, size: int):
        super().__init__()

        self.size = size

    def reset(self, observation):
        observation = observation.item()

        raise NotImplementedError

    def step(self, action, observation):
        observation = observation.item()

        raise NotImplementedError

    def sample_action(self):
        raise NotImplementedError
