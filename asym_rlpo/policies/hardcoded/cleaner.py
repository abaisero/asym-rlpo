import logging

from asym_rlpo.policies import Policy

logger = logging.getLogger(__name__)


class Cleaner_HardcodedPolicy(Policy):
    def __init__(self):
        super().__init__()

    def reset(self, observation):
        raise NotImplementedError

    def step(self, action, observation):
        raise NotImplementedError

    def sample_action(self):
        raise NotImplementedError
