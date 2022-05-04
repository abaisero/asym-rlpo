import abc

import gym

from asym_rlpo.features import HistoryIntegrator


class Policy(metaclass=abc.ABCMeta):
    def __init__(self):
        self.epsilon = 1.0

    @abc.abstractmethod
    def reset(self, observation):
        assert False

    @abc.abstractmethod
    def step(self, action, observation):
        assert False

    @abc.abstractmethod
    def sample_action(self, state):
        assert False


class FullyObservablePolicy(Policy):
    def reset(self, observation):
        pass

    def step(self, action, observation):
        pass

    def sample_action(self, state):
        return self.fo_sample_action(state)

    @abc.abstractmethod
    def fo_sample_action(self, state):
        assert False


class PartiallyObservablePolicy(Policy):
    def sample_action(self, state):
        return self.po_sample_action()

    @abc.abstractmethod
    def po_sample_action(self):
        assert False


class RandomPolicy(Policy):
    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def reset(self, observation):
        pass

    def step(self, action, observation):
        pass

    def sample_action(self, state):
        return self.action_space.sample()


class HistoryPolicy(PartiallyObservablePolicy):
    def __init__(self, history_integrator: HistoryIntegrator):
        super().__init__()
        self.history_integrator = history_integrator

    def reset(self, observation):
        self.history_integrator.reset(observation)

    def step(self, action, observation):
        self.history_integrator.step(action, observation)
