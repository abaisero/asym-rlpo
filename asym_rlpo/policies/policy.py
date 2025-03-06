import abc


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
    def sample_action(self):
        assert False
