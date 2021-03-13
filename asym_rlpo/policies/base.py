import abc


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self, observation):
        assert False

    @abc.abstractmethod
    def step(self, action, observation):
        assert False

    @abc.abstractmethod
    def sample_action(self):
        assert False
