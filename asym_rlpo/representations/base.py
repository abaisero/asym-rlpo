import abc

import torch.nn as nn


class Representation(nn.Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def dim(self):
        assert False
