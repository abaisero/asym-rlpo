from typing import Dict

import wandb

from asym_rlpo.utils.checkpointing import Serializer

from .logger import DataLogger


class WandbLogger(DataLogger):
    def __init__(self, step: int = 0):
        super().__init__()
        self.step = step

    def log(self, data: Dict):
        wandb.log(data, step=self.step)
        self.step += 1


class WandbLoggerSerializer(Serializer[WandbLogger]):
    def serialize(self, obj: WandbLogger) -> Dict:
        return {'step': obj.step}

    def deserialize(self, obj: WandbLogger, data: Dict):
        obj.step = data['step']
