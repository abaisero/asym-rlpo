from typing import Dict

import wandb

from asym_rlpo.utils.checkpointing import Serializer


class WandbLogger:
    def __init__(self, step=0):
        self.step = step

    def log(self, *args, **kwargs):
        wandb.log(*args, step=self.step, **kwargs)
        self.step += 1


class WandbLoggerSerializer(Serializer[WandbLogger]):
    def serialize(self, obj: WandbLogger) -> Dict:
        return {'step': obj.step}

    def deserialize(self, obj: WandbLogger, data: Dict):
        obj.step = data['step']
