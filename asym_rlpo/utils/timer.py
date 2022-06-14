import time
from typing import Dict

from asym_rlpo.utils.checkpointing import Serializer


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    @property
    def seconds(self) -> float:
        return time.time() - self.start

    @property
    def hours(self) -> float:
        return self.seconds / 3600


class TimerSerializer(Serializer[Timer]):
    def serialize(self, obj: Timer) -> Dict:
        return {'seconds': obj.seconds}

    def deserialize(self, obj: Timer, data: Dict):
        obj.start = time.time() - data['seconds']
