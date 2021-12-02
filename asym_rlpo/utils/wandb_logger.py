import wandb

from asym_rlpo.utils.checkpointing import Serializable


class WandbLogger(Serializable):
    def __init__(self, step=0):
        self.step = step

    def log(self, *args, **kwargs):
        wandb.log(*args, step=self.step, **kwargs)
        self.step += 1

    def state_dict(self):
        return {'step': self.step}

    def load_state_dict(self, data):
        self.step = data['step']
