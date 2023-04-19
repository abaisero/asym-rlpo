import wandb

from .logger import DataLogger


class WandbLogger(DataLogger):
    def __init__(self):
        super().__init__()
        self.__step = 0

    def log(self, data: dict, *, commit: bool = True):
        wandb.log(data, step=self.__step, commit=commit)
        if commit:
            self.__step += 1

    def commit(self):
        wandb.log({}, step=self.__step)
        self.__step += 1
