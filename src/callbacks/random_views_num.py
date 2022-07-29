import random
import typing as t

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class RandomNumRenders(Callback):
    def __init__(self):
        super().__init__()
        self.state = {"num_views": -1}

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: t.Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        train_dataset = trainer.datamodule.train_dataset
        # If first batch
        if self.state["num_views"] == -1:
            self.state["num_views"] = train_dataset.num_renders

        train_dataset.num_renders = random.randint(1, self.state["num_views"])

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()
