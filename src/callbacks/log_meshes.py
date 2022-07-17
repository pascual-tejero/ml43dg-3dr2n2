import typing as t
from io import StringIO

import pytorch_lightning as pl
import torch
import wandb
from pytorch3d.ops import cubify
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Subset
from trimesh import Trimesh
from trimesh.exchange.obj import export_obj

from src.model.threedr2n2 import ThreeDeeR2N2


class LogMeshesCallback(Callback):
    NUM_BOARDS = 5

    def __init__(self, log_every=5):
        super().__init__()
        self.state = {"epochs": 0}
        self.log_every = log_every

    @staticmethod
    def _log_mesh(
        logger,
        train_meshes: t.List[Trimesh],
        val_meshes: t.List[Trimesh],
        model: ThreeDeeR2N2,
    ):

        if isinstance(logger, WandbLogger):
            wandb.log(
                {
                    f"test_reconstructions": [
                        wandb.Object3D(
                            StringIO(export_obj(mesh)),
                            file_type="obj",
                            caption=f"Train Object #{i}",
                        )
                        for i, mesh in enumerate(train_meshes)
                    ],
                    f"val_reconstructions": [
                        wandb.Object3D(
                            StringIO(export_obj(mesh)),
                            file_type="obj",
                            caption=f"Val Object #{i}",
                        )
                        for i, mesh in enumerate(val_meshes)
                    ],
                },
            )

        elif isinstance(logger, TensorBoardLogger):
            for i, mesh in enumerate(train_meshes):
                tensorboard = model.logger.experiment
                tensorboard.add_mesh(
                    "train_reconstructions",
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    global_step=model.global_step,
                )

            for i, mesh in enumerate(val_meshes):
                tensorboard = model.logger.experiment
                tensorboard.add_mesh(
                    "val_reconstructions",
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    global_step=model.global_step,
                )

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        model: ThreeDeeR2N2,
        unused=None,
    ):
        self.state["epochs"] += 1

        if self.state["epochs"] % self.log_every == 0:

            # Compute meshes
            model.eval()

            train_dataset = trainer.datamodule.train_dataset
            num_train_boards = min(self.NUM_BOARDS, len(train_dataset))

            train_dataloader = DataLoader(
                Subset(train_dataset, indices=torch.arange(num_train_boards)),
                batch_size=1,
                num_workers=1,
                shuffle=False,
            )
            val_dataloader = trainer.datamodule.val_dataloader()

            train_meshes = []
            for batch in train_dataloader:
                batch = model.transfer_batch_to_device(batch, model.device, 0)
                images = batch["images"].permute(1, 0, 2, 3, 4)

                with torch.no_grad():
                    batch_pred = model.forward(images)
                    batch_meshes = cubify(batch_pred[:, 1], thresh=0.5)

                train_meshes.extend(
                    [
                        Trimesh(
                            faces=faces.detach().cpu().numpy(),
                            vertices=vertices.detach().cpu().numpy(),
                        )
                        for faces, vertices in zip(
                            batch_meshes.faces_list(), batch_meshes.verts_list()
                        )
                    ]
                )

            val_meshes = []
            for batch in val_dataloader:
                batch = model.transfer_batch_to_device(batch, model.device, 0)
                images = batch["images"].permute(1, 0, 2, 3, 4)

                with torch.no_grad():
                    batch_pred = model.forward(images)
                    batch_meshes = cubify(batch_pred[:, 1], thresh=0.5)

                val_meshes.extend(
                    [
                        Trimesh(
                            faces=faces.detach().cpu().numpy(),
                            vertices=vertices.detach().cpu().numpy(),
                        )
                        for faces, vertices in zip(
                            batch_meshes.faces_list(), batch_meshes.verts_list()
                        )
                    ]
                )

            # Check whether we have one logger or multiple
            # and log to all loggers we have
            if isinstance(trainer.logger, LoggerCollection):
                for logger in trainer.logger:
                    self._log_mesh(logger, train_meshes, val_meshes, model)
            else:
                self._log_mesh(trainer.logger, train_meshes, val_meshes, model)

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()
