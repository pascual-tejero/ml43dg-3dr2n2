import typing as t
from io import StringIO
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch3d.ops import cubify
from pytorch3d.structures.meshes import Meshes
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

    def _log_to_wandb(
        self,
        train_meshes: t.List[Trimesh],
        target_train_meshes: t.List[Trimesh],
        val_meshes: t.List[Trimesh],
        target_val_meshes: t.List[Trimesh],
    ):
        train_obj_data = [export_obj(mesh) for mesh in train_meshes]
        target_train_obj_data = [export_obj(mesh) for mesh in target_train_meshes]

        val_obj_data = [export_obj(mesh) for mesh in val_meshes]
        target_val_obj_data = [export_obj(mesh) for mesh in target_val_meshes]

        # Log panels for visualisation
        log_data = dict(
            **{
                f"train_reconstructions_{i}": [
                    wandb.Object3D(
                        StringIO(obj_data),
                        file_type="obj",
                        caption=f"Predicted Object #{i}",
                    ),
                    wandb.Object3D(
                        StringIO(target_obj_data),
                        file_type="obj",
                        caption=f"Target Object #{i}",
                    ),
                ]
                for i, (obj_data, target_obj_data) in enumerate(
                    zip(train_obj_data, target_train_obj_data)
                )
            },
            **{
                f"val_reconstructions_{i}": [
                    wandb.Object3D(
                        StringIO(obj_data),
                        file_type="obj",
                        caption=f"Predicted Object #{i}",
                    ),
                    wandb.Object3D(
                        StringIO(target_obj_data),
                        file_type="obj",
                        caption=f"Target Object #{i}",
                    ),
                ]
                for i, (obj_data, target_obj_data) in enumerate(
                    zip(
                        val_obj_data[: self.NUM_BOARDS],
                        target_val_obj_data[: self.NUM_BOARDS],
                    )
                )
            },
        )

        wandb.log(log_data)

        # Save val evaluations as artifacts
        artifact = wandb.Artifact(f"Eval reconstructions", type="evaluation")

        tmp_folder = Path("tmp_artifacts")
        tmp_folder.mkdir(parents=True, exist_ok=True)

        for i, obj_data in enumerate(val_obj_data):
            file_name = tmp_folder / f"val_reconstruction_{i}.obj"
            with open(file_name, "w") as obj_file:
                obj_file.write(obj_data)

            artifact.add_file(str(file_name), is_tmp=True)

        wandb.log_artifact(artifact)

    @staticmethod
    def _log_to_tensor_board(
        model: ThreeDeeR2N2,
        train_meshes: t.List[Trimesh],
        val_meshes: t.List[Trimesh],
    ):
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

    def _log_mesh(
        self,
        logger,
        train_meshes: t.List[Trimesh],
        target_train_meshes: t.List[Trimesh],
        val_meshes: t.List[Trimesh],
        target_val_meshes: t.List[Trimesh],
        model: ThreeDeeR2N2,
    ):

        if isinstance(logger, WandbLogger):
            self._log_to_wandb(
                train_meshes=train_meshes,
                target_train_meshe=target_train_meshes,
                val_meshes=val_meshes,
                target_val_meshes=target_val_meshes,
            )

        elif isinstance(logger, TensorBoardLogger):
            self._log_to_tensor_board(
                model=model, train_meshes=train_meshes, val_meshes=val_meshes
            )

    @staticmethod
    def _pytorch3d_to_trimesh(meshes: Meshes) -> t.List[Trimesh]:
        return [
            Trimesh(
                faces=faces.detach().cpu().numpy(),
                vertices=vertices.detach().cpu().numpy(),
            )
            for faces, vertices in zip(meshes.faces_list(), meshes.verts_list())
        ]

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
            target_train_meshes = []

            for batch in train_dataloader:
                batch = model.transfer_batch_to_device(batch, model.device, 0)
                images = batch["images"].permute(1, 0, 2, 3, 4)
                labels = batch["label"]

                with torch.no_grad():
                    batch_pred = model.forward(images)
                    batch_meshes = cubify(batch_pred[:, 1], thresh=0.5)
                    batch_labels = cubify(labels[:, 1], thresh=0.5)

                train_meshes.extend(self._pytorch3d_to_trimesh(batch_meshes))

                target_train_meshes.extend(self._pytorch3d_to_trimesh(batch_labels))

            val_meshes = []
            target_val_meshes = []
            for batch in val_dataloader:
                batch = model.transfer_batch_to_device(batch, model.device, 0)
                images = batch["images"].permute(1, 0, 2, 3, 4)
                labels = batch["label"]

                with torch.no_grad():
                    batch_pred = model.forward(images)
                    batch_meshes = cubify(batch_pred[:, 1], thresh=0.5)
                    batch_labels = cubify(labels[:, 1], thresh=0.5)

                val_meshes.extend(self._pytorch3d_to_trimesh(batch_meshes))
                target_val_meshes.extend(self._pytorch3d_to_trimesh(batch_labels))

            # Check whether we have one logger or multiple
            # and log to all loggers we have
            if isinstance(trainer.logger, LoggerCollection):
                for logger in trainer.logger:
                    self._log_mesh(
                        logger=logger,
                        train_meshes=train_meshes,
                        target_train_meshes=target_train_meshes,
                        val_meshes=val_meshes,
                        target_val_meshes=target_val_meshes,
                        model=model,
                    )
            else:
                self._log_mesh(
                    logger=trainer.logger,
                    train_meshes=train_meshes,
                    target_train_meshes=target_train_meshes,
                    val_meshes=val_meshes,
                    target_val_meshes=target_val_meshes,
                    model=model,
                )

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()
