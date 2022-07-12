import argparse
import wandb
import typing as t
from pytorch_lightning import loggers as pl_loggers, Trainer
from configparser import ConfigParser

from pytorch_lightning import seed_everything

from src.configuration.config import TrainConfig
from src.data.shapenet import ShapeNetDataModule


def train_loop(config: TrainConfig, resume_from: t.Optional[str], run_id: t.Optional[str]) -> None:
    # Create DataModule
    datamodule = ShapeNetDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_renders=config.num_renders,
        path_to_split=config.path_to_split,
        path_to_dataset=config.path_to_dataset
    )
    datamodule.setup()

    # Instantiate model
    model = ...

    # Create logger
    if config.logger_type == "wandb":
        wandb.init(project="CNPM")
        logger = pl_loggers.WandbLogger(project="CNPM", log_model="all")
        logger.watch(model)
    elif config.logger_type == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(save_dir=config.logging_path)
    else:
        logger = None

    # Training
    trainer = Trainer(
        gpus=config.gpus,
        check_val_every_n_epoch=config.validate_every_n,
        default_root_dir=config.resume_ckpt_path,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=config.max_epochs,
        amp_backend="native",
        resume_from_checkpoint=resume_from,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model trainer")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
        default="./configuration/train.ini"
    )
    parser.add_argument(
        "--resume_from",
        nargs="?",
        type=str,
        help="path to resume model",
        default="",
    )
    parser.add_argument(
        "--seed",
        nargs="?",
        type=int,
        help="seed for random",
        default=42,
    )
    parser.add_argument(
        "--run_id",
        nargs="?",
        type=str,
        help="wandb run id",
        default="",
    )
    args = parser.parse_args()

    # make it deterministic
    seed_everything(args.seed)

    # Reading configuration from ini file
    print(f"Reading training configuration file..")
    ini_config = ConfigParser()
    ini_config.read(args.config)

    # Unpack config to typed config class
    config = TrainConfig.construct_typed_config(ini_config)

    # Run training process
    print(f"Running training process..")
    try:
        train_loop(config, args.resume_from, args.run_id)
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
