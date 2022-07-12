from configparser import ConfigParser

from pydantic import BaseModel


class TrainConfig(BaseModel):
    # Training
    gpus: str
    is_overfit: bool
    batch_size: int
    resume_ckpt_path: str
    max_epochs: int
    validate_every_n: int
    num_workers: int
    accumulate_grad_batches: int

    # Dataset
    num_renders: int
    path_to_split: str
    path_to_dataset: str

    # Logging
    logger_type: str
    logging_path: str

    @staticmethod
    def construct_typed_config(ini_config: ConfigParser) -> "TrainConfig":
        """
        Creates typed version of ini configuration file

        :param ini_config: ConfigParser instance
        :return: Instance of TrainConfig
        """

        config = TrainConfig(
            **ini_config["training"],
            **ini_config["dataset"],
            **ini_config["logging"],
        )

        return config
