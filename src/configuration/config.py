from configparser import ConfigParser

from pydantic import BaseModel


class TrainConfig(BaseModel):
    # Training
    gpus: str
    batch_size: int
    resume_ckpt_path: str
    max_epochs: int
    validate_every_n: int
    num_workers: int
    accumulate_grad_batches: int
    learning_rate: float

    # Dataset
    num_renders: int
    path_to_dataset: str
    train_split: str
    val_split: str

    # Logging
    logger_type: str
    logging_path: str

    # Model
    encoder_decoder_type: str
    conv_rnn3d_type: str
    conv_rnn3d_kernel_size: int

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
            **ini_config["model"],
        )

        return config
