import typing as t

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .convRNN3D import ConvGRU3D, ConvLSTM3D
from .decoder import Decoder
from .encoder import Encoder
from .utils import initialize_tensor


class ThreeDeeR2N2(pl.LightningModule):
    def __init__(
        self,
        encoder_decoder_type: str,
        convRNN3D_type: str,
        convRNN3D_kernel_size: int,
        batch_size: int,
        learning_rate: float = 0.001,
    ):
        super(ThreeDeeR2N2, self).__init__()
        self.encoder_decoder_type = encoder_decoder_type
        self.convRNN3D_kernel_size = convRNN3D_type
        self.convRNN3D_type = convRNN3D_type
        self.convRNN3D_kernel_size = convRNN3D_kernel_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.image_shape = (3, 127, 127)
        self.input_shape = (self.batch_size, *self.image_shape)

        self.grid_convRNN3D = 4
        self.hidden_size = 128
        self.h_shape = (
            self.hidden_size,
            self.grid_convRNN3D,
            self.grid_convRNN3D,
            self.grid_convRNN3D,
        )  # Include batch size in forward pass

        self.encoder, self.decoder, self.convRNN3D = None, None, None

        self.initialize_encoder(encoder_decoder_type)
        self.initialize_decoder(encoder_decoder_type)
        self.initialize_convRNN3d(convRNN3D_type, convRNN3D_kernel_size)

        self.loss = nn.CrossEntropyLoss()

    def initialize_encoder(self, type):
        if type.lower() not in ["simple", "residual"]:
            raise Exception("Type Error: Encoder")
        print(f"Initializing {type.lower()} encoder")
        self.encoder = Encoder(type.lower())

    def initialize_decoder(self, type):
        if type.lower() not in ["simple", "residual"]:
            raise Exception("Type Error: Decoder")
        print(f"Initializing {type.lower()} decoder")
        self.decoder = Decoder(type.lower())

    def initialize_convRNN3d(self, type, kernel_size):
        if type.lower() not in ["lstm", "gru"]:
            raise Exception("Type Error: 3D Convolutional RNN")
        if kernel_size not in [1, 3]:
            raise Exception("Value Error: Kernel size of 3D Convolutional RNN")
        if type == "gru":
            print(f"Initializing ConvGRU3D with kernel size {kernel_size}")
            self.convRNN3D = ConvGRU3D(
                fan_in=1024,
                hidden_size=self.hidden_size,
                grid_size=self.grid_convRNN3D,
                kernel_size=kernel_size,
            )
        else:
            print(f"Initializing ConvLSTM3D with kernel size {kernel_size}")
            self.convRNN3D = ConvLSTM3D(
                fan_in=1024,
                hidden_size=self.hidden_size,
                grid_size=self.grid_convRNN3D,
                kernel_size=kernel_size,
            )

    def forward(self, X):
        if self.encoder is None:
            raise Exception("The encoder is not initialized!")
        if self.convRNN3D is None:
            raise Exception("The convolutional recurrent network is not initialized!")
        if self.decoder is None:
            raise Exception("The decoder is not initialized!")

        if self.convRNN3D_type.lower() == "gru":
            batch_size = X.shape[1]
            h, u = initialize_tensor((batch_size, *self.h_shape)), initialize_tensor(
                (batch_size, *self.h_shape)
            )
            u_list = []

            """
            x is the input and the size of x is (num_views, batch_size, channels, heights, widths).
            h and u is the hidden state and activation of last time step respectively.
            The following loop computes the forward pass of the whole network. 
            """
            for time_step in range(X.shape[0]):
                encoder_out = self.encoder(X[time_step])
                convRNN3D_out, update_gate = self.convRNN3D(encoder_out, h)
                h = convRNN3D_out
                u = update_gate
                u_list.append(u)
            decoder_out = self.decoder(h)
        else:
            batch_size = X.shape[1]
            h, c = initialize_tensor((batch_size, *self.h_shape)), initialize_tensor(
                (batch_size, *self.h_shape)
            )
            c_list = []
            for time_step in range(X.shape[0]):
                encoder_out = self.encoder(X[time_step])
                convRNN3D_out, cell_gate = self.convRNN3D(encoder_out, (h, c))
                h = convRNN3D_out
                c_list.append(cell_gate)
            decoder_out = self.decoder(h)

        return decoder_out

    # region Pytorch Lightning

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.learning_rate, weight_decay=0.00005
        )

        return optimizer

    def training_step(self, batch: t.Dict[str, t.Any], batch_idx):
        x = batch["images"].permute(1, 0, 2, 3, 4)
        y = batch["label"]

        prediction = self.forward(x)
        train_loss = self.loss(prediction, y)

        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch: t.Dict[str, t.Any], batch_idx):
        x = batch["images"].permute(1, 0, 2, 3, 4)
        y = batch["label"]

        prediction = self.forward(x)
        val_loss = self.loss(prediction, y)

        self.log("val_loss", val_loss)
        return val_loss

    def transfer_batch_to_device(
        self,
        batch: t.Dict[str, t.Any],
        device: torch.device,
        dataloader_idx: int,
    ) -> t.Dict[str, torch.Tensor]:
        batch["images"] = batch["images"].to(device)
        batch["label"] = batch["label"].to(device)

        return batch

    # endregion
